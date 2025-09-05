"""
Enhanced Queue with Deduplication, Backpressure, and Replay

Provides at-least-once delivery with tenant isolation and priority handling.
"""
import heapq
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List, Set, Tuple
from threading import Lock, Condition
import hashlib
import logging

logger = logging.getLogger(__name__)

class Priority(Enum):
    """Message priority levels"""
    HIGH = 0  # 0-3
    NORMAL = 1  # 4-6  
    LOW = 2  # 7-9
    
    @classmethod
    def from_int(cls, value: int) -> 'Priority':
        if 0 <= value <= 3:
            return cls.HIGH
        elif 4 <= value <= 6:
            return cls.NORMAL
        else:
            return cls.LOW

@dataclass(order=True)
class QueueMessage:
    """Message in the execution queue"""
    priority_order: Tuple[int, float] = field(compare=True)  # (priority_level, timestamp)
    
    # Message attributes
    message_id: str = field(compare=False)
    tenant_id: str = field(compare=False)
    idempotency_key: str = field(compare=False)
    priority: int = field(compare=False)  # 0-9
    retry_count: int = field(compare=False, default=0)
    trace_id: str = field(compare=False, default="")
    correlation_id: str = field(compare=False, default="")
    submitted_at: datetime = field(compare=False, default_factory=datetime.utcnow)
    deadline: Optional[datetime] = field(compare=False, default=None)
    
    # Payload
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)
    
    # Queue management
    visibility_timeout: Optional[datetime] = field(compare=False, default=None)
    receipt_handle: Optional[str] = field(compare=False, default=None)
    dequeue_count: int = field(compare=False, default=0)
    
    def __post_init__(self):
        # Set priority order for heap queue
        if not hasattr(self, 'priority_order') or self.priority_order is None:
            priority_level = Priority.from_int(self.priority).value
            timestamp = self.submitted_at.timestamp()
            self.priority_order = (priority_level, timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['submitted_at'] = self.submitted_at.isoformat()
        if self.deadline:
            d['deadline'] = self.deadline.isoformat()
        if self.visibility_timeout:
            d['visibility_timeout'] = self.visibility_timeout.isoformat()
        del d['priority_order']  # Internal field
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueueMessage':
        data = data.copy()
        data['submitted_at'] = datetime.fromisoformat(data['submitted_at'])
        if data.get('deadline'):
            data['deadline'] = datetime.fromisoformat(data['deadline'])
        if data.get('visibility_timeout'):
            data['visibility_timeout'] = datetime.fromisoformat(data['visibility_timeout'])
        return cls(**data)

class BackpressurePolicy:
    """Backpressure configuration per tenant"""
    def __init__(self, 
                 max_inflight: int = 100,
                 max_queue_depth: int = 10000):
        self.max_inflight = max_inflight
        self.max_queue_depth = max_queue_depth

class ExecutionQueue:
    """
    Priority queue with tenant isolation, deduplication, and backpressure.
    
    Features:
    - Tenant-isolated queues with per-tenant backpressure
    - Message deduplication via idempotency keys
    - Priority-based ordering (0=highest, 9=lowest)
    - Visibility timeout for at-least-once delivery
    - Dead letter queue for failed messages
    """
    
    def __init__(self,
                 default_visibility_timeout: int = 300,  # 5 minutes
                 max_receive_count: int = 3,
                 backpressure_policy: Optional[BackpressurePolicy] = None):
        
        self.default_visibility_timeout = timedelta(seconds=default_visibility_timeout)
        self.max_receive_count = max_receive_count
        self.backpressure_policy = backpressure_policy or BackpressurePolicy()
        
        # Tenant-isolated queues
        self._queues: Dict[str, List[QueueMessage]] = defaultdict(list)
        self._inflight: Dict[str, Dict[str, QueueMessage]] = defaultdict(dict)
        self._dlq: Dict[str, List[QueueMessage]] = defaultdict(list)
        
        # Deduplication cache
        self._seen_keys: Dict[str, Set[str]] = defaultdict(set)  # tenant_id -> set of idempotency_keys
        self._key_to_message: Dict[str, Dict[str, str]] = defaultdict(dict)  # tenant_id -> {idem_key: message_id}
        
        # Thread safety
        self._lock = Lock()
        self._condition = Condition(self._lock)
        
        # Stats
        self._stats = defaultdict(lambda: defaultdict(int))
    
    def enqueue(self, message: QueueMessage) -> bool:
        """
        Add message to queue with backpressure and deduplication.
        
        Returns:
            True if enqueued, False if rejected (backpressure or duplicate)
        """
        with self._lock:
            tenant_id = message.tenant_id
            
            # Check deduplication
            if message.idempotency_key in self._seen_keys[tenant_id]:
                existing_msg_id = self._key_to_message[tenant_id].get(message.idempotency_key)
                logger.info(f"Duplicate message rejected: idempotency_key={message.idempotency_key}, "
                          f"existing_message={existing_msg_id}")
                self._stats[tenant_id]['duplicates'] += 1
                return False
            
            # Check backpressure - queue depth
            current_depth = len(self._queues[tenant_id]) + len(self._inflight[tenant_id])
            if current_depth >= self.backpressure_policy.max_queue_depth:
                logger.warning(f"Queue depth limit reached for tenant {tenant_id}: {current_depth}")
                self._stats[tenant_id]['backpressure_rejections'] += 1
                return False
            
            # Add to queue
            heapq.heappush(self._queues[tenant_id], message)
            
            # Track for deduplication
            self._seen_keys[tenant_id].add(message.idempotency_key)
            self._key_to_message[tenant_id][message.idempotency_key] = message.message_id
            
            # Update stats
            self._stats[tenant_id]['enqueued'] += 1
            self._stats[tenant_id][f'priority_{message.priority}'] += 1
            
            # Signal waiting consumers
            self._condition.notify()
            
            logger.debug(f"Message enqueued: tenant={tenant_id}, id={message.message_id}, "
                       f"priority={message.priority}")
            return True
    
    def dequeue(self, tenant_id: str, 
                max_messages: int = 1,
                wait_timeout: float = 0) -> List[QueueMessage]:
        """
        Dequeue messages for a tenant with visibility timeout.
        
        Args:
            tenant_id: Tenant to dequeue for
            max_messages: Maximum messages to return
            wait_timeout: Seconds to wait for messages (0 = no wait)
        
        Returns:
            List of messages (may be empty)
        """
        deadline = time.time() + wait_timeout if wait_timeout > 0 else None
        
        with self._lock:
            messages = []
            
            # Wait for messages if requested
            while wait_timeout > 0 and not self._queues[tenant_id]:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            
            # Check backpressure - inflight limit
            inflight_count = len(self._inflight[tenant_id])
            if inflight_count >= self.backpressure_policy.max_inflight:
                logger.warning(f"Max inflight reached for tenant {tenant_id}: {inflight_count}")
                return []
            
            # Process expired visibility timeouts first
            self._process_visibility_timeouts(tenant_id)
            
            # Dequeue up to max_messages
            available_slots = min(
                max_messages,
                self.backpressure_policy.max_inflight - inflight_count
            )
            
            for _ in range(available_slots):
                if not self._queues[tenant_id]:
                    break
                
                message = heapq.heappop(self._queues[tenant_id])
                
                # Set visibility timeout
                message.visibility_timeout = datetime.utcnow() + self.default_visibility_timeout
                message.receipt_handle = f"rcpt_{hashlib.md5(f'{message.message_id}_{time.time()}'.encode()).hexdigest()[:12]}"
                message.dequeue_count += 1
                
                # Move to inflight
                self._inflight[tenant_id][message.receipt_handle] = message
                messages.append(message)
                
                # Update stats
                self._stats[tenant_id]['dequeued'] += 1
            
            logger.debug(f"Dequeued {len(messages)} messages for tenant {tenant_id}")
            return messages
    
    def acknowledge(self, tenant_id: str, receipt_handle: str) -> bool:
        """
        Acknowledge successful message processing.
        
        Returns:
            True if acknowledged, False if not found
        """
        with self._lock:
            if receipt_handle not in self._inflight[tenant_id]:
                logger.warning(f"Receipt handle not found: {receipt_handle}")
                return False
            
            message = self._inflight[tenant_id].pop(receipt_handle)
            
            # Remove from deduplication cache (allow reuse after success)
            self._seen_keys[tenant_id].discard(message.idempotency_key)
            self._key_to_message[tenant_id].pop(message.idempotency_key, None)
            
            # Update stats
            self._stats[tenant_id]['acknowledged'] += 1
            
            logger.debug(f"Message acknowledged: tenant={tenant_id}, id={message.message_id}")
            return True
    
    def nack(self, tenant_id: str, receipt_handle: str, 
             retry_delay: Optional[int] = None) -> bool:
        """
        Negative acknowledgment - return message to queue or DLQ.
        
        Args:
            tenant_id: Tenant ID
            receipt_handle: Receipt handle from dequeue
            retry_delay: Optional delay in seconds before retry
        
        Returns:
            True if requeued, False if moved to DLQ
        """
        with self._lock:
            if receipt_handle not in self._inflight[tenant_id]:
                logger.warning(f"Receipt handle not found for NACK: {receipt_handle}")
                return False
            
            message = self._inflight[tenant_id].pop(receipt_handle)
            message.retry_count += 1
            
            # Check if should go to DLQ
            if message.retry_count >= self.max_receive_count:
                self._dlq[tenant_id].append(message)
                
                # Keep in dedup cache for DLQ items
                self._stats[tenant_id]['dlq'] += 1
                
                logger.warning(f"Message moved to DLQ: tenant={tenant_id}, id={message.message_id}, "
                             f"retries={message.retry_count}")
                return False
            
            # Apply retry delay if specified
            if retry_delay:
                message.visibility_timeout = datetime.utcnow() + timedelta(seconds=retry_delay)
                # Keep in inflight with new timeout
                new_handle = f"retry_{hashlib.md5(f'{message.message_id}_{time.time()}'.encode()).hexdigest()[:12]}"
                message.receipt_handle = new_handle
                self._inflight[tenant_id][new_handle] = message
            else:
                # Return to queue immediately
                message.visibility_timeout = None
                message.receipt_handle = None
                heapq.heappush(self._queues[tenant_id], message)
                self._condition.notify()
            
            self._stats[tenant_id]['nacked'] += 1
            
            logger.info(f"Message NACKed: tenant={tenant_id}, id={message.message_id}, "
                       f"retry={message.retry_count}/{self.max_receive_count}")
            return True
    
    def replay(self, tenant_id: str, message_id: str) -> bool:
        """
        Replay a specific message from history.
        
        Returns:
            True if message found and replayed
        """
        with self._lock:
            # Check inflight
            for receipt, msg in self._inflight[tenant_id].items():
                if msg.message_id == message_id:
                    # Reset and requeue
                    msg.visibility_timeout = None
                    msg.receipt_handle = None
                    msg.retry_count = 0
                    self._inflight[tenant_id].pop(receipt)
                    heapq.heappush(self._queues[tenant_id], msg)
                    self._condition.notify()
                    logger.info(f"Replayed inflight message: {message_id}")
                    return True
            
            # Check DLQ
            for i, msg in enumerate(self._dlq[tenant_id]):
                if msg.message_id == message_id:
                    # Reset and requeue
                    msg.retry_count = 0
                    msg.visibility_timeout = None
                    msg.receipt_handle = None
                    self._dlq[tenant_id].pop(i)
                    heapq.heappush(self._queues[tenant_id], msg)
                    self._condition.notify()
                    logger.info(f"Replayed DLQ message: {message_id}")
                    return True
            
            logger.warning(f"Message not found for replay: {message_id}")
            return False
    
    def get_dlq(self, tenant_id: str) -> List[QueueMessage]:
        """Get all messages in DLQ for a tenant"""
        with self._lock:
            return list(self._dlq[tenant_id])
    
    def purge_dlq(self, tenant_id: str, older_than_days: int = 30) -> int:
        """
        Purge old messages from DLQ.
        
        Returns:
            Number of messages purged
        """
        with self._lock:
            cutoff = datetime.utcnow() - timedelta(days=older_than_days)
            original_count = len(self._dlq[tenant_id])
            
            self._dlq[tenant_id] = [
                msg for msg in self._dlq[tenant_id]
                if msg.submitted_at > cutoff
            ]
            
            purged = original_count - len(self._dlq[tenant_id])
            if purged > 0:
                logger.info(f"Purged {purged} messages from DLQ for tenant {tenant_id}")
            
            return purged
    
    def _process_visibility_timeouts(self, tenant_id: str):
        """Return messages with expired visibility timeout to queue"""
        now = datetime.utcnow()
        expired = []
        
        for receipt, msg in list(self._inflight[tenant_id].items()):
            if msg.visibility_timeout and msg.visibility_timeout <= now:
                expired.append(receipt)
        
        for receipt in expired:
            msg = self._inflight[tenant_id].pop(receipt)
            msg.visibility_timeout = None
            msg.receipt_handle = None
            heapq.heappush(self._queues[tenant_id], msg)
            self._stats[tenant_id]['visibility_timeout'] += 1
            
            logger.debug(f"Message visibility timeout expired: {msg.message_id}")
    
    def get_stats(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get queue statistics"""
        with self._lock:
            if tenant_id:
                return {
                    "tenant_id": tenant_id,
                    "queue_depth": len(self._queues[tenant_id]),
                    "inflight": len(self._inflight[tenant_id]),
                    "dlq_depth": len(self._dlq[tenant_id]),
                    "stats": dict(self._stats[tenant_id])
                }
            else:
                # Global stats
                total_queued = sum(len(q) for q in self._queues.values())
                total_inflight = sum(len(i) for i in self._inflight.values())
                total_dlq = sum(len(d) for d in self._dlq.values())
                
                return {
                    "total_queued": total_queued,
                    "total_inflight": total_inflight,
                    "total_dlq": total_dlq,
                    "active_tenants": len(self._queues),
                    "tenant_stats": {
                        tid: self.get_stats(tid) for tid in self._queues.keys()
                    }
                }