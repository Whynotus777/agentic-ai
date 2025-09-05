"""
Tests for Execution Queue Semantics

Tests deduplication, backpressure, replay, and DLQ functionality.
"""
import pytest
import time
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid

from execution.queue import ExecutionQueue, QueueMessage, Priority, BackpressurePolicy

class TestQueueSemantics:
    """Test suite for ExecutionQueue semantics"""
    
    @pytest.fixture
    def queue(self):
        """Create queue instance with test configuration"""
        return ExecutionQueue(
            default_visibility_timeout=2,  # 2 seconds for faster tests
            max_receive_count=3,
            backpressure_policy=BackpressurePolicy(
                max_inflight=10,
                max_queue_depth=100
            )
        )
    
    @pytest.fixture
    def sample_message(self):
        """Create sample message for testing"""
        return QueueMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            tenant_id="tenant-test",
            idempotency_key=f"idem_{uuid.uuid4().hex[:8]}",
            priority=5,
            trace_id="trace_123",
            correlation_id="corr_456",
            payload={"action": "test", "data": "sample"},
            metadata={"source": "test_suite"}
        )
    
    # ===== Positive Test Cases =====
    
    def test_basic_enqueue_dequeue(self, queue, sample_message):
        """Test basic enqueue and dequeue operations"""
        # Enqueue
        result = queue.enqueue(sample_message)
        assert result is True
        
        # Dequeue
        messages = queue.dequeue(sample_message.tenant_id)
        assert len(messages) == 1
        assert messages[0].message_id == sample_message.message_id
        assert messages[0].receipt_handle is not None
        assert messages[0].dequeue_count == 1
    
    def test_priority_ordering(self, queue):
        """Test messages are dequeued in priority order"""
        tenant = "tenant-priority"
        
        # Enqueue messages with different priorities
        messages = []
        for priority in [9, 3, 0, 5, 7, 1]:  # Random order
            msg = QueueMessage(
                message_id=f"msg_p{priority}",
                tenant_id=tenant,
                idempotency_key=f"key_p{priority}",
                priority=priority,
                payload={"priority": priority}
            )
            queue.enqueue(msg)
            messages.append(msg)
        
        # Dequeue all
        dequeued = []
        for _ in range(6):
            msgs = queue.dequeue(tenant)
            if msgs:
                dequeued.append(msgs[0])
        
        # Verify priority order (0 is highest)
        priorities = [msg.priority for msg in dequeued]
        assert priorities == [0, 1, 3, 5, 7, 9]
    
    def test_deduplication(self, queue, sample_message):
        """Test idempotency key deduplication"""
        # First enqueue succeeds
        result1 = queue.enqueue(sample_message)
        assert result1 is True
        
        # Duplicate with same idempotency key rejected
        duplicate = QueueMessage(
            message_id="msg_different",
            tenant_id=sample_message.tenant_id,
            idempotency_key=sample_message.idempotency_key,  # Same key
            priority=5,
            payload={"data": "different"}
        )
        result2 = queue.enqueue(duplicate)
        assert result2 is False
        
        # Stats show duplicate
        stats = queue.get_stats(sample_message.tenant_id)
        assert stats["stats"]["duplicates"] == 1
    
    def test_acknowledge_removes_message(self, queue, sample_message):
        """Test acknowledge removes message and allows key reuse"""
        queue.enqueue(sample_message)
        
        # Dequeue and acknowledge
        messages = queue.dequeue(sample_message.tenant_id)
        receipt = messages[0].receipt_handle
        
        result = queue.acknowledge(sample_message.tenant_id, receipt)
        assert result is True
        
        # Message should be gone
        messages = queue.dequeue(sample_message.tenant_id)
        assert len(messages) == 0
        
        # Can now reuse idempotency key
        new_msg = QueueMessage(
            message_id="msg_new",
            tenant_id=sample_message.tenant_id,
            idempotency_key=sample_message.idempotency_key,  # Reuse key
            priority=5,
            payload={"data": "new"}
        )
        result = queue.enqueue(new_msg)
        assert result is True
    
    def test_nack_with_retry(self, queue, sample_message):
        """Test NACK returns message to queue for retry"""
        queue.enqueue(sample_message)
        
        # Dequeue
        messages = queue.dequeue(sample_message.tenant_id)
        receipt = messages[0].receipt_handle
        original_retry_count = messages[0].retry_count
        
        # NACK without delay
        result = queue.nack(sample_message.tenant_id, receipt)
        assert result is True
        
        # Message should be available again
        messages = queue.dequeue(sample_message.tenant_id)
        assert len(messages) == 1
        assert messages[0].retry_count == original_retry_count + 1
    
    def test_visibility_timeout(self, queue, sample_message):
        """Test visibility timeout returns message to queue"""
        queue.enqueue(sample_message)
        
        # Dequeue
        messages = queue.dequeue(sample_message.tenant_id)
        assert len(messages) == 1
        
        # Message not available immediately
        messages2 = queue.dequeue(sample_message.tenant_id)
        assert len(messages2) == 0
        
        # Wait for visibility timeout
        time.sleep(2.5)
        
        # Message should be available again
        messages3 = queue.dequeue(sample_message.tenant_id)
        assert len(messages3) == 1
        assert messages3[0].message_id == sample_message.message_id
    
    def test_dlq_after_max_retries(self, queue, sample_message):
        """Test message moves to DLQ after max retries"""
        queue.enqueue(sample_message)
        
        # Retry max_receive_count times
        for i in range(3):
            messages = queue.dequeue(sample_message.tenant_id)
            assert len(messages) == 1
            assert messages[0].retry_count == i
            
            # NACK to trigger retry
            is_requeued = queue.nack(sample_message.tenant_id, 
                                     messages[0].receipt_handle)
            
            if i < 2:
                assert is_requeued is True
            else:
                # Last retry should go to DLQ
                assert is_requeued is False
        
        # Verify in DLQ
        dlq_messages = queue.get_dlq(sample_message.tenant_id)
        assert len(dlq_messages) == 1
        assert dlq_messages[0].message_id == sample_message.message_id
    
    def test_replay_from_dlq(self, queue, sample_message):
        """Test replaying message from DLQ"""
        queue.enqueue(sample_message)
        
        # Move to DLQ
        for _ in range(3):
            messages = queue.dequeue(sample_message.tenant_id)
            queue.nack(sample_message.tenant_id, messages[0].receipt_handle)
        
        # Verify in DLQ
        dlq = queue.get_dlq(sample_message.tenant_id)
        assert len(dlq) == 1
        
        # Replay
        result = queue.replay(sample_message.tenant_id, sample_message.message_id)
        assert result is True
        
        # Should be back in queue with reset retry count
        messages = queue.dequeue(sample_message.tenant_id)
        assert len(messages) == 1
        assert messages[0].retry_count == 0
        
        # DLQ should be empty
        dlq = queue.get_dlq(sample_message.tenant_id)
        assert len(dlq) == 0
    
    def test_backpressure_queue_depth(self, queue):
        """Test backpressure based on queue depth"""
        tenant = "tenant-backpressure"
        
        # Fill queue to max depth
        for i in range(100):
            msg = QueueMessage(
                message_id=f"msg_{i}",
                tenant_id=tenant,
                idempotency_key=f"key_{i}",
                priority=5,
                payload={"index": i}
            )
            result = queue.enqueue(msg)
            assert result is True
        
        # Next message should be rejected
        overflow = QueueMessage(
            message_id="msg_overflow",
            tenant_id=tenant,
            idempotency_key="key_overflow",
            priority=5,
            payload={"overflow": True}
        )
        result = queue.enqueue(overflow)
        assert result is False
        
        # Stats show backpressure rejection
        stats = queue.get_stats(tenant)
        assert stats["stats"]["backpressure_rejections"] == 1
    
    def test_backpressure_inflight_limit(self, queue):
        """Test backpressure based on inflight messages"""
        tenant = "tenant-inflight"
        
        # Enqueue more than max_inflight
        for i in range(15):
            msg = QueueMessage(
                message_id=f"msg_{i}",
                tenant_id=tenant,
                idempotency_key=f"key_{i}",
                priority=5,
                payload={"index": i}
            )
            queue.enqueue(msg)
        
        # Dequeue up to max_inflight
        messages = queue.dequeue(tenant, max_messages=20)
        assert len(messages) == 10  # Limited by max_inflight
        
        # Can't dequeue more until some are acknowledged
        more = queue.dequeue(tenant)
        assert len(more) == 0
        
        # Acknowledge one
        queue.acknowledge(tenant, messages[0].receipt_handle)
        
        # Now can dequeue one more
        more = queue.dequeue(tenant)
        assert len(more) == 1
    
    def test_concurrent_enqueue(self, queue):
        """Test concurrent enqueue operations"""
        tenant = "tenant-concurrent"
        results = []
        
        def enqueue_task(i):
            msg = QueueMessage(
                message_id=f"msg_concurrent_{i}",
                tenant_id=tenant,
                idempotency_key=f"key_concurrent_{i}",
                priority=5,
                payload={"index": i}
            )
            result = queue.enqueue(msg)
            results.append(result)
        
        # Run concurrent enqueues
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(enqueue_task, i) for i in range(50)]
            for f in futures:
                f.result()
        
        # All should succeed
        assert all(results)
        
        # Verify all messages present
        stats = queue.get_stats(tenant)
        assert stats["queue_depth"] == 50
    
    def test_tenant_isolation(self, queue):
        """Test messages are isolated by tenant"""
        # Enqueue for different tenants
        for tenant_num in range(3):
            tenant = f"tenant-{tenant_num}"
            for i in range(5):
                msg = QueueMessage(
                    message_id=f"msg_{tenant}_{i}",
                    tenant_id=tenant,
                    idempotency_key=f"key_{tenant}_{i}",
                    priority=i,
                    payload={"tenant": tenant, "index": i}
                )
                queue.enqueue(msg)
        
        # Each tenant should only see their messages
        for tenant_num in range(3):
            tenant = f"tenant-{tenant_num}"
            total_dequeued = 0
            
            while True:
                messages = queue.dequeue(tenant, max_messages=10)
                if not messages:
                    break
                    
                # Verify all messages are for this tenant
                for msg in messages:
                    assert msg.tenant_id == tenant
                    assert msg.payload["tenant"] == tenant
                
                total_dequeued += len(messages)
            
            assert total_dequeued == 5
    
    def test_purge_dlq(self, queue):
        """Test purging old DLQ messages"""
        tenant = "tenant-purge"
        
        # Create old message
        old_msg = QueueMessage(
            message_id="msg_old",
            tenant_id=tenant,
            idempotency_key="key_old",
            priority=5,
            payload={"old": True},
            submitted_at=datetime.utcnow() - timedelta(days=31)
        )
        
        # Move to DLQ
        queue._dlq[tenant].append(old_msg)
        
        # Add recent message
        recent_msg = QueueMessage(
            message_id="msg_recent",
            tenant_id=tenant,
            idempotency_key="key_recent",
            priority=5,
            payload={"recent": True}
        )
        queue._dlq[tenant].append(recent_msg)
        
        # Purge old messages
        purged = queue.purge_dlq(tenant, older_than_days=30)
        assert purged == 1
        
        # Only recent message remains
        dlq = queue.get_dlq(tenant)
        assert len(dlq) == 1
        assert dlq[0].message_id == "msg_recent"
    
    # ===== Negative Test Cases =====
    
    def test_acknowledge_invalid_receipt(self, queue):
        """Test acknowledge with invalid receipt handle"""
        result = queue.acknowledge("tenant-test", "invalid_receipt")
        assert result is False
    
    def test_nack_invalid_receipt(self, queue):
        """Test NACK with invalid receipt handle"""
        result = queue.nack("tenant-test", "invalid_receipt")
        assert result is False
    
    def test_replay_nonexistent_message(self, queue):
        """Test replay with non-existent message ID"""
        result = queue.replay("tenant-test", "nonexistent_msg")
        assert result is False
    
    def test_dequeue_empty_queue(self, queue):
        """Test dequeue from empty queue"""
        messages = queue.dequeue("tenant-empty")
        assert messages == []
    
    def test_dequeue_with_wait_timeout(self, queue):
        """Test dequeue with wait timeout on empty queue"""
        start = time.time()
        messages = queue.dequeue("tenant-empty", wait_timeout=1.0)
        elapsed = time.time() - start
        
        assert messages == []
        assert 0.9 < elapsed < 1.2  # Approximately 1 second wait
    
    def test_concurrent_dequeue_same_message(self, queue, sample_message):
        """Test concurrent dequeue prevents double processing"""
        queue.enqueue(sample_message)
        
        dequeued = []
        
        def dequeue_task():
            msgs = queue.dequeue(sample_message.tenant_id)
            if msgs:
                dequeued.append(msgs[0])
        
        # Multiple threads try to dequeue
        threads = [threading.Thread(target=dequeue_task) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Only one should succeed
        assert len(dequeued) == 1
        assert dequeued[0].message_id == sample_message.message_id
    
    def test_nack_with_retry_delay(self, queue, sample_message):
        """Test NACK with retry delay keeps message invisible"""
        queue.enqueue(sample_message)
        
        # Dequeue and NACK with delay
        messages = queue.dequeue(sample_message.tenant_id)
        queue.nack(sample_message.tenant_id, messages[0].receipt_handle, 
                  retry_delay=3)
        
        # Message should not be available immediately
        messages = queue.dequeue(sample_message.tenant_id)
        assert len(messages) == 0
        
        # Wait for retry delay
        time.sleep(3.5)
        
        # Now should be available
        messages = queue.dequeue(sample_message.tenant_id)
        assert len(messages) == 1
    
    def test_stats_accuracy(self, queue):
        """Test queue statistics accuracy"""
        tenant = "tenant-stats"
        
        # Perform various operations
        for i in range(5):
            msg = QueueMessage(
                message_id=f"msg_{i}",
                tenant_id=tenant,
                idempotency_key=f"key_{i}",
                priority=i % 3,
                payload={"index": i}
            )
            queue.enqueue(msg)
        
        # Dequeue some
        messages = queue.dequeue(tenant, max_messages=2)
        
        # Acknowledge one
        queue.acknowledge(tenant, messages[0].receipt_handle)
        
        # NACK one
        queue.nack(tenant, messages[1].receipt_handle)
        
        # Check stats
        stats = queue.get_stats(tenant)
        assert stats["queue_depth"] == 4  # 3 never dequeued + 1 NACKed
        assert stats["inflight"] == 0
        assert stats["stats"]["enqueued"] == 5
        assert stats["stats"]["dequeued"] == 2
        assert stats["stats"]["acknowledged"] == 1
        assert stats["stats"]["nacked"] == 1
    
    def test_fifo_within_priority(self, queue):
        """Test FIFO ordering within same priority level"""
        tenant = "tenant-fifo"
        
        # Enqueue messages with same priority
        for i in range(5):
            msg = QueueMessage(
                message_id=f"msg_{i}",
                tenant_id=tenant,
                idempotency_key=f"key_{i}",
                priority=5,  # Same priority
                payload={"index": i},
                submitted_at=datetime.utcnow() + timedelta(microseconds=i * 1000)
            )
            queue.enqueue(msg)
            time.sleep(0.001)  # Ensure different timestamps
        
        # Dequeue all
        dequeued = []
        for _ in range(5):
            msgs = queue.dequeue(tenant)
            if msgs:
                dequeued.append(msgs[0])
        
        # Verify FIFO order
        indices = [msg.payload["index"] for msg in dequeued]
        assert indices == [0, 1, 2, 3, 4]