# execution/queue.py
import json
import logging
from typing import Dict, Any
from queue import Queue
from threading import Thread

logger = logging.getLogger(__name__)

# Global task queue (replace with Redis/RabbitMQ in production)
task_queue = Queue()

def enqueue_task(task: Dict[str, Any]) -> bool:
    """
    Enqueue task for asynchronous processing
    
    Args:
        task: Task dictionary with task_id, operation, payload, etc.
        
    Returns:
        True if successfully enqueued
    """
    try:
        task_queue.put(task)
        logger.info(f"Task {task.get('task_id')} enqueued for processing")
        return True
    except Exception as e:
        logger.error(f"Failed to enqueue task: {str(e)}")
        return False

def process_queue():
    """
    Background worker to process queued tasks
    This would run in a separate thread/process in production
    """
    from orchestrator.brain import OrchestrationBrain
    brain = OrchestrationBrain()
    
    while True:
        try:
            task = task_queue.get(timeout=1)
            logger.info(f"Processing task {task.get('task_id')}")
            
            # Process task through orchestrator
            result = brain.process_task(task)
            
            # Update task status in storage
            # This would update the actual task database in production
            logger.info(f"Task {task.get('task_id')} completed with status: {result.get('status')}")
            
        except Exception as e:
            if "Empty" not in str(e):  # Ignore queue empty exceptions
                logger.error(f"Error processing task: {str(e)}")

# Start background worker (in production, use proper worker management)
# worker_thread = Thread(target=process_queue, daemon=True)
# worker_thread.start()

class QueueMetrics:
    """Basic queue metrics for monitoring"""
    
    @staticmethod
    def get_queue_size() -> int:
        """Get current queue size"""
        return task_queue.qsize()
    
    @staticmethod
    def is_queue_empty() -> bool:
        """Check if queue is empty"""
        return task_queue.empty()