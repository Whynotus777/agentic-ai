import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.core import Orchestrator, Task, TaskStatus

@pytest.mark.asyncio
async def test_orchestrator_process_task():
    orchestrator = Orchestrator()
    task = Task(description="Test task")
    
    result = await orchestrator.process_task(task)
    
    assert result.status == TaskStatus.COMPLETED
    assert result.id == task.id
