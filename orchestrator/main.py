# orchestrator/main.py - Orchestrator Entry Point

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.core import Orchestrator, Task

async def main():
    print("Starting Orchestrator...")
    orchestrator = Orchestrator()
    
    # Example task
    task = Task(description="Test task")
    result = await orchestrator.process_task(task)
    print(f"Task completed: {result.status}")

if __name__ == "__main__":
    asyncio.run(main())
