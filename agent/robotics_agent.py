# agent/robotics_agent.py
"""
Complete robotics domain implementation with RT-2 orchestrator, π-0 vision agent,
safety protocols (E-stop, watchdog, rate limits), and HITL approval for actuation.
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import uuid
from collections import deque

from opentelemetry import trace
tracer = trace.get_tracer(__name__)


class RoboticsTaskType(Enum):
    """Types of robotics tasks"""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INSPECTION = "inspection"
    ASSEMBLY = "assembly"
    MAINTENANCE = "maintenance"
    EMERGENCY_RESPONSE = "emergency_response"


class SafetyLevel(Enum):
    """Safety levels for robotics operations"""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGER = "danger"
    EMERGENCY = "emergency"


class ActuatorType(Enum):
    """Types of robot actuators"""
    ARM = "arm"
    GRIPPER = "gripper"
    WHEEL = "wheel"
    TRACK = "track"
    JOINT = "joint"
    TOOL = "tool"


@dataclass
class RobotState:
    """Current state of the robot"""
    robot_id: str
    position: Dict[str, float]  # x, y, z, roll, pitch, yaw
    velocity: Dict[str, float]
    joint_positions: List[float]
    gripper_state: str  # open, closed, holding
    battery_level: float
    temperature: float
    safety_level: SafetyLevel
    active_task: Optional[str]
    error_state: Optional[str]
    timestamp: datetime


@dataclass
class SafetyConstraints:
    """Safety constraints for robot operations"""
    max_velocity: float = 1.0  # m/s
    max_acceleration: float = 0.5  # m/s²
    max_joint_velocity: float = 1.0  # rad/s
    max_force: float = 100.0  # N
    min_distance_to_human: float = 1.0  # meters
    min_distance_to_obstacle: float = 0.2  # meters
    max_operation_time: int = 300  # seconds
    emergency_stop_deceleration: float = 2.0  # m/s²
    watchdog_timeout: int = 1  # seconds
    rate_limit_actions_per_second: int = 10


@dataclass
class VisionAnalysis:
    """Result from vision agent analysis"""
    scene_description: str
    objects_detected: List[Dict[str, Any]]
    obstacles: List[Dict[str, Any]]
    humans_detected: List[Dict[str, Any]]
    path_clear: bool
    safety_risks: List[str]
    confidence_score: float
    timestamp: datetime


class RoboticsOrchestrator:
    """
    RT-2 based orchestrator for robotics tasks with safety protocols
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.robots: Dict[str, RobotState] = {}
        self.safety_constraints = SafetyConstraints()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.emergency_stop_engaged = False
        self.watchdog_timers: Dict[str, asyncio.Task] = {}
        self.rate_limiters: Dict[str, deque] = {}
        self.hitl_approvals: Dict[str, bool] = {}
        self.telemetry_buffer: deque = deque(maxlen=10000)
        
    @tracer.start_as_current_span("plan_robotics_task")
    async def plan_task(
        self,
        task_type: RoboticsTaskType,
        robot_id: str,
        goal: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Plan a robotics task using RT-2 policy
        """
        span = trace.get_current_span()
        
        # Check emergency stop
        if self.emergency_stop_engaged:
            raise Exception("Emergency stop engaged - cannot plan new tasks")
        
        # Get robot state
        robot_state = self.robots.get(robot_id)
        if not robot_state:
            raise ValueError(f"Robot {robot_id} not found")
        
        # Vision analysis for environment understanding
        vision_result = await self._request_vision_analysis(robot_state)
        
        # Safety check
        safety_assessment = await self._assess_safety(
            robot_state,
            vision_result,
            task_type,
            goal
        )
        
        if safety_assessment["level"] in [SafetyLevel.DANGER, SafetyLevel.EMERGENCY]:
            raise Exception(f"Safety violation: {safety_assessment['reason']}")
        
        # Generate plan based on task type
        if task_type == RoboticsTaskType.NAVIGATION:
            plan = await self._plan_navigation(robot_state, goal, vision_result)
        elif task_type == RoboticsTaskType.MANIPULATION:
            plan = await self._plan_manipulation(robot_state, goal, vision_result)
        elif task_type == RoboticsTaskType.INSPECTION:
            plan = await self._plan_inspection(robot_state, goal, vision_result)
        else:
            plan = await self._plan_generic(robot_state, goal, task_type)
        
        # Add safety constraints to plan
        plan["safety_constraints"] = {
            "max_velocity": self.safety_constraints.max_velocity,
            "max_acceleration": self.safety_constraints.max_acceleration,
            "min_distance_to_obstacle": self.safety_constraints.min_distance_to_obstacle,
            "watchdog_timeout": self.safety_constraints.watchdog_timeout
        }
        
        # Request HITL approval for critical actions
        if await self._requires_hitl_approval(task_type, plan):
            approval_id = await self._request_hitl_approval(robot_id, task_type, plan)
            plan["hitl_approval_required"] = True
            plan["approval_id"] = approval_id
        
        # Store active task
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        self.active_tasks[task_id] = {
            "robot_id": robot_id,
            "task_type": task_type,
            "plan": plan,
            "status": "planned",
            "created_at": datetime.utcnow()
        }
        
        # Start watchdog timer
        self.watchdog_timers[task_id] = asyncio.create_task(
            self._watchdog_timer(task_id, robot_id)
        )
        
        span.set_attributes({
            "robot_id": robot_id,
            "task_type": task_type.value,
            "task_id": task_id,
            "safety_level": safety_assessment["level"].value
        })
        
        return {
            "task_id": task_id,
            "plan": plan,
            "safety_assessment": safety_assessment,
            "estimated_duration": plan.get("estimated_duration", 60)
        }
    
    async def _plan_navigation(
        self,
        robot_state: RobotState,
        goal: Dict[str, Any],
        vision: VisionAnalysis
    ) -> Dict[str, Any]:
        """Plan navigation task"""
        # Calculate path using A* or RRT*
        waypoints = self._calculate_path(
            robot_state.position,
            goal["target_position"],
            vision.obstacles
        )
        
        return {
            "type": "navigation",
            "waypoints": waypoints,
            "actions": [
                {
                    "type": "move_to",
                    "position": wp,
                    "max_velocity": min(
                        self.safety_constraints.max_velocity,
                        self._calculate_safe_velocity(robot_state, vision)
                    )
                }
                for wp in waypoints
            ],
            "estimated_duration": len(waypoints) * 10,
            "recovery_policy": {
                "obstacle_detected": "replan_path",
                "human_detected": "emergency_stop",
                "battery_low": "return_to_base"
            }
        }
    
    async def _plan_manipulation(
        self,
        robot_state: RobotState,
        goal: Dict[str, Any],
        vision: VisionAnalysis
    ) -> Dict[str, Any]:
        """Plan manipulation task"""
        target_object = goal.get("target_object")
        
        # Find object in vision
        object_info = None
        for obj in vision.objects_detected:
            if obj["type"] == target_object:
                object_info = obj
                break
        
        if not object_info:
            raise ValueError(f"Target object {target_object} not found")
        
        return {
            "type": "manipulation",
            "target": object_info,
            "actions": [
                {"type": "approach", "position": object_info["position"]},
                {"type": "open_gripper"},
                {"type": "move_to", "position": object_info["grasp_point"]},
                {"type": "close_gripper", "force": 50},
                {"type": "lift", "height": 0.1},
                {"type": "move_to", "position": goal["destination"]},
                {"type": "open_gripper"}
            ],
            "estimated_duration": 60,
            "recovery_policy": {
                "grasp_failed": "retry_grasp",
                "object_dropped": "reacquire_object",
                "collision_detected": "back_off"
            }
        }
    
    async def _plan_inspection(
        self,
        robot_state: RobotState,
        goal: Dict[str, Any],
        vision: VisionAnalysis
    ) -> Dict[str, Any]:
        """Plan inspection task"""
        inspection_points = goal.get("inspection_points", [])
        
        return {
            "type": "inspection",
            "inspection_points": inspection_points,
            "actions": [
                {
                    "type": "move_to",
                    "position": point,
                    "capture_image": True,
                    "analyze": True
                }
                for point in inspection_points
            ],
            "estimated_duration": len(inspection_points) * 30,
            "recovery_policy": {
                "vision_unclear": "adjust_position",
                "anomaly_detected": "detailed_scan"
            }
        }
    
    async def _plan_generic(
        self,
        robot_state: RobotState,
        goal: Dict[str, Any],
        task_type: RoboticsTaskType
    ) -> Dict[str, Any]:
        """Generic task planning"""
        return {
            "type": task_type.value,
            "goal": goal,
            "actions": [
                {"type": "execute", "parameters": goal}
            ],
            "estimated_duration": 120,
            "recovery_policy": {
                "error": "safe_stop",
                "timeout": "abort"
            }
        }
    
    def _calculate_path(
        self,
        start: Dict[str, float],
        goal: Dict[str, float],
        obstacles: List[Dict[str, Any]]
    ) -> List[Dict[str, float]]:
        """Calculate collision-free path"""
        # Simplified path planning - in production use RRT*, A*, or similar
        waypoints = []
        
        # Direct path with obstacle avoidance
        steps = 10
        for i in range(steps + 1):
            t = i / steps
            waypoint = {
                "x": start["x"] + t * (goal["x"] - start["x"]),
                "y": start["y"] + t * (goal["y"] - start["y"]),
                "z": start.get("z", 0)
            }
            
            # Check for obstacles and adjust
            for obstacle in obstacles:
                dist = self._distance(waypoint, obstacle["position"])
                if dist < self.safety_constraints.min_distance_to_obstacle:
                    # Adjust waypoint to avoid obstacle
                    waypoint["x"] += (waypoint["x"] - obstacle["position"]["x"]) * 0.5
                    waypoint["y"] += (waypoint["y"] - obstacle["position"]["y"]) * 0.5
            
            waypoints.append(waypoint)
        
        return waypoints
    
    def _distance(self, p1: Dict[str, float], p2: Dict[str, float]) -> float:
        """Calculate Euclidean distance"""
        return np.sqrt(
            (p1.get("x", 0) - p2.get("x", 0)) ** 2 +
            (p1.get("y", 0) - p2.get("y", 0)) ** 2 +
            (p1.get("z", 0) - p2.get("z", 0)) ** 2
        )
    
    def _calculate_safe_velocity(
        self,
        robot_state: RobotState,
        vision: VisionAnalysis
    ) -> float:
        """Calculate safe velocity based on environment"""
        base_velocity = self.safety_constraints.max_velocity
        
        # Reduce velocity if humans detected
        if vision.humans_detected:
            min_human_dist = min(
                self._distance(robot_state.position, h["position"])
                for h in vision.humans_detected
            )
            
            if min_human_dist < self.safety_constraints.min_distance_to_human * 2:
                base_velocity *= 0.3
            elif min_human_dist < self.safety_constraints.min_distance_to_human * 3:
                base_velocity *= 0.5
        
        # Reduce velocity for obstacles
        if vision.obstacles:
            min_obstacle_dist = min(
                self._distance(robot_state.position, o["position"])
                for o in vision.obstacles
            )
            
            if min_obstacle_dist < self.safety_constraints.min_distance_to_obstacle * 2:
                base_velocity *= 0.5
        
        return base_velocity
    
    async def _assess_safety(
        self,
        robot_state: RobotState,
        vision: VisionAnalysis,
        task_type: RoboticsTaskType,
        goal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess safety of planned operation"""
        level = SafetyLevel.SAFE
        risks = []
        
        # Check robot health
        if robot_state.battery_level < 20:
            risks.append("Low battery")
            level = SafetyLevel.WARNING
        
        if robot_state.temperature > 60:
            risks.append("High temperature")
            level = SafetyLevel.WARNING
        
        if robot_state.error_state:
            risks.append(f"Error state: {robot_state.error_state}")
            level = SafetyLevel.DANGER
        
        # Check environment
        if vision.humans_detected:
            for human in vision.humans_detected:
                dist = self._distance(robot_state.position, human["position"])
                if dist < self.safety_constraints.min_distance_to_human:
                    risks.append("Human too close")
                    level = SafetyLevel.DANGER
        
        # Check task-specific risks
        if task_type == RoboticsTaskType.MANIPULATION:
            if goal.get("weight", 0) > 50:
                risks.append("Heavy object")
                level = SafetyLevel.CAUTION
        
        if task_type == RoboticsTaskType.EMERGENCY_RESPONSE:
            level = SafetyLevel.WARNING  # Always heightened for emergency
        
        return {
            "level": level,
            "risks": risks,
            "reason": "; ".join(risks) if risks else "No safety concerns",
            "timestamp": datetime.utcnow()
        }
    
    async def _requires_hitl_approval(
        self,
        task_type: RoboticsTaskType,
        plan: Dict[str, Any]
    ) -> bool:
        """Check if HITL approval is required"""
        # Always require approval for certain task types
        if task_type in [
            RoboticsTaskType.EMERGENCY_RESPONSE,
            RoboticsTaskType.MAINTENANCE
        ]:
            return True
        
        # Require approval for manipulation of heavy/dangerous objects
        if task_type == RoboticsTaskType.MANIPULATION:
            if plan.get("target", {}).get("weight", 0) > 20:
                return True
            if plan.get("target", {}).get("hazardous", False):
                return True
        
        # Require approval for high-speed navigation
        if task_type == RoboticsTaskType.NAVIGATION:
            max_vel = max(
                action.get("max_velocity", 0)
                for action in plan.get("actions", [])
            )
            if max_vel > self.safety_constraints.max_velocity * 0.8:
                return True
        
        return False
    
    async def _request_hitl_approval(
        self,
        robot_id: str,
        task_type: RoboticsTaskType,
        plan: Dict[str, Any]
    ) -> str:
        """Request HITL approval for actuation"""
        approval_id = f"approval-{uuid.uuid4().hex[:12]}"
        
        self.hitl_approvals[approval_id] = False
        
        # In production, this would send to HITL system
        print(f"HITL Approval Required: {approval_id}")
        print(f"Robot: {robot_id}")
        print(f"Task: {task_type.value}")
        print(f"Plan: {json.dumps(plan, indent=2, default=str)}")
        
        return approval_id
    
    async def approve_actuation(self, approval_id: str) -> bool:
        """Approve actuation request"""
        if approval_id in self.hitl_approvals:
            self.hitl_approvals[approval_id] = True
            return True
        return False
    
    async def execute_task(
        self,
        task_id: str
    ) -> Dict[str, Any]:
        """Execute a planned task with rate limiting"""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.active_tasks[task_id]
        robot_id = task["robot_id"]
        
        # Check HITL approval if required
        if task["plan"].get("hitl_approval_required"):
            approval_id = task["plan"]["approval_id"]
            if not self.hitl_approvals.get(approval_id, False):
                return {
                    "status": "waiting_approval",
                    "approval_id": approval_id
                }
        
        # Rate limiting
        if not await self._check_rate_limit(robot_id):
            return {
                "status": "rate_limited",
                "retry_after": 1
            }
        
        # Execute actions
        task["status"] = "executing"
        results = []
        
        for action in task["plan"]["actions"]:
            # Check emergency stop
            if self.emergency_stop_engaged:
                await self._safe_stop(robot_id)
                return {
                    "status": "emergency_stopped",
                    "completed_actions": results
                }
            
            # Execute action
            try:
                result = await self._execute_action(robot_id, action)
                results.append(result)
                
                # Reset watchdog
                if task_id in self.watchdog_timers:
                    self.watchdog_timers[task_id].cancel()
                    self.watchdog_timers[task_id] = asyncio.create_task(
                        self._watchdog_timer(task_id, robot_id)
                    )
                
            except Exception as e:
                # Error recovery
                await self._handle_error(robot_id, task, action, e)
                return {
                    "status": "error",
                    "error": str(e),
                    "completed_actions": results
                }
        
        task["status"] = "completed"
        
        return {
            "status": "completed",
            "results": results
        }
    
    async def _check_rate_limit(self, robot_id: str) -> bool:
        """Check rate limit for robot actions"""
        now = datetime.utcnow()
        
        if robot_id not in self.rate_limiters:
            self.rate_limiters[robot_id] = deque()
        
        # Remove old entries
        cutoff = now - timedelta(seconds=1)
        while self.rate_limiters[robot_id] and self.rate_limiters[robot_id][0] < cutoff:
            self.rate_limiters[robot_id].popleft()
        
        # Check limit
        if len(self.rate_limiters[robot_id]) >= self.safety_constraints.rate_limit_actions_per_second:
            return False
        
        # Add current request
        self.rate_limiters[robot_id].append(now)
        
        return True
    
    async def _execute_action(
        self,
        robot_id: str,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single robot action"""
        action_type = action["type"]
        
        # Log telemetry
        self.telemetry_buffer.append({
            "timestamp": datetime.utcnow(),
            "robot_id": robot_id,
            "action": action,
            "state": self.robots.get(robot_id)
        })
        
        # Simulate action execution
        # In production, this would send commands to robot
        await asyncio.sleep(0.1)
        
        return {
            "action": action_type,
            "status": "completed",
            "timestamp": datetime.utcnow()
        }
    
    async def _watchdog_timer(self, task_id: str, robot_id: str):
        """Watchdog timer for task execution"""
        await asyncio.sleep(self.safety_constraints.watchdog_timeout)
        
        # Timeout - trigger safe stop
        print(f"Watchdog timeout for task {task_id}")
        await self._safe_stop(robot_id)
        
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["status"] = "timeout"
    
    async def _handle_error(
        self,
        robot_id: str,
        task: Dict[str, Any],
        action: Dict[str, Any],
        error: Exception
    ):
        """Handle execution error with recovery"""
        recovery_policy = task["plan"].get("recovery_policy", {})
        error_type = type(error).__name__
        
        if error_type in recovery_policy:
            recovery_action = recovery_policy[error_type]
            
            if recovery_action == "safe_stop":
                await self._safe_stop(robot_id)
            elif recovery_action == "replan_path":
                # Trigger replanning
                pass
            elif recovery_action == "emergency_stop":
                await self.emergency_stop()
        else:
            # Default: safe stop
            await self._safe_stop(robot_id)
    
    async def _safe_stop(self, robot_id: str):
        """Execute safe stop for robot"""
        if robot_id in self.robots:
            # Decelerate safely
            print(f"Safe stop initiated for robot {robot_id}")
            # In production, send stop command to robot
    
    async def emergency_stop(self):
        """Emergency stop for all robots"""
        self.emergency_stop_engaged = True
        print("EMERGENCY STOP ENGAGED")
        
        # Stop all robots
        for robot_id in self.robots:
            await self._safe_stop(robot_id)
        
        # Cancel all tasks
        for task_id in list(self.active_tasks.keys()):
            if task_id in self.watchdog_timers:
                self.watchdog_timers[task_id].cancel()
            self.active_tasks[task_id]["status"] = "emergency_stopped"
    
    async def reset_emergency_stop(self):
        """Reset emergency stop"""
        self.emergency_stop_engaged = False
        print("Emergency stop reset")
    
    async def _request_vision_analysis(
        self,
        robot_state: RobotState
    ) -> VisionAnalysis:
        """Request analysis from vision agent"""
        # In production, call π-0-small vision model
        # For now, return mock analysis
        return VisionAnalysis(
            scene_description="Warehouse environment with shelving units",
            objects_detected=[
                {
                    "type": "box",
                    "position": {"x": 2.0, "y": 1.0, "z": 0.5},
                    "size": {"w": 0.3, "h": 0.2, "l": 0.4},
                    "weight": 5.0,
                    "grasp_point": {"x": 2.0, "y": 1.0, "z": 0.6}
                }
            ],
            obstacles=[
                {
                    "type": "shelf",
                    "position": {"x": 1.0, "y": 0, "z": 0},
                    "size": {"w": 2.0, "h": 2.0, "l": 0.5}
                }
            ],
            humans_detected=[],
            path_clear=True,
            safety_risks=[],
            confidence_score=0.95,
            timestamp=datetime.utcnow()
        )
    
    def register_robot(
        self,
        robot_id: str,
        initial_state: Optional[RobotState] = None
    ):
        """Register a robot in the system"""
        if initial_state:
            self.robots[robot_id] = initial_state
        else:
            self.robots[robot_id] = RobotState(
                robot_id=robot_id,
                position={"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
                velocity={"linear": 0, "angular": 0},
                joint_positions=[0] * 7,
                gripper_state="open",
                battery_level=100,
                temperature=25,
                safety_level=SafetyLevel.SAFE,
                active_task=None,
                error_state=None,
                timestamp=datetime.utcnow()
            )


class VisionAgent:
    """
    π-0-small based vision agent for scene understanding
    """
    
    def __init__(self):
        self.model = "pi-0-small"
        self.analysis_history = deque(maxlen=100)
    
    async def analyze_scene(
        self,
        image_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> VisionAnalysis:
        """
        Analyze scene using vision model
        """
        # In production, call π-0-small model
        # Process image and return analysis
        
        analysis = VisionAnalysis(
            scene_description=self._generate_description(image_data),
            objects_detected=await self._detect_objects(image_data),
            obstacles=await self._detect_obstacles(image_data),
            humans_detected=await self._detect_humans(image_data),
            path_clear=await self._check_path_clear(image_data),
            safety_risks=await self._identify_risks(image_data),
            confidence_score=0.92,
            timestamp=datetime.utcnow()
        )
        
        self.analysis_history.append(analysis)
        
        return analysis
    
    def _generate_description(self, image_data: Any) -> str:
        """Generate scene description"""
        # In production, use vision-language model
        return "Industrial environment with equipment and clear pathways"
    
    async def _detect_objects(self, image_data: Any) -> List[Dict[str, Any]]:
        """Detect objects in scene"""
        # In production, use object detection model
        return [
            {
                "type": "pallet",
                "position": {"x": 3.0, "y": 2.0, "z": 0},
                "confidence": 0.95,
                "bounding_box": [100, 100, 200, 200]
            }
        ]
    
    async def _detect_obstacles(self, image_data: Any) -> List[Dict[str, Any]]:
        """Detect obstacles"""
        # In production, use segmentation model
        return [
            {
                "type": "column",
                "position": {"x": 5.0, "y": 3.0, "z": 0},
                "size": {"radius": 0.3, "height": 3.0}
            }
        ]
    
    async def _detect_humans(self, image_data: Any) -> List[Dict[str, Any]]:
        """Detect humans in scene"""
        # In production, use person detection model
        return []
    
    async def _check_path_clear(self, image_data: Any) -> bool:
        """Check if path is clear"""
        # In production, analyze path
        return True
    
    async def _identify_risks(self, image_data: Any) -> List[str]:
        """Identify safety risks"""
        # In production, use risk assessment model
        risks = []
        
        # Check for common hazards
        # - Wet floor
        # - Unstable objects
        # - Fire/smoke
        # - Equipment malfunction
        
        return risks


# Example usage
async def main():
    config = {}
    
    # Initialize orchestrator
    orchestrator = RoboticsOrchestrator(config)
    
    # Register robot
    orchestrator.register_robot("robot-001")
    
    # Plan navigation task
    nav_plan = await orchestrator.plan_task(
        task_type=RoboticsTaskType.NAVIGATION,
        robot_id="robot-001",
        goal={"target_position": {"x": 10, "y": 5, "z": 0}}
    )
    
    print(f"Navigation plan: {json.dumps(nav_plan, indent=2, default=str)}")
    
    # Plan manipulation task
    manip_plan = await orchestrator.plan_task(
        task_type=RoboticsTaskType.MANIPULATION,
        robot_id="robot-001",
        goal={
            "target_object": "box",
            "destination": {"x": 5, "y": 5, "z": 1}
        }
    )
    
    print(f"Manipulation plan: {json.dumps(manip_plan, indent=2, default=str)}")
    
    # Execute task (would require approval)
    if manip_plan["plan"].get("hitl_approval_required"):
        approval_id = manip_plan["plan"]["approval_id"]
        await orchestrator.approve_actuation(approval_id)
    
    result = await orchestrator.execute_task(manip_plan["task_id"])
    print(f"Execution result: {result}")
    
    # Initialize vision agent
    vision = VisionAgent()
    
    # Analyze scene
    scene_analysis = await vision.analyze_scene(None)  # Would pass actual image
    print(f"Scene analysis: {scene_analysis.scene_description}")


if __name__ == "__main__":
    asyncio.run(main())