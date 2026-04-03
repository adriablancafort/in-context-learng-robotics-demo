import logging
import math
import time
from typing import Any

import cv2

from src.models import Action, RobotConfig, RobotState, Scene, SceneObject

logger = logging.getLogger(__name__)


class PerceptionModule:
    def __init__(self, use_camera: bool = False):
        self.camera = cv2.VideoCapture(0) if use_camera else None
        self._objects = [
            SceneObject(name="red_block", x=100, y=50, confidence=0.95),
            SceneObject(name="blue_block", x=150, y=80, confidence=0.92),
            SceneObject(name="bowl", x=250, y=200, confidence=0.98),
        ]

    def get_scene(self) -> Scene:
        if self.camera is not None:
            ok, frame = self.camera.read()
            if ok:
                return Scene(objects=list(self._objects), timestamp=time.time(), camera_frame=frame)
        return Scene(objects=list(self._objects), timestamp=time.time())

    def update_object_position(self, object_name: str, x: float, y: float, z: float = 0.0) -> None:
        for index, obj in enumerate(self._objects):
            if obj.name == object_name:
                self._objects[index] = SceneObject(
                    name=obj.name,
                    x=x,
                    y=y,
                    z=z,
                    confidence=obj.confidence,
                    metadata=obj.metadata,
                )
                break

    def close(self) -> None:
        if self.camera is not None:
            self.camera.release()
            self.camera = None


class SafetyFilter:
    def __init__(self, config: RobotConfig | None = None):
        self.config = config or RobotConfig()

    def validate_action(self, action: Action, current_state: RobotState) -> Action | None:
        try:
            self._check_numeric(action)
            self._check_bounds(action)
            self._check_delta(action, current_state)
        except ValueError as exc:
            logger.warning("Action validation failed: %s", exc)
            return None
        return action

    def clip_action(self, action: Action, current_state: RobotState) -> Action:
        clipped = Action(
            x=self._clip(action.x, self.config.x_min, self.config.x_max),
            y=self._clip(action.y, self.config.y_min, self.config.y_max),
            z=self._clip(action.z, self.config.z_min, self.config.z_max),
            rotation=action.rotation,
            gripper=max(0, min(1, int(action.gripper))),
            done=action.done,
        )
        clipped.x = self._clip_delta(clipped.x, current_state.x)
        clipped.y = self._clip_delta(clipped.y, current_state.y)
        clipped.z = self._clip_delta(clipped.z, current_state.z)
        if clipped.to_dict() != action.to_dict():
            logger.info("Action clipped: %s -> %s", action, clipped)
        return clipped

    def _check_bounds(self, action: Action) -> None:
        if not self.config.x_min <= action.x <= self.config.x_max:
            raise ValueError(f"X coordinate {action.x} out of bounds [{self.config.x_min}, {self.config.x_max}]")
        if not self.config.y_min <= action.y <= self.config.y_max:
            raise ValueError(f"Y coordinate {action.y} out of bounds [{self.config.y_min}, {self.config.y_max}]")
        if not self.config.z_min <= action.z <= self.config.z_max:
            raise ValueError(f"Z coordinate {action.z} out of bounds [{self.config.z_min}, {self.config.z_max}]")

    def _check_delta(self, action: Action, current_state: RobotState) -> None:
        if abs(action.x - current_state.x) > self.config.max_delta:
            raise ValueError(
                f"X movement delta {abs(action.x - current_state.x)} exceeds max {self.config.max_delta}. "
                f"Current: {current_state.x}, Target: {action.x}"
            )
        if abs(action.y - current_state.y) > self.config.max_delta:
            raise ValueError(
                f"Y movement delta {abs(action.y - current_state.y)} exceeds max {self.config.max_delta}. "
                f"Current: {current_state.y}, Target: {action.y}"
            )
        if abs(action.z - current_state.z) > self.config.max_delta:
            raise ValueError(
                f"Z movement delta {abs(action.z - current_state.z)} exceeds max {self.config.max_delta}. "
                f"Current: {current_state.z}, Target: {action.z}"
            )

    @staticmethod
    def _check_numeric(action: Action) -> None:
        for name in ("x", "y", "z", "rotation"):
            value = getattr(action, name)
            if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                raise ValueError(f"Invalid numeric value for {name}: {value}")

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _clip_delta(self, target: float, current: float) -> float:
        delta = target - current
        if abs(delta) <= self.config.max_delta:
            return target
        return current + (self.config.max_delta if delta > 0 else -self.config.max_delta)


class RobotInterface:
    def __init__(self, initial_state: RobotState | None = None):
        self._state = initial_state or RobotState(x=200, y=200, z=50)
        self._execution_log: list[dict[str, Any]] = []
        logger.info("Robot initialized at (%s, %s, %s)", self._state.x, self._state.y, self._state.z)

    def get_state(self) -> RobotState:
        return RobotState(**self._state.__dict__)

    def execute(self, action: Action) -> bool:
        logger.info("Executing action: x=%s, y=%s, z=%s, gripper=%s", action.x, action.y, action.z, action.gripper)
        self._state = RobotState(
            x=action.x,
            y=action.y,
            z=action.z,
            rotation=action.rotation,
            gripper=action.gripper,
        )
        self._execution_log.append({"action": action.to_dict(), "result": "success"})
        return True

    def open_gripper(self) -> bool:
        return self.execute(Action(self._state.x, self._state.y, self._state.z, self._state.rotation, 0))

    def close_gripper(self) -> bool:
        return self.execute(Action(self._state.x, self._state.y, self._state.z, self._state.rotation, 1))

    def reset_to_home(self) -> bool:
        return self.execute(Action(x=200, y=200, z=50))

    def get_execution_log(self) -> list[dict[str, Any]]:
        return self._execution_log.copy()