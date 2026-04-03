from dataclasses import dataclass, field
from typing import Any


@dataclass
class RobotConfig:
    x_min: float = 0
    x_max: float = 400
    y_min: float = 0
    y_max: float = 400
    z_min: float = 0
    z_max: float = 150
    max_delta: float = 50


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000


@dataclass
class SystemConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    debug: bool = False


DEFAULT_CONFIG = SystemConfig()


@dataclass
class SceneObject:
    name: str
    x: float
    y: float
    z: float = 0.0
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scene:
    objects: list[SceneObject]
    timestamp: float = 0.0
    camera_frame: Any | None = None


@dataclass
class RobotState:
    x: float
    y: float
    z: float
    rotation: float = 0.0
    gripper: int = 0


@dataclass
class Action:
    x: float
    y: float
    z: float
    rotation: float = 0.0
    gripper: int = 0
    done: bool = False

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "rotation": self.rotation,
            "gripper": self.gripper,
            "done": self.done,
        }


@dataclass
class TaskPlan:
    actions: list[str]
    objects: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)