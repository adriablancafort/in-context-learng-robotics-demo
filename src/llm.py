import json
import logging

from openai import OpenAI

from src.models import Action, LLMConfig, RobotState, Scene, TaskPlan

logger = logging.getLogger(__name__)


class TaskPreprocessor:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def decompose_task(self, task: str) -> TaskPlan:
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You decompose robot tasks into atomic actions. "
                        "Return JSON with keys actions and objects. "
                        "Keep actions short, sequential, and executable."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Example task: Put all the red blocks into the bowl\n"
                        "Example output: {\"actions\": [\"find red block\", \"move to red block\", "
                        "\"pick up red block\", \"find bowl\", \"move to bowl\", "
                        "\"place block in bowl\", \"repeat until done\"], "
                        "\"objects\": [\"red block\", \"bowl\"]}\n\n"
                        f"Task: {task}"
                    ),
                },
            ],
        )
        data = json.loads(response.choices[0].message.content)
        return TaskPlan(
            actions=data.get("actions", []),
            objects=data.get("objects", []),
            metadata={"task": task, "model": self.config.model},
        )


class LLMController:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def step(self, task: str, scene: Scene, robot_state: RobotState) -> Action | None:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a robot controller. Return only JSON with keys "
                            "x, y, z, rotation, gripper, done. Keep moves small and safe."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Task: {task}\n"
                            f"Robot: {self._format_robot_state(robot_state)}\n"
                            f"Scene: {self._format_scene(scene)}"
                        ),
                    },
                ],
            )
            data = json.loads(response.choices[0].message.content)
        except Exception as exc:
            logger.error("Failed to generate action: %s", exc)
            return None

        try:
            return Action(
                x=float(data.get("x", robot_state.x)),
                y=float(data.get("y", robot_state.y)),
                z=float(data.get("z", robot_state.z)),
                rotation=float(data.get("rotation", robot_state.rotation)),
                gripper=int(data.get("gripper", robot_state.gripper)),
                done=bool(data.get("done", False)),
            )
        except (TypeError, ValueError) as exc:
            logger.error("Invalid action payload: %s", exc)
            return None

    @staticmethod
    def _format_scene(scene: Scene) -> list[dict[str, float | str]]:
        return [{"name": obj.name, "x": obj.x, "y": obj.y, "z": obj.z} for obj in scene.objects]

    @staticmethod
    def _format_robot_state(state: RobotState) -> dict[str, float | int]:
        return {
            "x": state.x,
            "y": state.y,
            "z": state.z,
            "rotation": state.rotation,
            "gripper": state.gripper,
        }