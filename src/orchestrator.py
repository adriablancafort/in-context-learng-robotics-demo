import logging
from typing import Any

from src.llm import LLMController, TaskPreprocessor
from src.models import DEFAULT_CONFIG, RobotState, SystemConfig
from src.runtime import PerceptionModule, RobotInterface, SafetyFilter

logger = logging.getLogger(__name__)


class InCoRoOrchestrator:
    def __init__(self, config: SystemConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self.preprocessor = TaskPreprocessor(self.config.llm)
        self.perception = PerceptionModule()
        self.controller = LLMController(self.config.llm)
        self.safety_filter = SafetyFilter(self.config.robot)
        self.robot = RobotInterface()
        self._task_log: list[dict[str, Any]] = []

    def run_task(self, task: str, max_iterations: int = 100, max_steps_per_action: int = 10) -> bool:
        plan = self.preprocessor.decompose_task(task)
        for atomic_action in plan.actions:
            completed = self._run_atomic_action(atomic_action, max_iterations, max_steps_per_action)
            if not completed:
                logger.warning("Atomic action did not complete: %s", atomic_action)
        self._task_log.append({"task": task, "plan": plan, "robot_log": self.robot.get_execution_log()})
        return True

    def _run_atomic_action(self, atomic_action: str, max_iterations: int, max_steps: int) -> bool:
        steps = 0
        for _ in range(max_iterations):
            scene = self.perception.get_scene()
            robot_state = self.robot.get_state()
            action = self.controller.step(atomic_action, scene, robot_state)
            if action is None:
                continue
            safe_action = self.safety_filter.clip_action(action, robot_state)
            self.robot.execute(safe_action)
            steps += 1
            if safe_action.done:
                return True
            if steps >= max_steps:
                return False
        return False

    def get_robot_state(self) -> RobotState:
        return self.robot.get_state()

    def get_execution_log(self) -> list[dict[str, Any]]:
        return self._task_log.copy()

    def reset(self) -> None:
        self.robot.reset_to_home()
        self._task_log.clear()

    def close(self) -> None:
        self.perception.close()