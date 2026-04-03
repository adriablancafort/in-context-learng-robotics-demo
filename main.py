import logging
from dotenv import load_dotenv

load_dotenv()

from src.models import DEFAULT_CONFIG
from src.orchestrator import InCoRoOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    print("=" * 60)
    print("InCoRo: In-Context Learning for Robotics MVP")
    print("=" * 60)
    print()

    orchestrator = InCoRoOrchestrator(config=DEFAULT_CONFIG)

    task = "Pick up the red block and place it in the bowl"

    print(f"Task: {task}")
    print()

    try:
        orchestrator.run_task(task, max_iterations=50, max_steps_per_action=10)
        print()
        print("=" * 60)
        print("Execution Complete")
        print("=" * 60)
        final_state = orchestrator.get_robot_state()
        print(f"Final robot position: ({final_state.x:.1f}, {final_state.y:.1f}, {final_state.z:.1f})")
        print(f"Gripper: {'CLOSED' if final_state.gripper else 'OPEN'}")
        log = orchestrator.get_execution_log()
        if log:
            print(f"\nTotal actions executed: {len(log[0]['robot_log'])}")
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
