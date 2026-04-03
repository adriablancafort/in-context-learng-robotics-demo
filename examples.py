import logging
from src.models import Action, RobotState
from src.runtime import RobotInterface, SafetyFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def example_4_robot_interface():
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Robot Interface")
    print("=" * 70)

    robot = RobotInterface()

    print(f"\nInitial state: {robot.get_state()}")
    print("\nMoving to (300, 100, 80)...")

    action1 = Action(x=300, y=100, z=80, rotation=0, gripper=0)
    robot.execute(action1)
    print(f"State after move: {robot.get_state()}")
    print("\nClosing gripper...")
    robot.close_gripper()
    print(f"State with gripper closed: {robot.get_state()}")

    print("\nOpening gripper...")
    robot.open_gripper()
    print(f"State with gripper open: {robot.get_state()}")
    print("\nResetting to home...")
    robot.reset_to_home()
    print(f"State after reset: {robot.get_state()}")
    print(f"\nExecution log ({len(robot.get_execution_log())} actions):")
    for i, record in enumerate(robot.get_execution_log(), 1):
        action = record["action"]
        result = record["result"]
        print(f"  {i}. {action} -> {result}")


def example_5_safety_constraints():
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Safety Constraints Demo")
    print("=" * 70)

    safety = SafetyFilter()
    state = RobotState(x=200, y=200, z=50)

    test_cases = [
        ("Valid movement", Action(x=210, y=210, z=60, done=False)),
        ("Out of bounds (x)", Action(x=500, y=200, z=50, done=False)),
        ("Out of bounds (z)", Action(x=200, y=200, z=200, done=False)),
        ("Large delta", Action(x=300, y=200, z=50, done=False)),
    ]

    print(f"\nRobot at: ({state.x}, {state.y}, {state.z})\n")

    for test_name, action in test_cases:
        print(f"{test_name}:")
        print(f"  Target: ({action.x}, {action.y}, {action.z})")

        validated = safety.validate_action(action, state)
        if validated:
            print(f"  ✓ Valid")
        else:
            print(f"  ✗ Invalid")
            clipped = safety.clip_action(action, state)
            print(f"  Clipped to: ({clipped.x}, {clipped.y}, {clipped.z})")


def main():
    print("\n" + "#" * 70)
    print("# InCoRo MVP: Examples and Demonstrations")
    print("#" * 70)

    example_4_robot_interface()
    example_5_safety_constraints()

    print("\n" + "#" * 70)
    print("# Examples Complete")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
