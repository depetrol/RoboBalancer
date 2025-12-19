from control import Robot
from scripts.config import ROBOT_IP


if __name__ == "__main__":
    try:
        robot = Robot(ROBOT_IP)

        robot.home()

        diff = [0, 0, 0.1, 0, 0, 0.5]
        robot.move_diff(diff)
    except KeyboardInterrupt as e:
        exit()
