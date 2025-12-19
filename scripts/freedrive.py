from control import Robot
from scripts.config import ROBOT_IP
import pickle
if __name__ == "__main__":
    robot = Robot(ROBOT_IP)

    current_pose = None
    robot.start_freedrive()
    print("Starting Freedrive Mode")
    try:
        while True:
            current_pose = robot.get_pose()
            print(f"Freedrive: {robot.get_pose()}")
    except KeyboardInterrupt as e:
        exit()
