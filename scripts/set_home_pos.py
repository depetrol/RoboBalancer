from control import Robot
from scripts.config import ROBOT_IP, HOME_PKL_LOC
import pickle

def get_home_pos():
    with open(HOME_PKL_LOC, "rb") as f:
        home_pos = pickle.load(f)
    return home_pos

if __name__ == "__main__":
    robot = Robot(ROBOT_IP)

    robot.home()
    current_pose = None
    robot.start_freedrive()
    print("Starting Freedrive Mode")
    try:
        while True:
            current_pose = robot.get_pose()
            print(f"Pose: {robot.get_pose()}")
    except KeyboardInterrupt as e:
        location = HOME_PKL_LOC
        print(f"Saving to {HOME_PKL_LOC}")
        with open(HOME_PKL_LOC, "wb") as f:
            pickle.dump(current_pose, f)
        exit()
