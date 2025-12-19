from control import Robot
from scripts.config import ROBOT_IP, CAMERA_PKL_LOC
import pickle

def get_camera_pos():
    with open(CAMERA_PKL_LOC, "rb") as f:
        camera_pos = pickle.load(f)
    return camera_pos

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
        location = CAMERA_PKL_LOC
        print(f"Saving to {CAMERA_PKL_LOC}")
        with open(CAMERA_PKL_LOC, "wb") as f:
            pickle.dump(current_pose, f)
        exit()
