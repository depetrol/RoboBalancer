from control import Robot
from scripts.config import ROBOT_IP


if __name__ == "__main__":
    
    robot = Robot(ROBOT_IP)

    robot.home()
     
