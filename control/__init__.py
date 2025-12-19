from typing import List
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface
import numpy as np
import math

class Robot:

    def __init__(self, robot_ip: str, initial_position = None, default_speed = 3, default_accl = 120):
        self.ip = robot_ip
        self.rtde_c = RTDEControlInterface(self.ip)
        self.rtde_r = RTDEReceiveInterface(self.ip)
        self.default_speed = default_speed
        self.default_accl = default_accl
        self.end_freedrive()
        if initial_position == None:
            self.initial_position = np.array([0, -0.4, 0.25, math.pi / 2, 0, 0])
        else:
            assert len(initial_position) == 6
            self.initial_position = np.array(initial_position)

    def clip_pos(self, goal_pos: np.array) -> List[float]:
        assert goal_pos.shape == (6,)
        init = np.asarray(self.initial_position, dtype=float)
        goal = np.asarray(goal_pos, dtype=float)
        low = init.copy()
        high = init.copy()
        low[:3]  -= 0.2
        high[:3] += 0.2
        low[3:]  -= 0.5
        high[3:] += 0.5
        clipped = np.clip(goal, low, high)
        if np.any(clipped != goal_pos):
            print(f"Action clipped to {clipped}")
        return clipped.tolist()
        

    def home(self):
        self.movel(self.initial_position, self.default_speed, self.default_accl)

    def move_diff(self, diff: np.array):
        diff = np.array(diff)
        assert len(diff) == 6
        current_pose = self.get_pose()
        target_pose = current_pose + diff
        target_pose = self.clip_pos(target_pose)
        assert len(target_pose) == 6
        self.movel(target_pose, speed=self.default_speed, acceleration=self.default_accl)

    def move_to_pose(self, target_pose: np.array):
        assert len(target_pose) == 6
        target_pose = self.clip_pos(target_pose)
        assert len(target_pose) == 6
        self.movel(target_pose, speed=self.default_speed, acceleration=self.default_accl)
    def get_pose(self):
        tcp_pose = self.rtde_r.getActualTCPPose()
        return np.array(tcp_pose)

    def get_joint_pos(self):
        joint_positions = self.rtde_r.getActualQ()
        return np.array(joint_positions)

    def get_tcp_speed(self):
        return np.array(self.rtde_r.getActualTCPSpeed())

    def get_joint_speeds(self):
        return np.array(self.rtde_r.getActualQd())

    def movej(self, q, speed: float = 1.0, acceleration: float = 1.2):
        return self.rtde_c.moveJ(q, speed, acceleration)

    def movel(self, pose, speed: float = 0.25, acceleration: float = 1.2):
        pose = list(pose)
        assert len(pose) == 6
        return self.rtde_c.moveL(pose, speed, acceleration, True)

    def speedj(self, qd, acceleration: float = 1.2, time: float = 0.5):
        return self.rtde_c.speedJ(qd, acceleration, time)

    def speedl(self, xd, acceleration: float = 1.2, time: float = 0.5):
        return self.rtde_c.speedL(xd, acceleration, time)

    def stopj(self, a: float = 1.2):
        return self.rtde_c.stopJ(a)

    def stopl(self, a: float = 1.2):
        return self.rtde_c.stopL(a)

    def servoj(
        self,
        q,
        speed: float = 1.0,
        acceleration: float = 1.2,
        time: float = 0.008,
        lookahead_time: float = 0.1,
        gain: float = 300.0,
    ):
        return self.rtde_c.servoJ(q, speed, acceleration, time, lookahead_time, gain)

    def set_tcp(self, tcp_pose):
        return self.rtde_c.setTcp(tcp_pose)

    def set_payload(self, mass: float, cog):
        return self.rtde_c.setPayload(mass, cog)
    
    def start_freedrive(self):
        self.rtde_c.freedriveMode()
        print("Robot in freedrive mode")

    def end_freedrive(self):
        self.rtde_c.endFreedriveMode()


    def is_moving(self) -> bool:
        qd = self.get_joint_speeds()
        return any(abs(x) > 1e-3 for x in qd)

    def stop(self, a: float = 1.2):
        self.stopj(a)
        self.stopl(a)

    def get_tcp_force(self):
        return self.rtde_r.getActualTCPForce()

    def close(self):
        try:
            self.rtde_c.stopScript()
        except Exception:
            pass
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
