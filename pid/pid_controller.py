import numpy as np
import time

PHYSICAL_WIDTH_METERS = 0.30

CAMERA_WIDTH_PIXELS   = 640.0

METERS_PER_PIXEL = PHYSICAL_WIDTH_METERS / CAMERA_WIDTH_PIXELS
print(f"Conversion Ratio: {METERS_PER_PIXEL:.6f} m/px")


SIM_KP = 1.50
SIM_KI = 0.10
SIM_KD = 0.80

kp = SIM_KP * METERS_PER_PIXEL
ki = SIM_KI * METERS_PER_PIXEL
kd = SIM_KD * METERS_PER_PIXEL

print(f"Converted Real-World Gains -> Kp: {kp:.6f}, Ki: {ki:.6f}, Kd: {kd:.6f}")


class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, tau=0.1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        
        self.tau = tau 

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.last_time = None

    def compute(self, measurement, dt=None):
        current_time = time.time()

        if dt is None:
            if self.last_time is None:
                self.last_time = current_time
                return 0.0
            dt = current_time - self.last_time
            self.last_time = current_time

        if dt <= 0.0:
            return 0.0

        error = self.setpoint - measurement

        self.integral += error * dt

        alpha = 0.0
        if (self.tau + dt) > 0:
            alpha = dt / (self.tau + dt)
            
        raw_derivative = (error - self.prev_error) / dt
        derivative = (alpha * raw_derivative) + ((1.0 - alpha) * self.prev_derivative)

        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

        self.prev_error = error
        self.prev_derivative = derivative

        return output
    
    def set_target(self, setpoint):
        self.setpoint = setpoint

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.last_time = None


class PIDController:
    def __init__(self, robot_home: np.ndarray, 
                 kp_x=kp, ki_x=ki, kd_x=kd, 
                 kp_y=kp, ki_y=ki, kd_y=kd,
                 deadband=5.0):
        
        self.robot_home = robot_home
        
        self.deadband = deadband

        self.pid_x = PID(kp_x, ki_x, kd_x, setpoint=0.0, tau=0.1)
        self.pid_y = PID(kp_y, ki_y, kd_y, setpoint=0.0, tau=0.1)
        
        self.ball_setpoint = np.array([0.0, 0.0])

    def set_setpoint(self, ball_setpoint: np.ndarray):
        assert ball_setpoint.shape == (2,)
        self.ball_setpoint = ball_setpoint
        
        self.pid_x.set_target(ball_setpoint[0])
        self.pid_y.set_target(ball_setpoint[1])

    def set_robot_home(self, robot_home: np.ndarray):
        assert robot_home.shape == (6,)
        self.robot_home = robot_home

    def control(self, measurement: np.ndarray) -> np.ndarray:
        assert measurement.shape == (2,)
        
        error_x = self.ball_setpoint[0] - measurement[0]
        error_y = self.ball_setpoint[1] - measurement[1]

        if abs(error_x) < self.deadband:
            _ = self.pid_x.compute(measurement[0])
            self.pid_x.integral = 0.0
            self.pid_x.prev_derivative = 0.0
            control_x = 0.0
        else:
            control_x = self.pid_x.compute(measurement[0])

        if abs(error_y) < self.deadband:
            _ = self.pid_y.compute(measurement[1])
            self.pid_y.integral = 0.0
            self.pid_y.prev_derivative = 0.0
            control_y = 0.0
        else:
            control_y = self.pid_y.compute(measurement[1])

        max_angle = 0.78
        control_x = max(min(control_x, max_angle), -max_angle)
        control_y = max(min(control_y, max_angle), -max_angle)

        target_pose = self.robot_home.copy()
        target_pose[3] += control_x
        target_pose[4] += control_y

        return target_pose


if __name__ == "__main__":
    import cv2
    from scripts.config import ROBOT_IP
    from control import Robot
    from camera.realsense import RealsenseCamera
    from camera.detect import PingPongBallDetector

    detector = PingPongBallDetector()
    camera = RealsenseCamera()
    time.sleep(1.0)

    robot = Robot(ROBOT_IP)
    
    initial_pose = robot.get_pose()
    print(f"Initial Robot Pose: {initial_pose}")

    pid = PIDController(initial_pose, deadband=5.0)
    
    pid.set_setpoint(np.array([CAMERA_WIDTH_PIXELS/2.0, 240.0])) 

    np.set_printoptions(
        formatter={"float_kind": lambda x: f"{float(x):7.3f}"}
    )

    smoothed_pos = None
    alpha = 0.3

    last_cmd_pose = initial_pose.copy()
    last_cmd_time = time.time()

    min_angle_step = 0.005

    max_command_interval = 0.15

    try:
        while True:
            frame = camera.get_color_frame()
            result, pos = detector.detect_ping_pong_ball(frame)
            cv2.imshow("PID Control Ping-Pong Detection", result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
            if pos is not None:
                pos = np.array(pos, dtype=float)

                if smoothed_pos is None:
                    smoothed_pos = pos
                else:
                    smoothed_pos = alpha * pos + (1.0 - alpha) * smoothed_pos

                target_pose = pid.control(smoothed_pos)
                
                delta_rx = abs(target_pose[3] - last_cmd_pose[3])
                delta_ry = abs(target_pose[4] - last_cmd_pose[4])
                now = time.time()
                time_since_last = now - last_cmd_time

                should_send = (
                    (delta_rx > min_angle_step) or 
                    (delta_ry > min_angle_step) or
                    (time_since_last > max_command_interval)
                )

                if should_send:
                    print(f"Target: {target_pose}")
                    robot.move_to_pose(target_pose)
                    last_cmd_pose = target_pose.copy()
                    last_cmd_time = now

    finally:
        camera.stop()
        cv2.destroyAllWindows()
