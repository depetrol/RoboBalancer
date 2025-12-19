import pyrealsense2 as rs # type: ignore

from camera.detect import PingPongBallDetector
import numpy as np
import cv2

class RealsenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(self.config)
        color_sensor = profile.get_device().first_color_sensor()

        # Let the camera handle exposure and white balance for natural color
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.white_balance, 3000)
        color_sensor.set_option(rs.option.exposure, 400)
        # Keep everything else near neutral / default
        color_sensor.set_option(rs.option.brightness, 30.0)          # neutral
        color_sensor.set_option(rs.option.contrast, 50.0)           # typical mid-range
        color_sensor.set_option(rs.option.saturation, 50.0)         # do not over-saturate
        color_sensor.set_option(rs.option.sharpness, 50.0)          # avoid oversharpening halos
        color_sensor.set_option(rs.option.gamma, 100.0)             # near-neutral gamma
        color_sensor.set_option(rs.option.gain, 0.0)                # let auto-exposure handle brightness


    def get_color_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        if not color_frame:
            return None
        return color_image

    def stop(self):
        self.pipeline.stop()

def main():
    detector = PingPongBallDetector()
    camera = RealsenseCamera()
    try:
        while True:
            frame = camera.get_color_frame()
            if frame is None:
                continue

            result, pos = detector.detect_ping_pong_ball(frame)
            print("Detected position:", pos)
            cv2.imshow("RealSense Ping-Pong Detection", result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        
    finally:
        camera.stop()

if __name__ == "__main__":
    main()
