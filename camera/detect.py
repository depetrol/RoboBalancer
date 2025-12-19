import numpy as np
import cv2


class PingPongBallDetector:

    def __init__(self):

        self.lower_orange = np.array([5, 100, 150])
        self.upper_orange = np.array([50, 200, 200])

        self.min_area = 80
        self.min_radius = 25
        self.max_radius = 30

        self.min_visible_fraction = 0.15

    def _detect_with_hough(self, mask, frame_shape):
        h, w = frame_shape[:2]

        blurred_mask = cv2.GaussianBlur(mask, (9, 9), 2)

        try:
            circles = cv2.HoughCircles(
                blurred_mask,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=30,
                param1=100,
                param2=10,
                minRadius=self.min_radius,
                maxRadius=self.max_radius if self.max_radius is not None else 0,
            )
        except cv2.error:
            circles = None

        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype("int")

        best_circle = None
        best_score = 0.0

        for (cx, cy, r) in circles:
            if r < self.min_radius:
                continue
            if self.max_radius is not None and r > self.max_radius:
                continue

            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                continue

            circle_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.circle(circle_mask, (cx, cy), r, 255, -1)

            visible = cv2.bitwise_and(mask, mask, mask=circle_mask)
            visible_area = cv2.countNonZero(visible)
            circle_area = np.pi * (r ** 2)

            if circle_area <= 0:
                continue

            visible_fraction = float(visible_area / circle_area)

            score = visible_fraction * circle_area

            if score > best_score and visible_fraction >= self.min_visible_fraction:
                best_score = score
                best_circle = (cx, cy, r, visible_fraction)

        if best_circle is None:
            return None

        return best_circle

    def detect_ping_pong_ball(self, frame):

        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)

        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        h, w = frame.shape[:2]

        hough_result = self._detect_with_hough(mask, frame.shape)
        if hough_result is not None:
            cx, cy, radius, conf = hough_result

            x1, y1 = cx - radius, cy - radius
            x2, y2 = cx + radius, cy + radius

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"ping-pong {conf:.2f} ({cx}, {cy})",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            return frame, (cx, cy)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return frame, None

        best_cnt = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)

            if radius < self.min_radius:
                continue
            if self.max_radius is not None and radius > self.max_radius:
                continue

            circle_area = np.pi * (radius ** 2)
            visible_fraction = float(area / circle_area) if circle_area > 0 else 0.0

            shape_score = 0.6 * circularity + 0.4 * visible_fraction
            score = shape_score * area

            if score > best_score:
                best_score = score
                best_cnt = cnt

        if best_cnt is None:
            return frame, None

        (cx, cy), radius = cv2.minEnclosingCircle(best_cnt)
        cx, cy, radius = int(cx), int(cy), int(radius)

        x1, y1 = cx - radius, cy - radius
        x2, y2 = cx + radius, cy + radius

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        area = cv2.contourArea(best_cnt)
        perimeter = cv2.arcLength(best_cnt, True)
        circularity = 0.0
        if perimeter > 0:
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        conf = float(np.clip(circularity, 0.0, 1.0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame,
            f"ping-pong {conf:.2f} ({cx}, {cy})",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return frame, (cx, cy)
