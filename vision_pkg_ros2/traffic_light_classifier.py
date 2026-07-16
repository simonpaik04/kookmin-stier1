import cv2


class TrafficLightColorClassifier:
    """색 분류: YOLO box crop -> STOP/GO/LEFT/YELLOW 상태."""

    def classify(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pad = 3
        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(w, int(x2) + pad)
        y2 = min(h, int(y2) + pad)
        if x2 <= x1 or y2 <= y1:
            return {"state": "없음", "state_en": "NONE", "color": "unknown", "shape": "unknown"}

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 80, 90), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 80, 90), (180, 255, 255))
        masks = {
            "red": cv2.bitwise_or(red1, red2),
            "yellow": cv2.inRange(hsv, (16, 80, 100), (40, 255, 255)),
            "green": cv2.inRange(hsv, (40, 50, 75), (95, 255, 255)),
        }

        stats = {}
        for color, mask in masks.items():
            stats[color] = self.mask_stats(mask)

        min_pixels = 150  # 각 색깔이 150픽셀 이상일 때만 존재한다고 판단
        valid = {color: stat for color, stat in stats.items() if stat["pixels"] >= min_pixels}
        if not valid:
            return {"state": "없음", "state_en": "NONE", "color": "unknown", "shape": "unknown"}

        red_ok = "red" in valid
        yellow_ok = "yellow" in valid
        green_ok = "green" in valid

        # 빨강색만 있으면 정지
        if red_ok and not green_ok:
            return {"state": "정지", "state_en": "STOP", "color": "red", "shape": "circle"}

        # 빨강색 + 초록색이면 좌회전
        if red_ok and green_ok:
            return {"state": "좌회전", "state_en": "LEFT", "color": "green", "shape": "circle"}

        # 초록색만 있으면 직진
        if green_ok and not red_ok and not yellow_ok:
            return {"state": "직진", "state_en": "GO", "color": "green", "shape": "circle"}

        # 노랑색이면 정지
        if yellow_ok:
            return {"state": "정지", "state_en": "YELLOW", "color": "yellow", "shape": "circle"}
        
        # 인식 불가
        return {"state": "없음", "state_en": "NONE", "color": "unknown", "shape": "unknown"}

    @staticmethod
    def mask_stats(mask):
        pixels = int(cv2.countNonZero(mask))
        return {"pixels": pixels}
