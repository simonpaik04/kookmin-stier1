import os

import cv2
import numpy as np

from . import config as cfg
from .traffic_light_decider import TrafficLightDecider


class VisionVisualizer:
    """Debug visualization for line and traffic-light pipeline outputs."""

    def __init__(self):
        self._pil_font_cache = {}

    def build_images(self, line, traffic, bev_processor, raw_frame=None):
        # Final Result: compact debug overlay for the active perception outputs.
        lane_mask = line["lane_mask"]
        stop_line = line["stop_line"]
        stopline_detected = bool(line.get("stopline_detect", False))
        stopline_distance = float(line.get("stopline_distance", -1.0))
        lane_objects = line["lane_objects"]
        schoolzone_detected = bool(line.get("schoolzone_detect", False))

        h, w = lane_mask.shape[:2]
        overlay_vis = np.zeros((h, w, 3), dtype=np.uint8)

        self._draw_stop_line(overlay_vis, stop_line)

        final_img = None
        front_yolo_img = None
        if getattr(cfg, "PUBLISH_DEBUG_IMAGES", True):
            final_img = overlay_vis.copy()
            self.draw_lane_objects(final_img, lane_objects)
            self.draw_yolo_overlay(
                final_img,
                traffic.get("detections"),
                traffic.get("traffic_light_status", "없음"),
            )
            self.draw_schoolzone_status(final_img, schoolzone_detected)
            self.draw_stopline_status(final_img, stopline_distance, stopline_detected)

            if raw_frame is not None:
                front_yolo_img = raw_frame.copy()
                self.draw_yolo_boxes(
                    front_yolo_img,
                    traffic.get("detections"),
                    traffic.get("traffic_light_status", "없음"),
                )
                self.draw_schoolzone_status(front_yolo_img, schoolzone_detected)
                self.draw_stopline_status(front_yolo_img, stopline_distance, stopline_detected)

        return {
            "final": final_img,
            "front_yolo": front_yolo_img,
        }

    @staticmethod
    def _draw_stop_line(overlay_vis, stop_line):
        if stop_line is None:
            return

        x1, y1, x2, y2 = stop_line
        center_y = int(round((y1 + y2) / 2.0))
        cv2.line(overlay_vis, (x1, center_y), (x2, center_y), (0, 255, 255), 3)
        cv2.putText(
            overlay_vis,
            "STOP LINE",
            (max(0, x1), max(14, center_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def draw_lane_objects(image, lane_objects):
        if image is None:
            return image
        h, w = image.shape[:2]
        m_per_px = max(1e-6, float(getattr(cfg, "M_PER_PIXEL", 0.04)))
        bottom_forward_px = float(getattr(cfg, "BEV_BOTTOM_FORWARD_M", 0.75)) / m_per_px
        for obj in lane_objects or []:
            px = int(round((w / 2.0) - float(obj["y"]) / m_per_px))
            py = int(round(h + bottom_forward_px - float(obj["x"]) / m_per_px))
            color = (
                (255, 255, 255)
                if int(obj.get("label", 0)) == int(getattr(cfg, "LANE_OBJECT_WHITE_LABEL", 2))
                else (0, 255, 255)
            )
            cv2.circle(image, (px, py), 3, color, -1, cv2.LINE_AA)
        return image

    def draw_yolo_overlay(self, final_img, detections, traffic_light_status):
        if final_img is None:
            return final_img

        summary_y = 136
        for det in detections or []:
            name = det["name"]
            if not TrafficLightDecider.is_traffic_light_name(name):
                continue
            label = f"{name} {det['conf']:.2f}"
            signal = det.get("signal", {})
            state_en = signal.get("state_en")
            if state_en and state_en != "NONE":
                label += f" {state_en}"
            cv2.putText(
                final_img,
                label,
                (8, summary_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            summary_y += 18

        self.draw_traffic_light_status(final_img, traffic_light_status)
        return final_img

    def draw_yolo_boxes(self, image, detections, traffic_light_status):
        """Draw YOLO boxes in front-camera pixel coordinates."""
        if image is None:
            return image

        h, w = image.shape[:2]
        for det in detections or []:
            bbox = det.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            signal = det.get("signal", {}) or {}
            state_en = signal.get("state_en", "NONE")
            color = self._traffic_state_color(state_en)
            name = str(det.get("name", "object"))
            confidence = float(det.get("conf", 0.0))
            label = f"{name} {confidence:.2f}"
            if state_en and state_en != "NONE":
                label += f" {state_en}"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )
            text_y = max(text_h + baseline + 2, y1)
            cv2.rectangle(
                image,
                (x1, text_y - text_h - baseline - 4),
                (min(w - 1, x1 + text_w + 6), text_y + 2),
                color,
                -1,
            )
            cv2.putText(
                image,
                label,
                (x1 + 3, text_y - baseline - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        self.draw_traffic_light_status(image, traffic_light_status)
        return image

    @staticmethod
    def _traffic_state_color(state):
        return {
            "STOP": (0, 0, 255),
            "YELLOW": (0, 255, 255),
            "GO": (0, 220, 0),
            "LEFT": (255, 180, 0),
        }.get(state, (255, 255, 0))

    def draw_traffic_light_status(self, image, status):
        text = f"신호: {status}"
        color_map = {
            "정지": (0, 0, 255),
            "노랑": (0, 255, 255),
            "직진": (0, 220, 0),
            "좌회전": (0, 220, 0),
        }
        color = color_map.get(status, (220, 220, 220))
        self._put_korean_text(
            image,
            text,
            (8, 34),
            color,
            24,
            fallback=f"TRAFFIC: {TrafficLightDecider.status_to_ascii(status)}",
        )

    @staticmethod
    def draw_schoolzone_status(image, detected):
        if image is None:
            return image

        text = f"SCHOOL ZONE: {'ON' if detected else 'OFF'}"
        color = (0, 255, 255) if detected else (180, 180, 180)
        cv2.putText(
            image,
            text,
            (8, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        return image

    @staticmethod
    def draw_stopline_status(image, distance_m, stopline_detected):
        if image is None:
            return image

        distance_text = "STOP LINE: --"
        distance_color = (180, 180, 180)
        if distance_m >= 0.0:
            distance_text = f"STOP LINE: {distance_m:.2f} m"
            distance_color = (0, 255, 255)
        cv2.putText(
            image,
            distance_text,
            (8, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            distance_color,
            2,
            cv2.LINE_AA,
        )

        trigger_distance = float(getattr(cfg, "STOP_LINE_TRIGGER_DISTANCE_M", 10.0))
        bool_text = f"/stoplane: {'TRUE' if stopline_detected else 'FALSE'} <= {trigger_distance:.1f}m"
        bool_color = (0, 0, 255) if stopline_detected else (180, 180, 180)
        if distance_m >= 0.0 and not stopline_detected:
            bool_color = (0, 255, 255)
        cv2.putText(
            image,
            bool_text,
            (8, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            bool_color,
            2,
            cv2.LINE_AA,
        )
        return image

    def _put_korean_text(self, image, text, org, color, size, fallback=None):
        font_path = self._find_korean_font()
        if font_path is None:
            cv2.putText(
                image,
                fallback or text.encode("ascii", "ignore").decode("ascii"),
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
                cv2.LINE_AA,
            )
            return
        try:
            from PIL import Image, ImageDraw, ImageFont

            font_key = (font_path, int(size))
            font = self._pil_font_cache.get(font_key)
            if font is None:
                font = ImageFont.truetype(font_path, int(size))
                self._pil_font_cache[font_key] = font
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil_img)
            draw.text(org, text, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))
            image[:, :] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            cv2.putText(
                image,
                fallback or text.encode("ascii", "ignore").decode("ascii"),
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _find_korean_font():
        configured = getattr(cfg, "TRAFFIC_LIGHT_FONT_PATH", None)
        candidates = [
            configured,
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None
