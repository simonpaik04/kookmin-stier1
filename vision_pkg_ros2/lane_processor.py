import time

import cv2
import numpy as np

from . import config as cfg
from .line_processor import LineProcessor
from .traffic_light_processor import TrafficLightProcessor
from .visualization import VisionVisualizer


class LaneProcessor:
    """Top-level vision processor.

    This class intentionally only orchestrates the two perception pipelines:
    Line Process and Traffic Light Process. Detailed work lives in smaller
    modules such as line_processor.py, bev.py, and traffic_light_processor.py.
    """

    def __init__(self, logger=None, clock=None):
        self.logger = logger
        self.clock = clock
        self._last_perf_log_time = 0.0
        self.line_processor = LineProcessor()
        self.traffic_light_processor = TrafficLightProcessor(logger=logger)
        self.visualizer = VisionVisualizer()

    def rebuild_bev_projector(self):
        projector = self.line_processor.rebuild_bev()
        self._info(
            f"[vision] BEV rebuilt: {cfg.BEV_IMAGE_WIDTH}x{cfg.BEV_IMAGE_HEIGHT}, "
            f"m_per_px={cfg.M_PER_PIXEL:.4f}, tuning={cfg.BEV_TUNING_ENABLED}"
        )
        return projector

    def process(self, frame):
        t0 = time.time()

        # Image input: frame comes from /usb_cam/image_raw/front in LaneNode.
        frame = self._undistort_if_needed(frame)
        raw_frame = frame

        # Line Process: BEV -> masks -> clustering -> Schoolzone/Stopline bools.
        line = self.line_processor.process(frame)
        t1 = time.time()

        # Traffic Light Process: Crop -> YOLO_net -> color classifier -> decide.
        traffic = self.traffic_light_processor.process(raw_frame)
        t2 = time.time()

        images = self.visualizer.build_images(
            line,
            traffic,
            self.line_processor.bev,
            raw_frame=raw_frame,
        )
        t3 = time.time()

        self._log_performance(t0, t1, t2, t3, line)

        return {
            "lane_mask": line["lane_mask"],
            "bev": line["bev"],
            "final": images["final"],
            "front_yolo": images["front_yolo"],
            "schoolzone_detect": line["schoolzone_detect"],
            "stopline_detect": line["stopline_detect"],
            "stopline_distance": line["stopline_distance"],
            "lane_objects": line["lane_objects"],
            "traffic_decide": traffic["traffic_decide"],
        }

    @staticmethod
    def _undistort_if_needed(frame):
        if (
            getattr(cfg, "ENABLE_UNDISTORT", True)
            and getattr(cfg, "DIST_COEFF", None) is not None
            and np.any(cfg.DIST_COEFF != 0)
            and getattr(cfg, "K", None) is not None
        ):
            return cv2.undistort(frame, cfg.K, cfg.DIST_COEFF)
        return frame

    def _log_performance(self, t0, t1, t2, t3, line):
        line_ms = (t1 - t0) * 1000
        traffic_ms = (t2 - t1) * 1000
        total_ms = (t3 - t0) * 1000
        target_fps = float(getattr(cfg, "TARGET_FPS", 30.0))
        shown_fps = 1000.0 / max(total_ms, 1.0)
        if target_fps > 0.0:
            shown_fps = min(shown_fps, target_fps)

        self._info_throttle(
            2.0,
            f"[vision] line={line_ms:.0f}ms  traffic={traffic_ms:.0f}ms  "
            f"total={total_ms:.0f}ms  ({shown_fps:.1f}fps)  "
            f"yolo={self.traffic_light_processor.last_yolo_count} "
            f"lane_obj={len(line['lane_objects'])}",
        )

    def _info(self, message):
        if self.logger is not None:
            self.logger.info(message)

    def _info_throttle(self, seconds, message):
        if self.logger is None:
            return
        now = time.time()
        if self.clock is not None:
            now = self.clock.now().nanoseconds / 1e9
        if now - self._last_perf_log_time >= seconds:
            self.logger.info(message)
            self._last_perf_log_time = now
