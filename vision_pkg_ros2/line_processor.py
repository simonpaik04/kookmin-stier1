import cv2

from . import config as cfg
from . import vision_utils
from .bev import BevProcessor
from .lane_object_clusterer import LaneObjectClusterer
from .schoolzone_detector import SchoolzoneDetector


class LineProcessor:
    """Line Process: Image -> BEV -> masks -> clustering -> bool detections."""

    def __init__(self):
        self.bev = BevProcessor()
        self.clusterer = LaneObjectClusterer()
        self.schoolzone_detector = SchoolzoneDetector()

    def rebuild_bev(self):
        return self.bev.rebuild()

    def process(self, frame):
        bev_color = self.bev.warp(frame)
        bev_img = vision_utils.keep_lane_colors(bev_color)
        start_black_box = vision_utils.find_largest_black_box(bev_color)
        bev_img = vision_utils.black_out_box(bev_img, start_black_box)

        # Yellow mask / White mask.
        white_mask, yellow_mask = vision_utils.split_white_yellow_masks(bev_img)

        # Stopline_Detect.
        stop_line = vision_utils.detect_stop_line(white_mask)
        stopline_distance = vision_utils.stop_line_distance_m(stop_line, white_mask.shape[0])
        stopline_detect = self._stopline_in_trigger_range(stop_line, stopline_distance)
        lane_white_mask, lane_yellow_mask = vision_utils.remove_stop_line_from_masks(
            stop_line,
            (white_mask, yellow_mask),
        )

        object_white_mask = lane_white_mask.copy()
        object_yellow_mask = lane_yellow_mask.copy()
        self._remove_center_yellow_for_lane_mask(lane_yellow_mask)

        lane_mask = cv2.bitwise_or(lane_white_mask, lane_yellow_mask)

        # Clustering.
        lane_objects = self.clusterer.build(object_white_mask, object_yellow_mask)

        # Schoolzone_Detect: ON when at least the configured number of yellow
        # clustering objects are present.
        schoolzone_detect = self.schoolzone_detector.detect(lane_objects)

        return {
            "bev": bev_img,
            "lane_mask": lane_mask,
            "stop_line": stop_line,
            "stopline_detect": stopline_detect,
            "stopline_distance": stopline_distance,
            "lane_objects": lane_objects,
            "schoolzone_detect": schoolzone_detect,
        }

    @staticmethod
    def _stopline_in_trigger_range(stop_line, stopline_distance):
        stopline_trigger_distance = float(getattr(cfg, "STOP_LINE_TRIGGER_DISTANCE_M", 10.0))
        return (
            stop_line is not None
            and 0.0 <= stopline_distance <= stopline_trigger_distance
        )

    @staticmethod
    def _remove_center_yellow_for_lane_mask(lane_yellow_mask):
        if lane_yellow_mask is None or not getattr(cfg, "IGNORE_CENTER_YELLOW", True):
            return

        ignore_width = int(getattr(cfg, "CENTER_YELLOW_IGNORE_WIDTH_PX", 90))
        if ignore_width <= 0:
            return

        _, w_mask = lane_yellow_mask.shape[:2]
        cx = w_mask // 2
        x1 = max(0, cx - ignore_width // 2)
        x2 = min(w_mask, cx + (ignore_width + 1) // 2)
        lane_yellow_mask[:, x1:x2] = 0
