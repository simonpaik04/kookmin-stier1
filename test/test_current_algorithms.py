import cv2
import numpy as np

from vision_pkg_ros2 import config as cfg
from vision_pkg_ros2 import vision_utils
from vision_pkg_ros2.lane_object_clusterer import LaneObjectClusterer
from vision_pkg_ros2.line_processor import LineProcessor
from vision_pkg_ros2.schoolzone_detector import SchoolzoneDetector
from vision_pkg_ros2.traffic_light_classifier import TrafficLightColorClassifier
from vision_pkg_ros2.traffic_light_decider import TrafficLightDecider


def test_black_start_box_covers_checker_and_preserves_boundary_lane():
    bev = np.full((200, 240, 3), 100, dtype=np.uint8)
    bev[:, 5:10] = 255
    bev[70:130, 20:220] = 0
    for row, y in enumerate(range(70, 130, 20)):
        for col, x in enumerate(range(20, 220, 20)):
            if (row + col) % 2 == 0:
                bev[y:y + 20, x:x + 20] = 255

    box = vision_utils.find_largest_black_box(bev)
    assert box == (20, 70, 220, 130)

    lane_image = vision_utils.keep_lane_colors(bev)
    filtered = vision_utils.black_out_box(lane_image, box)
    checker_gray = cv2.cvtColor(filtered[70:130, 20:220], cv2.COLOR_BGR2GRAY)
    boundary_gray = cv2.cvtColor(filtered[:, 5:10], cv2.COLOR_BGR2GRAY)
    assert cv2.countNonZero(checker_gray) == 0
    assert cv2.countNonZero(boundary_gray) > 0


def test_lane_cluster_minimum_is_40_pixels():
    clusterer = LaneObjectClusterer()
    empty = np.zeros((100, 100), dtype=np.uint8)
    under = empty.copy()
    under[2:5, 2:15] = 255  # 39 pixels
    threshold = empty.copy()
    threshold[2:6, 2:12] = 255  # 40 pixels

    assert clusterer.build(under, empty) == []
    objects = clusterer.build(threshold, empty)
    assert len(objects) == 1
    assert objects[0]["label"] == cfg.LANE_OBJECT_WHITE_LABEL


def test_schoolzone_turns_on_at_20_yellow_clusters():
    detector = SchoolzoneDetector()
    yellow = {"label": cfg.LANE_OBJECT_YELLOW_LABEL}
    assert detector.detect([yellow] * 19) is False
    assert detector.detect([yellow] * 20) is True


def test_stopline_distance_uses_nearest_edge_and_missing_is_minus_one():
    mask = np.zeros((200, 250), dtype=np.uint8)
    mask[120:132, 20:230] = 255
    stop_line = vision_utils.detect_stop_line(mask)
    assert stop_line is not None

    _, _, _, nearest_y = stop_line
    expected = (
        mask.shape[0]
        + cfg.BEV_BOTTOM_FORWARD_M / cfg.M_PER_PIXEL
        - nearest_y
    ) * cfg.M_PER_PIXEL
    assert vision_utils.stop_line_distance_m(stop_line, mask.shape[0]) == expected
    assert vision_utils.stop_line_distance_m(None, mask.shape[0]) == -1.0


def test_stopline_bool_waits_until_trigger_distance():
    original = cfg.STOP_LINE_TRIGGER_DISTANCE_M
    try:
        cfg.STOP_LINE_TRIGGER_DISTANCE_M = 10.0
        stop_line = (0, 100, 200, 110)

        assert LineProcessor._stopline_in_trigger_range(stop_line, 10.1) is False
        assert LineProcessor._stopline_in_trigger_range(stop_line, 10.0) is True
        assert LineProcessor._stopline_in_trigger_range(stop_line, 4.0) is True
        assert LineProcessor._stopline_in_trigger_range(None, -1.0) is False
    finally:
        cfg.STOP_LINE_TRIGGER_DISTANCE_M = original


def test_traffic_light_color_truth_table_and_150_pixel_minimum():
    classifier = TrafficLightColorClassifier()
    bbox = (0, 0, 80, 80)

    def classify(red=False, green=False, size=15):
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        if red:
            frame[10:10 + size, 10:10 + size] = (0, 0, 255)
        if green:
            frame[40:40 + size, 40:40 + size] = (0, 255, 0)
        return classifier.classify(frame, bbox)["state_en"]

    assert classify(red=True) == "STOP"
    assert classify(red=True, green=True) == "LEFT"
    assert classify(green=True) == "GO"
    assert classify(red=True, size=10) == "NONE"


def test_traffic_light_decision_separates_stop_from_none():
    assert cfg.TRAFFIC_DECIDE_NONE == 0
    assert cfg.TRAFFIC_DECIDE_GO == 1
    assert cfg.TRAFFIC_DECIDE_LEFT == 2
    assert cfg.TRAFFIC_DECIDE_STOP == 3
    assert cfg.TRAFFIC_DECIDE_YELLOW == 3

    assert TrafficLightDecider.status_to_decision("STOP") == cfg.TRAFFIC_DECIDE_STOP
    assert TrafficLightDecider.status_to_decision("NONE") == cfg.TRAFFIC_DECIDE_NONE
    assert TrafficLightDecider.status_to_decision("없음") == cfg.TRAFFIC_DECIDE_NONE
    assert TrafficLightDecider.status_to_decision("GO") == cfg.TRAFFIC_DECIDE_GO
    assert TrafficLightDecider.status_to_decision("LEFT") == cfg.TRAFFIC_DECIDE_LEFT
    assert TrafficLightDecider.status_to_decision("YELLOW") == cfg.TRAFFIC_DECIDE_YELLOW
    assert TrafficLightDecider.status_to_decision("unexpected") == cfg.TRAFFIC_DECIDE_NONE
