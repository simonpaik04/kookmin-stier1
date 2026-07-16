import os

import numpy as np


def package_file_path(*parts):
    """Return a package asset path that works from source or after colcon install."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share_path = os.path.join(get_package_share_directory("vision_pkg_ros2"), *parts)
        if os.path.exists(share_path):
            return share_path
    except Exception:
        pass

    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not prefix:
            continue
        share_path = os.path.join(prefix, "share", "vision_pkg_ros2", *parts)
        if os.path.exists(share_path):
            return share_path

    source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", *parts))
    if os.path.exists(source_path):
        return source_path

    return source_path

# ============================================================
# ROS 토픽
# ============================================================
IMAGE_TOPIC = "/usb_cam/image_raw/front"  # Image input: front camera topic.
FRONT_YOLO_TOPIC = "/vision/front_yolo"  # Front image with YOLO detection boxes.

# ============================================================
# 카메라 내부 파라미터 (Intrinsic)
# ============================================================
# 실제 카메라 캘리브레이션 (렌즈 왜곡 보정) — 640×360 기준
K_REAL = np.array([
    [484.99055,   0.     , 309.09636],
    [  0.     , 454.50456, 186.28021],
    [  0.     ,   0.     ,   1.     ]
], dtype=np.float32)
DIST_COEFF_REAL = np.array([-0.337306, 0.076342, -0.003823, 0.004501, 0.0], dtype=np.float32)

# 가변 파라미터 (lane_node.py에서 모드에 따라 결정됨)
K          = None                    # sim: None, real: K_REAL
DIST_COEFF = np.zeros(5, dtype=np.float32)  # sim: 0, real: DIST_COEFF_REAL
ENABLE_UNDISTORT = True              # 렌즈 왜곡 보정 ON/OFF

# ============================================================
# IPM / BEV 설정
# ============================================================
M_PER_PIXEL    = 0.04  # fast_front_rear_avm 기준 25 px/m
BEV_WIDTH_M    = 10.0  # 폭은 원본 fast AVM과 동일
BEV_LENGTH_M   = 21.0  # 전방 16m + 차량 5m

BEV_IMAGE_WIDTH  = int(BEV_WIDTH_M  / M_PER_PIXEL)   # 250 px
BEV_IMAGE_HEIGHT = int(BEV_LENGTH_M / M_PER_PIXEL)   # 525 px
BEV_TUNING_ENABLED = True  # False면 fast_front_rear_avm 기본 BEV 값 사용
BEV_VEHICLE_LENGTH_M = 5.0
BEV_VEHICLE_WIDTH_M = 2.5
BEV_BOTTOM_FORWARD_M = 0.75  # BEV bottom row is treated as 0.75 m ahead.
BEV_LANE_WIDTH_M = 7.0
BEV_SOURCE_CUTOFF_Y = 260
BEV_SRC_TOP_LEFT_X = 265
BEV_SRC_TOP_RIGHT_X = 375
BEV_SRC_TOP_Y = 260
BEV_SRC_BOTTOM_LEFT_X = 0
BEV_SRC_BOTTOM_RIGHT_X = 640
BEV_SRC_BOTTOM_Y = 480

# ============================================================
# Line Process
# ============================================================
IGNORE_CENTER_YELLOW = True # BEV 중앙 노란 점선은 차선 후보에서 제거
CENTER_YELLOW_IGNORE_WIDTH_PX = 90  # 중앙 기준 제거 폭(px)

# ============================================================
# Lane Objects for RRT
# ============================================================
LANE_OBJECT_TOPIC = "/vision_objs"
LANE_OBJECT_MAX_COUNT = 40
LANE_OBJECT_MAX_FORWARD_M = BEV_LENGTH_M + BEV_BOTTOM_FORWARD_M
LANE_OBJECT_LENGTH_M = 1.30
LANE_OBJECT_WIDTH_M = 0.12
LANE_OBJECT_MIN_PIXELS = 40
LANE_OBJECT_CLUSTER_GAP_PX = 60
LANE_OBJECT_WHITE_LABEL = 2
LANE_OBJECT_YELLOW_LABEL = 3

# Detect the internally black start area on the original BEV and black out its
# largest rectangular region before line/object processing.
START_BLACK_BOX_FILTER_ENABLED = True
START_BLACK_BOX_MAX_VALUE = 35
START_BLACK_BOX_CLOSE_KERNEL_PX = 31
START_BLACK_BOX_OPEN_KERNEL_PX = 5
START_BLACK_BOX_BORDER_MARGIN_RATIO = 0.03
START_BLACK_BOX_MIN_WIDTH_RATIO = 0.40
START_BLACK_BOX_MIN_HEIGHT_RATIO = 0.05
START_BLACK_BOX_MAX_HEIGHT_RATIO = 0.45
START_BLACK_BOX_MIN_AREA_RATIO = 0.02
START_BLACK_BOX_MIN_FILL_RATIO = 0.25

# Semantic Line Process outputs. These topic names match judgement_pkg.
SCHOOLZONE_DETECT_TOPIC = "/schoolzone"
STOPLINE_DETECT_TOPIC = "/stoplane"
STOPLINE_DISTANCE_TOPIC = "/vision/stopline_distance"

# Schoolzone_Detect is ON when this many yellow lane clusters are present.
SCHOOLZONE_MIN_YELLOW_CLUSTERS = 20

# ============================================================
# 정지선 (Stop Line)
# ============================================================
STOP_LINE_MIN_WIDTH_PX = 120    # 흰색 수평 성분 최소 폭
STOP_LINE_MIN_HEIGHT_PX = 8     # 두꺼운 정지선 최소 높이
STOP_LINE_ROW_RATIO = 0.45      # 한 row에서 흰색 픽셀이 이미지 폭의 이 비율 이상이면 후보
STOP_LINE_CLEAR_MARGIN_PX = 14  # 검출된 정지선 주변을 차선/클러스터링 마스크에서 제거할 여유
STOP_LINE_TRIGGER_DISTANCE_M = 15.0  # /stoplane Bool을 true로 바꾸는 정지선 거리 임계값

# ============================================================
# 시각화
# ============================================================
PUBLISH_DEBUG_IMAGES = True  # False면 BEV/final/lane_mask 디버그 이미지 퍼블리시 최소화
TARGET_FPS = 30.0            # lane_node 최대 처리 FPS

# ============================================================
# YOLO / Object Detection
# ============================================================
YOLO_ENABLED = True
YOLO_MODEL_PATH = package_file_path("models", "traffic_light_yolo11n_best.pt")
YOLO_FRAME_SKIP = 2
YOLO_TRAFFIC_LIGHT_CONF = 0.80
YOLO_TRAFFIC_LIGHT_IMGSZ = 640
YOLO_TRAFFIC_LIGHT_TOP_RATIO = 0.60
YOLO_TRAFFIC_LIGHT_FRAME_SKIP = 4
TRAFFIC_LIGHT_FONT_PATH = None  # 한글 표시용 폰트 경로. None이면 시스템 Noto/Nanum 폰트 자동 탐색.

# Traffic_Decide output. Values match judgement_pkg TrafficLight:
# 0=UNKNOWN, 1=GREEN, 2=LEFT, 3=RED. Yellow is treated as RED/stop.
TRAFFIC_DECIDE_TOPIC = "/trafficlight"
TRAFFIC_DECIDE_NONE = 0
TRAFFIC_DECIDE_GO = 1
TRAFFIC_DECIDE_LEFT = 2
TRAFFIC_DECIDE_STOP = 3
TRAFFIC_DECIDE_YELLOW = 3
