import cv2
import numpy as np

from . import config as cfg


PX_PER_M = 25
VIEW_WIDTH_M = 10.0
FRONT_LENGTH_M = 16.0
VEHICLE_LENGTH_M = 5.0
VEHICLE_WIDTH_M = 2.5
LANE_WIDTH_M = 7.0

CAM_WIDTH = 640
CAM_HEIGHT = 480
SOURCE_CUTOFF_Y = 260


class FrontBevProjector:
    """Front-only BEV projector based on fast_front_rear_avm.py."""

    def __init__(self):
        if getattr(cfg, 'BEV_TUNING_ENABLED', True):
            self.px_per_m = max(1, int(round(1.0 / float(cfg.M_PER_PIXEL))))
            view_width_m = float(cfg.BEV_WIDTH_M)
            total_length_m = float(cfg.BEV_LENGTH_M)
            vehicle_length_m = float(cfg.BEV_VEHICLE_LENGTH_M)
            vehicle_width_m = float(cfg.BEV_VEHICLE_WIDTH_M)
            lane_width_m = float(cfg.BEV_LANE_WIDTH_M)
            source_cutoff_y = int(cfg.BEV_SOURCE_CUTOFF_Y)
            src_pts = np.float32([
                [float(cfg.BEV_SRC_TOP_LEFT_X), float(cfg.BEV_SRC_TOP_Y)],
                [float(cfg.BEV_SRC_TOP_RIGHT_X), float(cfg.BEV_SRC_TOP_Y)],
                [float(cfg.BEV_SRC_BOTTOM_LEFT_X), float(cfg.BEV_SRC_BOTTOM_Y)],
                [float(cfg.BEV_SRC_BOTTOM_RIGHT_X), float(cfg.BEV_SRC_BOTTOM_Y)],
            ])
        else:
            self.px_per_m = PX_PER_M
            view_width_m = VIEW_WIDTH_M
            total_length_m = FRONT_LENGTH_M + VEHICLE_LENGTH_M
            vehicle_length_m = VEHICLE_LENGTH_M
            vehicle_width_m = VEHICLE_WIDTH_M
            lane_width_m = LANE_WIDTH_M
            source_cutoff_y = SOURCE_CUTOFF_Y
            src_pts = np.float32([[265, 260], [375, 260], [0, 480], [640, 480]])

        self.source_cutoff_y = source_cutoff_y
        self.bev_w = int(view_width_m * self.px_per_m)
        self.bev_h = int(total_length_m * self.px_per_m)
        self.cx = self.bev_w / 2.0

        self.car_rear_y = float(self.bev_h)
        self.car_front_y = self.car_rear_y - vehicle_length_m * self.px_per_m
        bottom_forward_m = float(getattr(cfg, 'BEV_BOTTOM_FORWARD_M', 0.75))
        self.reference_origin_y = self.car_rear_y + bottom_forward_m * self.px_per_m
        self.front_far_y = 0.0

        lane_w_px = lane_width_m * self.px_per_m
        dst_pts = np.float32([
            [self.cx - lane_w_px / 2.0, self.front_far_y],
            [self.cx + lane_w_px / 2.0, self.front_far_y],
            [self.cx - lane_w_px / 2.0, self.car_rear_y],
            [self.cx + lane_w_px / 2.0, self.car_rear_y],
        ])

        self.homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.inv_homography = np.linalg.inv(self.homography)
        self.map_x, self.map_y = self._build_warp_lut(self.homography)

        half_w = int(vehicle_width_m * self.px_per_m / 2.0)
        self.car_pt1 = (int(self.cx - half_w), int(self.car_front_y))
        self.car_pt2 = (int(self.cx + half_w), int(self.car_rear_y - 1))

    @property
    def output_shape(self):
        return self.bev_h, self.bev_w

    @property
    def meters_per_pixel(self):
        return 1.0 / float(self.px_per_m)

    def warp(self, image):
        source = image.copy()
        source[:self.source_cutoff_y, :] = 0
        warped = cv2.remap(
            source,
            self.map_x,
            self.map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped

    def draw_vehicle(self, image):
        cv2.rectangle(image, self.car_pt1, self.car_pt2, (50, 50, 50), -1)
        cv2.rectangle(image, self.car_pt1, self.car_pt2, (0, 0, 255), 2)
        cv2.putText(
            image,
            "CAR",
            (self.car_pt1[0] + 5, int((self.car_pt1[1] + self.car_pt2[1]) / 2.0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )
        cv2.arrowedLine(
            image,
            (int(self.cx), self.car_pt1[1] + 35),
            (int(self.cx), self.car_pt1[1] + 10),
            (0, 255, 0),
            2,
            tipLength=0.4,
        )
        return image

    def _build_warp_lut(self, homography):
        xs, ys = np.meshgrid(
            np.arange(self.bev_w, dtype=np.float32),
            np.arange(self.bev_h, dtype=np.float32),
        )

        homography_inv = np.linalg.inv(homography)
        ones = np.ones_like(xs)
        coords = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3).T

        src = homography_inv @ coords
        src /= src[2:3, :]

        map_x = src[0].reshape(self.bev_h, self.bev_w).astype(np.float32)
        map_y = src[1].reshape(self.bev_h, self.bev_w).astype(np.float32)
        return map_x, map_y
