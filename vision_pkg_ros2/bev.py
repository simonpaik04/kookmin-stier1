from . import config as cfg
from .front_bev_projector import FrontBevProjector


class BevProcessor:
    """BEV: front camera image -> bird's-eye-view image."""

    def __init__(self):
        self.projector = FrontBevProjector()
        self._sync_config_shape()

    def rebuild(self):
        self.projector = FrontBevProjector()
        self._sync_config_shape()
        return self.projector

    def warp(self, frame):
        return self.projector.warp(frame)

    def draw_vehicle(self, image):
        return self.projector.draw_vehicle(image)

    def _sync_config_shape(self):
        cfg.M_PER_PIXEL = self.projector.meters_per_pixel
        cfg.BEV_IMAGE_HEIGHT, cfg.BEV_IMAGE_WIDTH = self.projector.output_shape
