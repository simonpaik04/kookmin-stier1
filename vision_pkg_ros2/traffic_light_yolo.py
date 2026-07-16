from . import config as cfg


class YoloNet:
    """YOLO_net: lazy-load Ultralytics YOLO and run prediction."""

    def __init__(self, logger=None):
        self.logger = logger
        self.model = None
        self.names = {}

    def load(self):
        if self.model is not None:
            return self.model
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(getattr(cfg, "YOLO_MODEL_PATH", "")))
            self.names = getattr(self.model, "names", {}) or {}
            self._info(f"[vision] YOLO loaded: {getattr(cfg, 'YOLO_MODEL_PATH', '')}")
            return self.model
        except Exception as exc:
            self._info(f"[vision] YOLO load failed: {exc}")
            self.model = None
            self.names = {}
            return None

    def predict(self, image, imgsz, conf, classes=None):
        model = self.load()
        if model is None or image is None:
            return []
        try:
            results = model.predict(
                image,
                imgsz=int(imgsz),
                conf=float(conf),
                classes=classes,
                verbose=False,
            )
        except Exception as exc:
            self._info(f"[vision] YOLO inference failed: {exc}")
            return []
        return results

    def _info(self, message):
        if self.logger is not None:
            self.logger.info(message)
