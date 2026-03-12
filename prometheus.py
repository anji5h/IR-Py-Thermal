from time import time
import numpy as np
from prometheus_client import Gauge, start_http_server


class PrometheusExporter:
    """Helper for exporting temperature frames to Prometheus as custom metrics.

    Coordinates are grouped into three families of five points each:
      - temp_hend_{1..5}
      - temp_ind_{1..5}
      - temp_ultra_{1..5}
    """

    def __init__(self, port: int = 8000, interval_sec: float = 15.0) -> None:
        self.enabled = False
        self.interval_sec = float(interval_sec)
        self.last_push = time()
        self.metrics = None

        start_http_server(port=port)

        self.metrics = {
            "hend": Gauge(
                "temp_pi_hend",
                "Temperature of high endurance sd cards.",
                ["slot"],
            ),
            "ind": Gauge(
                "temp_pi_ind",
                "Temperature of industrial sd cards.",
                ["slot"],
            ),
            "ultra": Gauge(
                "temp_pi_ultra",
                "Temperature of ultra sd cards.",
                ["slot"],
            ),
        }

        self.enabled = True

    def export(self, frame: np.ndarray, coords: list[tuple[int, int]]) -> None:
        """Export temperatures from `frame` at given coords to Prometheus.

        - `frame` must be a 2D numpy array of temperatures (e.g. °C).
        - `coords` is a list of (x, y) pixel coordinates.
        """
        if not self.enabled or self.metrics is None:
            return

        now = time()
        if (now - self.last_push) < self.interval_sec:
            return
        h, w = frame.shape

        for idx, (x, y) in enumerate(coords):
            xi = max(0, min(int(round(x)), w - 1))
            yi = max(0, min(int(round(y)), h - 1))
            temp_val = float(frame[yi, xi])

            group_idx = idx // 5
            slot = (idx % 5) + 1

            if group_idx == 0:
                self.metrics["hend"].labels(slot=str(slot)).set(temp_val)
            elif group_idx == 1:
                self.metrics["ind"].labels(slot=str(slot)).set(temp_val)
            else:
                self.metrics["ultra"].labels(slot=str(slot)).set(temp_val)

        self.last_push = now
