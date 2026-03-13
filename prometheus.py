import numpy as np
from prometheus_client import Gauge, start_http_server


class PrometheusExporter:
    """Helper for exporting temperature frames to Prometheus as custom metrics.

    Coordinates are grouped into three families of five points each:
      - temp_hend_{1..5}
      - temp_ind_{1..5}
      - temp_ultra_{1..5}
    """

    def __init__(self, port: int = 8000) -> None:
        self.enabled = False
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

    def export(self, values: list[float]) -> None:
        """Export temperatures from `frame` at given coords to Prometheus.

        - `frame` must be a 2D numpy array of temperatures (e.g. °C).
        - `coords` is a list of (x, y) pixel coordinates.
        """
        if not self.enabled or self.metrics is None:
            return

        for idx, temp_val in enumerate(values[:15]):
            group_idx = idx // 5
            slot = str((idx % 5) + 1)

            if group_idx == 0:
                self.metrics["hend"].labels(slot=slot).set(temp_val)
            elif group_idx == 1:
                self.metrics["ind"].labels(slot=slot).set(temp_val)
            elif group_idx == 2:
                self.metrics["ultra"].labels(slot=slot).set(temp_val)
