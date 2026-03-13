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

    def export(self, frame: np.ndarray, cordinates: list[tuple[int, int]]) -> None:
        if not self.enabled or self.metrics is None:
            return

        for idx, (x, y) in enumerate(cordinates):
            x_cord = max(0, min(x, frame.shape[1] - 1))
            y_cord = max(0, min(y, frame.shape[0] - 1))
            temp_val = float(frame[y_cord, x_cord])
            group_idx = idx // 5
            slot = str((idx % 5) + 1)

            if group_idx == 0:
                self.metrics["hend"].labels(slot=slot).set(temp_val)
            elif group_idx == 1:
                self.metrics["ind"].labels(slot=slot).set(temp_val)
            elif group_idx == 2:
                self.metrics["ultra"].labels(slot=slot).set(temp_val)
