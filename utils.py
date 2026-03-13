#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Custom coordinates (x, y) in thermal-frame pixel space.
CUSTOM_COORDINATES: list[tuple[int, int]] = [
    # high endurance SD cards
    (34, 57),
    (36, 78),
    (38, 96),
    (40, 111),
    (43, 125),
    # industrial SD cards
    (142, 51),
    (139, 72),
    (136, 90),
    (134, 106),
    (132, 120),
    # ultra SD cards
    (234, 60),
    (226, 79),
    (219, 95),
    (213, 110),
    (207, 123),
]


def draw_temperature(
    img: np.ndarray,
    point: tuple[int, int],
    temperature_c: float,
    color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw a crosshair and temperature label at a point."""
    cross_inner = 2
    cross_outer = 5
    thickness = 1
    font = cv2.FONT_HERSHEY_PLAIN

    x, y = point
    label = f"{temperature_c:.2f}C"

    cv2.line(img, (x + cross_inner, y), (x + cross_outer, y), color, thickness)
    cv2.line(img, (x - cross_inner, y), (x - cross_outer, y), color, thickness)
    cv2.line(img, (x, y + cross_inner), (x, y + cross_outer), color, thickness)
    cv2.line(img, (x, y - cross_inner), (x, y - cross_outer), color, thickness)

    text_size = cv2.getTextSize(label, font, 1, thickness)[0]
    text_x = x + cross_inner
    text_y = y + cross_inner + text_size[1]

    if text_x + text_size[0] > img.shape[1]:
        text_x = x - cross_inner - text_size[0]
    if text_y > img.shape[0]:
        text_y = y - cross_inner

    cv2.putText(
        img,
        label,
        (text_x, text_y),
        font,
        1,
        color,
        thickness,
        cv2.LINE_8,
    )


# Backward-compatible alias
drawTemperature = draw_temperature


def auto_exposure(update: bool, exposure: dict[str, Any], frame: np.ndarray) -> bool:
    """Adjust exposure limits in-place based on current frame values."""
    frame_min = float(frame.min())
    frame_max = float(frame.max())

    temp_min = float(exposure["T_min"])
    temp_max = float(exposure["T_max"])
    temp_margin = float(exposure["T_margin"])
    auto_type = exposure["auto_type"]

    if auto_type == "center":
        temp_center = (temp_min + temp_max) / 2.0
        delta = max(temp_center - frame_min, frame_max - temp_center, 0.0) + temp_margin

        if (
            frame_min < temp_min
            or temp_max < frame_max
            or (
                temp_min + 2 * temp_margin < frame_min
                and temp_max - 2 * temp_margin > frame_max
            )
        ):
            update = True
            temp_min = temp_center - delta
            temp_max = temp_center + delta

    elif auto_type == "ends":
        if temp_min > frame_min or temp_min + 2 * temp_margin < frame_min:
            update = True
            temp_min = frame_min - temp_margin

        if temp_max < frame_max or temp_max - 2 * temp_margin > frame_max:
            update = True
            temp_max = frame_max + temp_margin

    exposure["T_min"] = temp_min
    exposure["T_max"] = temp_max
    return update


# Backward-compatible alias
autoExposure = auto_exposure


def correct_roi(
    roi: tuple[tuple[int, int], tuple[int, int]],
    shape: tuple[int, ...],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Normalize ROI into top-left and bottom-right coordinates."""
    (x, y), (w, h) = roi
    height, width = shape[:2]

    x1 = max(0, min(x, x + w))
    y1 = max(0, min(y, y + h))
    x2 = min(width, max(x, x + w))
    y2 = min(height, max(y, y + h))

    return (x1, y1), (x2, y2)


# Backward-compatible alias
correctRoi = correct_roi


def in_roi(
    roi: tuple[tuple[int, int], tuple[int, int]],
    point: tuple[int, int],
    shape: tuple[int, ...],
) -> bool:
    """Return True if point lies inside ROI."""
    (x1, y1), (x2, y2) = correct_roi(roi, shape)
    px, py = point
    return x1 < px < x2 and y1 < py < y2


# Backward-compatible alias
inRoi = in_roi


class Annotations:
    """Manage matplotlib annotations and ROI overlay for thermal frames."""

    def __init__(self, ax: Any, patches: Any) -> None:
        self.ax = ax
        self.anns: dict[Any, Any] = {}
        self.roi = ((0, 0), (0, 0))

        self.astyle = dict(
            text="",
            xy=(0, 0),
            xytext=(0, 0),
            textcoords="offset pixels",
            arrowprops=dict(facecolor="black", arrowstyle="->"),
        )

        self.roi_patch = ax.add_patch(
            patches.Rectangle(
                (0, 0),
                0,
                0,
                linewidth=1,
                edgecolor="black",
                facecolor="none",
            )
        )
        self.set_roi(self.roi)

    def set_roi(self, roi: tuple[tuple[int, int], tuple[int, int]]) -> None:
        self.roi = roi
        (x, y), (w, h) = roi
        self.roi_patch.xy = (x, y)
        self.roi_patch.set_width(w)
        self.roi_patch.set_height(h)
        self.roi_patch.set_visible(w != 0 and h != 0)

    def get_ann(self, name: Any, color: str) -> Any:
        if name not in self.anns:
            self.anns[name] = self.ax.annotate(
                **self.astyle,
                bbox=dict(boxstyle="square", fc=color, alpha=0.3, lw=0),
            )
        return self.anns[name]

    def get_pos(self, name: Any) -> tuple[int, int]:
        pos = self.get_ann(name, "").xy
        return int(pos[0]), int(pos[1])

    def get_val(self, name: Any, annotation_frame: np.ndarray) -> float:
        x, y = self.get_pos(name)
        x = max(0, min(x, annotation_frame.shape[1] - 1))
        y = max(0, min(y, annotation_frame.shape[0] - 1))
        return float(annotation_frame[y, x])

    def update(
        self,
        temp_annotations: dict[str, dict[Any, str]],
        annotation_frame: np.ndarray,
        draw_temp: bool,
    ) -> None:
        for name, color in temp_annotations["std"].items():
            pos = self._get_pos(name, annotation_frame, self.roi)
            self._set_annotation(
                self.get_ann(name, color), pos, annotation_frame, draw_temp
            )

        for name, color in temp_annotations["user"].items():
            pos = self._get_pos(name, annotation_frame, self.roi)
            self._set_annotation(
                self.get_ann(name, color), pos, annotation_frame, draw_temp
            )

    def get(self) -> list[Any]:
        return list(self.anns.values()) + [self.roi_patch]

    def remove(self, annotations_dict: dict[Any, Any]) -> None:
        for name in list(annotations_dict.keys()):
            if name in self.anns:
                self.anns[name].remove()
                del self.anns[name]
        annotations_dict.clear()

    def _set_annotation(
        self,
        ann: Any,
        pos: tuple[int, int],
        annotation_frame: np.ndarray,
        draw_temp: bool,
    ) -> None:
        x = int(round(pos[0]))
        y = int(round(pos[1]))

        x = max(0, min(x, annotation_frame.shape[1] - 1))
        y = max(0, min(y, annotation_frame.shape[0] - 1))

        ann.xy = (x, y)
        value = float(annotation_frame[y, x])
        ann.set_text(f"{value:.2f}$^\\circ$C")
        ann.set_visible(draw_temp)

        text_offset_x = 20
        text_offset_y = 15

        if x > annotation_frame.shape[1] - 50:
            text_offset_x = -80
        if y < 30:
            text_offset_y = -15

        ann.xyann = (text_offset_x, text_offset_y)

    def _get_pos(
        self,
        name: Any,
        annotation_frame: np.ndarray,
        roi: tuple[tuple[int, int], tuple[int, int]],
    ) -> tuple[int, int]:
        (x1, y1), (x2, y2) = correct_roi(roi, annotation_frame.shape)
        roi_frame = annotation_frame[y1:y2, x1:x2]

        if roi_frame.size == 0:
            x1, y1 = 0, 0
            roi_frame = annotation_frame

        if name == "Tmin":
            pos = np.unravel_index(np.argmin(roi_frame), roi_frame.shape)
            return int(pos[1] + x1), int(pos[0] + y1)

        if name == "Tmax":
            pos = np.unravel_index(np.argmax(roi_frame), roi_frame.shape)
            return int(pos[1] + x1), int(pos[0] + y1)

        if name == "Tcenter":
            return annotation_frame.shape[1] // 2, annotation_frame.shape[0] // 2

        # user annotation names are expected to be (x, y) tuples
        return int(name[0]), int(name[1])
