#!/usr/bin/env python3

import argparse
import pickle
import time
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.exposure import equalize_hist, rescale_intensity

import irpythermal
import utils


WINDOW_NAME = "Thermal Camera"
UPSCALE_FACTOR = 4
FPS_SMOOTHING_ALPHA = 0.8
FPS_INIT_FRAMES = 10
DEFAULT_COLORMAP = cv2.COLORMAP_INFERNO


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thermal Camera OpenCV Viewer")
    parser.add_argument(
        "-c",
        "--custom",
        action="store_true",
        help=(
            "Show temperatures at predefined custom coordinates instead of "
            "automatic min / max / center points"
        ),
    )
    return parser.parse_args()


@dataclass
class ViewerState:
    draw_temp: bool = True
    orientation: int = 0  # 0, 90, 180, 270
    upscale_factor: int = UPSCALE_FACTOR


class FpsCounter:
    def __init__(self, alpha: float = 0.9, init_frame_count: int = 10) -> None:
        self.alpha = alpha
        self.init_frame_count = init_frame_count
        self.frame_times: list[float] = []
        self.start_time = time.time()
        self.ema_duration: float | None = None

    def update(self) -> None:
        current_time = time.time()
        frame_duration = current_time - self.start_time

        if len(self.frame_times) < self.init_frame_count:
            self.frame_times.append(frame_duration)
            self.ema_duration = sum(self.frame_times) / len(self.frame_times)
        else:
            assert self.ema_duration is not None
            self.ema_duration = (
                self.alpha * self.ema_duration + (1.0 - self.alpha) * frame_duration
            )

        self.start_time = current_time

    def get_fps(self) -> float:
        if not self.ema_duration or self.ema_duration <= 0:
            return 0.0
        return 1.0 / self.ema_duration


def increase_luminance_contrast(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def rotate_coordinate(
    pos: tuple[int, int],
    shape: tuple[int, int],
    orientation: int,
) -> tuple[int, int]:
    x, y = pos
    width, height = shape

    if orientation == 0:
        return x, y
    if orientation == 90:
        return y, width - x
    if orientation == 180:
        return width - x, height - y
    if orientation == 270:
        return height - y, x

    raise ValueError(f"Unsupported orientation: {orientation}")


def rotate_frame(frame: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 0:
        return frame
    if orientation == 90:
        return np.rot90(frame, 1).copy()
    if orientation == 180:
        return np.rot90(frame, 2).copy()
    if orientation == 270:
        return np.rot90(frame, 3).copy()

    raise ValueError(f"Unsupported orientation: {orientation}")


def upscale_frame(frame: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(frame, np.ones((factor, factor, 1), dtype=np.uint8)).astype(np.uint8)


def thermal_to_colormap(frame_raw: np.ndarray) -> np.ndarray:
    frame_float = frame_raw.astype(np.float32)
    frame_8bit = rescale_intensity(
        equalize_hist(frame_float),
        in_range="image",
        out_range=(0, 255),
    ).astype(np.uint8)

    colored = cv2.applyColorMap(frame_8bit, DEFAULT_COLORMAP)
    return increase_luminance_contrast(colored)


def scale_point(point: tuple[int, int], factor: int) -> tuple[int, int]:
    x, y = point
    return x * factor, y * factor


def draw_temperature_marker(
    frame: np.ndarray,
    point: tuple[int, int],
    temp_c: float,
    color: tuple[int, int, int],
    camera_width: int,
    camera_height: int,
    upscale_factor: int,
    orientation: int,
) -> None:
    scaled_point = scale_point(point, upscale_factor)
    rotated_point = rotate_coordinate(
        scaled_point,
        (camera_width * upscale_factor, camera_height * upscale_factor),
        orientation,
    )
    utils.drawTemperature(frame, rotated_point, temp_c, color)


def draw_custom_temperatures(
    frame: np.ndarray,
    info: dict,
    camera_width: int,
    camera_height: int,
    upscale_factor: int,
    orientation: int,
) -> None:
    i = 1
    while True:
        point_key = f"coord_{i}_point"
        temp_key = f"coord_{i}_C"

        if point_key not in info or temp_key not in info:
            break

        draw_temperature_marker(
            frame=frame,
            point=info[point_key],
            temp_c=info[temp_key],
            color=(255, 255, 255),
            camera_width=camera_width,
            camera_height=camera_height,
            upscale_factor=upscale_factor,
            orientation=orientation,
        )
        i += 1


def draw_default_temperatures(
    frame: np.ndarray,
    info: dict,
    camera_width: int,
    camera_height: int,
    upscale_factor: int,
    orientation: int,
) -> None:
    draw_temperature_marker(
        frame=frame,
        point=info["Tmin_point"],
        temp_c=info["Tmin_C"],
        color=(255, 128, 128),
        camera_width=camera_width,
        camera_height=camera_height,
        upscale_factor=upscale_factor,
        orientation=orientation,
    )
    draw_temperature_marker(
        frame=frame,
        point=info["Tmax_point"],
        temp_c=info["Tmax_C"],
        color=(0, 128, 255),
        camera_width=camera_width,
        camera_height=camera_height,
        upscale_factor=upscale_factor,
        orientation=orientation,
    )
    draw_temperature_marker(
        frame=frame,
        point=info["Tcenter_point"],
        temp_c=info["Tcenter_C"],
        color=(255, 255, 255),
        camera_width=camera_width,
        camera_height=camera_height,
        upscale_factor=upscale_factor,
        orientation=orientation,
    )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:0.1f}",
        (2, 12),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (255, 255, 255),
        1,
        cv2.LINE_8,
    )


def save_raw_capture(camera: irpythermal.Camera) -> None:
    ret, frame = camera.cap.read()
    if not ret or frame is None:
        print("Failed to capture raw frame for saving")
        return

    filename = time.strftime("%Y-%m-%d_%H-%M-%S") + ".pkl"
    with open(filename, "wb") as file_obj:
        pickle.dump(frame, file_obj)

    print(f"Saved raw capture to {filename}")


def apply_temperature_range(camera: irpythermal.Camera, high: bool) -> None:
    if high:
        camera.temperature_range_high()
    else:
        camera.temperature_range_normal()

    for _ in range(50):
        camera.read()

    camera.calibrate()


def handle_keypress(
    key: int,
    camera: irpythermal.Camera,
    state: ViewerState,
    window_name: str,
    current_frame: np.ndarray,
) -> bool:
    if key == ord("q"):
        return False

    if key == ord("u"):
        camera.calibrate()
    elif key == ord("k"):
        apply_temperature_range(camera, high=False)
    elif key == ord("l"):
        apply_temperature_range(camera, high=True)
    elif key == ord("s"):
        filename = time.strftime("%Y-%m-%d_%H-%M-%S") + ".png"
        cv2.imwrite(filename, current_frame)
        print(f"Saved image to {filename}")
    elif key == ord("o"):
        state.orientation = (state.orientation - 90) % 360
        _, _, width, height = cv2.getWindowImageRect(window_name)
        cv2.resizeWindow(window_name, height, width)
    elif key == ord("a"):
        save_raw_capture(camera)
    elif key == ord("t"):
        state.draw_temp = not state.draw_temp
        print(f"Draw temperatures: {state.draw_temp}")

    return True


def main() -> None:
    args = parse_arguments()
    state = ViewerState()
    fps_counter = FpsCounter(
        alpha=FPS_SMOOTHING_ALPHA, init_frame_count=FPS_INIT_FRAMES
    )

    camera = irpythermal.Camera()

    use_custom_coords = bool(args.custom and utils.CUSTOM_COORDINATES)
    if use_custom_coords:
        camera.set_custom_coords(tuple(utils.CUSTOM_COORDINATES))

    window_name = f"{WINDOW_NAME} - {type(camera).__name__}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame_raw = camera.read()
            if not ret:
                print("Failed to read frame")
                continue

            fps_counter.update()
            info, _ = camera.info()

            frame = thermal_to_colormap(frame_raw)
            frame = rotate_frame(frame, state.orientation)
            frame = upscale_frame(frame, state.upscale_factor)

            if state.draw_temp:
                if use_custom_coords:
                    draw_custom_temperatures(
                        frame=frame,
                        info=info,
                        camera_width=camera.width,
                        camera_height=camera.height,
                        upscale_factor=state.upscale_factor,
                        orientation=state.orientation,
                    )
                else:
                    draw_default_temperatures(
                        frame=frame,
                        info=info,
                        camera_width=camera.width,
                        camera_height=camera.height,
                        upscale_factor=state.upscale_factor,
                        orientation=state.orientation,
                    )

                draw_fps(frame, fps_counter.get_fps())

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            should_continue = handle_keypress(
                key=key,
                camera=camera,
                state=state,
                window_name=window_name,
                current_frame=frame,
            )
            if not should_continue:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
