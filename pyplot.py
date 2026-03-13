#!/usr/bin/env python3

import argparse
import csv
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import serial

import irpythermal
from prometheus import PrometheusExporter
import utils


CMAP_NAMES = [
    "inferno",
    "plasma",
    "coolwarm",
    "cividis",
    "jet",
    "nipy_spectral",
    "binary",
    "gray",
    "tab10",
]

FILE_NAME_FORMAT = "%Y-%m-%d_%H-%M-%S"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thermal Camera Viewer")

    parser.add_argument(
        "-r", "--rawcam", action="store_true", help="Use raw camera mode"
    )
    parser.add_argument(
        "-d", "--device", type=str, help="Video device path, e.g. /dev/video0"
    )
    parser.add_argument("-o", "--offset", type=float, help="Fixed temperature offset")

    parser.add_argument(
        "-l",
        "--lockin",
        type=float,
        help="Enable lock-in thermometry with the given frequency in Hz",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=str,
        help="Serial port for power/load control, e.g. /dev/ttyUSB0",
    )
    parser.add_argument(
        "-i",
        "--integration",
        type=float,
        help="Integration time for lock-in thermometry in seconds",
    )
    parser.add_argument(
        "-n",
        "--negate",
        action="store_true",
        help="Invert the serial load signal",
    )
    parser.add_argument(
        "-c",
        "--custom",
        action="store_true",
        help="Use predefined custom coordinates and export them to Prometheus",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI and only export metrics / process frames",
    )
    parser.add_argument(
        "--prometheus-port",
        type=int,
        default=8000,
        help="Port for Prometheus metrics endpoint",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Frame polling interval in headless mode, seconds",
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=str,
        help="Use emulator input from file.npy",
    )

    return parser.parse_args()


@dataclass
class ExposureState:
    auto: bool = True
    auto_type: str = "ends"
    t_min: float = 0.0
    t_max: float = 50.0
    t_margin: float = 2.0


@dataclass
class DiffState:
    enabled: bool = False
    annotation_enabled: bool = False
    frame: Optional[np.ndarray] = None


@dataclass
class AppState:
    args: argparse.Namespace
    camera: irpythermal.Camera
    fps: int = 40
    draw_temp: bool = True
    lockin: bool = False
    headless: bool = False
    negate: bool = False
    frequency: Optional[float] = None
    port: Optional[str] = None
    integration: Optional[float] = None

    frame: Optional[np.ndarray] = None
    quad_frame: Optional[np.ndarray] = None
    in_phase_frame: Optional[np.ndarray] = None

    exposure: ExposureState = field(default_factory=ExposureState)
    diff: DiffState = field(default_factory=DiffState)

    cmaps_idx: int = 1
    start_skips: int = 2
    paused: bool = False
    update_colormap: bool = True
    is_capturing: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    lock_in_thread: Optional[threading.Thread] = None

    prom_exporter: Optional[PrometheusExporter] = None
    temp_annotations: dict = field(default_factory=dict)
    csv_filename: Optional[str] = None

    mouse_action_pos: tuple[int, int] = (0, 0)
    mouse_action: Optional[str] = None
    roi: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0))

    fig: Optional[object] = None
    ax: Optional[object] = None
    im: Optional[object] = None
    im_in_phase: Optional[object] = None
    im_quadrature: Optional[object] = None
    status_text_obj: Optional[object] = None
    status_text: str = ""
    annotations: Optional[object] = None


def create_camera(args: argparse.Namespace) -> irpythermal.Camera:
    if args.file and args.file.endswith(".npy"):
        return irpythermal.CameraEmulator(args.file)

    camera_kwargs = {}

    if args.rawcam:
        camera_kwargs["camera_raw"] = True

    if args.device:
        camera_kwargs["video_dev"] = cv2.VideoCapture(args.device)

    if args.offset is not None:
        camera_kwargs["fixed_offset"] = args.offset

    return irpythermal.Camera(**camera_kwargs)


def create_app_state(args: argparse.Namespace) -> AppState:
    camera = create_camera(args)

    state = AppState(
        args=args,
        camera=camera,
        headless=args.headless,
        lockin=args.lockin is not None,
        draw_temp=not bool(args.lockin),
    )

    if state.lockin:
        if not args.port or not args.integration:
            print("Error: --lockin requires both --port and --integration", flush=True)
            sys.exit(1)

        state.frequency = args.lockin
        state.port = args.port
        state.integration = args.integration
        state.negate = args.negate

    state.frame = np.full((camera.height, camera.width), 25.0)
    state.quad_frame = np.zeros((camera.height, camera.width))
    state.in_phase_frame = np.zeros((camera.height, camera.width))
    state.diff.frame = np.zeros((camera.height, camera.width))

    if args.custom:
        state.temp_annotations = {"std": {}, "user": {}}
        for coord in utils.CUSTOM_COORDINATES:
            state.temp_annotations["user"][coord] = "white"

        state.prom_exporter = PrometheusExporter(port=args.prometheus_port)
    else:
        state.temp_annotations = {
            "std": {"Tmin": "lightblue", "Tmax": "red", "Tcenter": "yellow"},
            "user": {},
        }

    return state


def setup_gui(state: AppState) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if state.lockin:
        fig, axes = plt.subplots(nrows=2, ncols=2, layout="tight")
        state.fig = fig
        state.ax = axes[0][0]

        state.im = axes[0][0].imshow(state.frame, cmap=CMAP_NAMES[state.cmaps_idx])
        state.im_in_phase = axes[0][1].imshow(
            state.frame, cmap=CMAP_NAMES[state.cmaps_idx]
        )
        state.im_quadrature = axes[1][1].imshow(
            state.frame, cmap=CMAP_NAMES[state.cmaps_idx]
        )

        axes[0][0].set_title("Live")
        axes[0][1].set_title("In-phase")
        axes[1][1].set_title("Quadrature")
        axes[1][0].axis("off")

        divider = make_axes_locatable(axes[0][0])
        divider_in_phase = make_axes_locatable(axes[0][1])
        divider_quadrature = make_axes_locatable(axes[1][1])

        plt.colorbar(state.im, cax=divider.append_axes("right", size="5%", pad=0.05))
        plt.colorbar(
            state.im_in_phase,
            cax=divider_in_phase.append_axes("right", size="5%", pad=0.05),
        )
        plt.colorbar(
            state.im_quadrature,
            cax=divider_quadrature.append_axes("right", size="5%", pad=0.05),
        )

        state.status_text = (
            "Frame: -\n"
            "Time: -/-\n"
            "Load: -\n"
            "Frequency: - Hz\n"
            "Integration Time: - s\n"
            "Serial Port: -"
        )
        state.status_text_obj = axes[1][0].text(
            0.05,
            0.95,
            state.status_text,
            verticalalignment="top",
            horizontalalignment="left",
            transform=axes[1][0].transAxes,
            fontsize=12,
            color="black",
        )
    else:
        state.fig = plt.figure()
        state.ax = plt.gca()
        state.im = state.ax.imshow(state.frame, cmap=CMAP_NAMES[state.cmaps_idx])

        divider = make_axes_locatable(state.ax)
        plt.colorbar(state.im, cax=divider.append_axes("right", size="5%", pad=0.05))

    try:
        state.fig.canvas.manager.set_window_title("Thermal Camera")
    except Exception:
        pass

    state.annotations = utils.Annotations(state.ax, patches)


def cleanup(state: AppState) -> None:
    stop_capture(state.lock_in_thread)
    state.camera.release()


def log_annotations_to_csv(state: AppState, annotation_frame: np.ndarray) -> None:
    if state.csv_filename is None or state.annotations is None:
        return

    row = [datetime.now()]
    for ann_type in ("std", "user"):
        for ann_name in state.temp_annotations[ann_type]:
            pos = state.annotations.get_pos(ann_name)
            val = round(state.annotations.get_val(ann_name, annotation_frame), 2)
            row.extend([pos[0], pos[1], val])

    with open(state.csv_filename, "a", newline="") as f:
        csv.writer(f).writerow(row)


def get_lockin_frame(
    state: AppState,
    freq: float,
    port: str,
    integration: float,
    invert: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        ser = serial.Serial(port, 115200)
    except serial.SerialException as exc:
        print(f"Error: could not open serial port {port} ({exc})", flush=True)
        sys.exit(1)

    start_time = time.time()
    in_phase_sum = np.zeros((state.camera.height, state.camera.width))
    quadrature_sum = np.zeros((state.camera.height, state.camera.width))
    total_frames = 0

    period = 1.0 / freq
    half_period = period / 2.0
    load_on = True
    last_toggle_time = start_time

    while (time.time() - start_time) < integration:
        current_time = time.time() - start_time
        ret, raw_frame = state.camera.read()
        _, lut = state.camera.info()

        if not ret:
            print("Error: could not read frame from camera", flush=True)
            sys.exit(1)

        state.frame = state.camera.convert_to_frame(raw_frame, lut)
        total_frames += 1

        state.status_text = (
            f"Frame: {total_frames}\n"
            f"Time: {current_time:.2f}/{integration:.2f}\n"
            f"Load: {load_on}\n"
            f"Frequency: {freq:.2f} Hz\n"
            f"Integration Time: {integration:.2f} s\n"
            f"Serial Port: {port}"
        )

        if current_time - (last_toggle_time - start_time) >= half_period:
            if load_on:
                ser.write(b"1\n" if invert else b"0\n")
                load_on = False
            else:
                ser.write(b"0\n" if invert else b"1\n")
                load_on = True

            last_toggle_time += half_period

        phase = 2 * math.pi * freq * current_time
        sin_weight = 2 * math.sin(phase)
        cos_weight = -2 * math.cos(phase)

        in_phase_sum += raw_frame * sin_weight
        quadrature_sum += raw_frame * cos_weight

        if not state.is_capturing:
            break

    ser.write(b"0\n")
    ser.close()

    if total_frames > 0:
        in_phase_sum /= total_frames
        quadrature_sum /= total_frames

    return in_phase_sum, quadrature_sum


def capture_lock_in(state: AppState) -> None:
    while state.is_capturing:
        in_phase, quad = get_lockin_frame(
            state,
            state.frequency,
            state.port,
            state.integration,
            state.negate,
        )
        with state.lock:
            state.in_phase_frame = in_phase
            state.quad_frame = quad


def start_capture(state: AppState) -> threading.Thread:
    state.is_capturing = True
    thread = threading.Thread(target=capture_lock_in, args=(state,), daemon=True)
    thread.start()
    return thread


def stop_capture(thread: Optional[threading.Thread]) -> None:
    if thread is not None:
        thread.join(timeout=2)


def run_headless(state: AppState) -> None:
    print("Starting headless mode", flush=True)

    try:
        while True:
            frame = state.camera.get_frame()

            if state.prom_exporter is not None:
                state.prom_exporter.export(frame, utils.CUSTOM_COORDINATES)

            time.sleep(state.args.interval)
    finally:
        cleanup(state)


def build_animation_callback(state: AppState):
    def animate_func(_frame: int):
        if state.lockin and state.start_skips > 0:
            state.frame = state.camera.get_frame()
            state.start_skips -= 1
        elif state.lockin:
            if not state.is_capturing:
                state.lock_in_thread = start_capture(state)
        else:
            state.frame = state.camera.get_frame()

        if state.paused:
            return [state.im] + state.annotations.get()

        show_frame = (
            state.frame - state.diff.frame if state.diff.enabled else state.frame
        )
        annotation_frame = (
            state.frame - state.diff.frame
            if state.diff.annotation_enabled
            else state.frame
        )

        state.im.set_array(show_frame)

        if state.lockin:
            state.im_in_phase.set_array(state.in_phase_frame)
            state.im_quadrature.set_array(state.quad_frame)

        state.annotations.update(
            state.temp_annotations,
            annotation_frame,
            state.draw_temp,
        )

        if state.exposure.auto:
            state.update_colormap = utils.autoExposure(
                state.update_colormap,
                {
                    "auto": state.exposure.auto,
                    "auto_type": state.exposure.auto_type,
                    "T_min": state.exposure.t_min,
                    "T_max": state.exposure.t_max,
                    "T_margin": state.exposure.t_margin,
                },
                show_frame,
            )

        log_annotations_to_csv(state, annotation_frame)

        if state.prom_exporter is not None:
            state.prom_exporter.export(annotation_frame, utils.CUSTOM_COORDINATES)

        if state.update_colormap:
            state.im.set_clim(state.exposure.t_min, state.exposure.t_max)
            state.fig.canvas.draw_idle()
            state.update_colormap = False
            return []

        if state.lockin:
            state.im_in_phase.set_clim(
                np.min(state.in_phase_frame),
                np.max(state.in_phase_frame),
            )
            state.im_quadrature.set_clim(
                np.min(state.quad_frame),
                np.max(state.quad_frame),
            )
            state.status_text_obj.set_text(state.status_text)
            return [
                state.im,
                state.im_in_phase,
                state.im_quadrature,
                state.status_text_obj,
            ] + state.annotations.get()

        return [state.im] + state.annotations.get()

    return animate_func


def run_gui(state: AppState) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.backend_bases import MouseButton

    setup_gui(state)

    def print_help() -> None:
        print(
            """keys:
    h        help
    space    pause / resume
    d        set diff
    x        toggle diff
    c        toggle annotation diff
    t        toggle temperatures
    e        remove user annotations
    u        calibrate
    w        save image
    r        save raw frame
    v        toggle CSV recording
    , .      change color map
    a        toggle auto exposure
    z        change auto exposure mode
    k l      set temperature range
    arrows   adjust exposure
"""
        )

    def press(event) -> None:
        if event.key == "h":
            print_help()
        elif event.key == " ":
            state.paused = not state.paused
            print("paused:", state.paused)
        elif event.key == "d":
            state.diff.frame = state.frame.copy()
            state.diff.enabled = True
            state.diff.annotation_enabled = True
        elif event.key == "x":
            state.diff.enabled = not state.diff.enabled
        elif event.key == "c":
            state.diff.annotation_enabled = not state.diff.annotation_enabled
        elif event.key == "t":
            state.draw_temp = not state.draw_temp
        elif event.key == "e" and state.annotations is not None:
            state.annotations.remove(state.temp_annotations["user"])
        elif event.key == "u":
            state.camera.calibrate()
        elif event.key == "a":
            state.exposure.auto = not state.exposure.auto
        elif event.key == "z":
            state.exposure.auto_type = (
                "center" if state.exposure.auto_type == "ends" else "ends"
            )
        elif event.key == "w":
            filename = time.strftime(FILE_NAME_FORMAT) + ".png"
            plt.savefig(filename)
            print("saved:", filename)
        elif event.key == "r":
            filename = time.strftime(FILE_NAME_FORMAT) + ".npy"
            np.save(
                filename,
                state.camera.frame_raw_u16.reshape(
                    state.camera.height + 4,
                    state.camera.width,
                ),
            )
            print("saved:", filename)

    def onclick(event) -> None:
        if event.inaxes != state.ax:
            return

        pos = (int(event.xdata), int(event.ydata))

        if event.button == MouseButton.RIGHT:
            state.temp_annotations["user"][pos] = "white"
        elif event.button == MouseButton.LEFT:
            if utils.inRoi(state.annotations.roi, pos, state.frame.shape):
                state.mouse_action = "move_roi"
                state.mouse_action_pos = (
                    state.annotations.roi[0][0] - pos[0],
                    state.annotations.roi[0][1] - pos[1],
                )
            else:
                state.mouse_action = "create_roi"
                state.mouse_action_pos = pos
                state.annotations.set_roi((pos, (0, 0)))

    def onmotion(event) -> None:
        if event.inaxes != state.ax or event.button != MouseButton.LEFT:
            return

        pos = (int(event.xdata), int(event.ydata))

        if state.mouse_action == "create_roi":
            w = pos[0] - state.mouse_action_pos[0]
            h = pos[1] - state.mouse_action_pos[1]
            state.roi = (state.mouse_action_pos, (w, h))
            state.annotations.set_roi(state.roi)
        elif state.mouse_action == "move_roi":
            state.roi = (
                (
                    pos[0] + state.mouse_action_pos[0],
                    pos[1] + state.mouse_action_pos[1],
                ),
                state.annotations.roi[1],
            )
            state.annotations.set_roi(state.roi)

    anim = animation.FuncAnimation(
        state.fig,
        build_animation_callback(state),
        interval=1000 / state.fps,
        blit=True,
        cache_frame_data=False,
    )

    state.fig.canvas.mpl_connect("button_press_event", onclick)
    state.fig.canvas.mpl_connect("motion_notify_event", onmotion)
    state.fig.canvas.mpl_connect("key_press_event", press)

    print_help()

    try:
        plt.show()
    finally:
        _ = anim
        cleanup(state)


def main() -> None:
    state = create_app_state(parse_arguments())

    if state.headless:
        run_headless(state)
        return

    run_gui(state)


if __name__ == "__main__":
    main()
