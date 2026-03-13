#!/usr/bin/env python3

import math
import time
from dataclasses import dataclass
from sys import platform
from time import sleep
from typing import Optional

import cv2
import numpy as np

SET_CORRECTION = 0 * 4
SET_REFLECTION = 1 * 4
SET_AMB = 2 * 4
SET_HUMIDITY = 3 * 4
SET_EMISSIVITY = 4 * 4
SET_DISTANCE = 5 * 4

ROWS_SPECIAL_DATA = 4
MAX_RAW_VALUE = 16384


def read_u16(arr_u16: np.ndarray, offset: int) -> np.uint16:
    return arr_u16[offset]


def read_f32(arr_u16: np.ndarray, offset: int, step: int = 2) -> np.float32:
    return arr_u16[offset : offset + step].view(np.float32)[0]


def read_u8(arr_u16: np.ndarray, offset: int, step: int) -> np.ndarray:
    return arr_u16[offset : offset + step].view(np.uint8)


@dataclass
class ResolutionParams:
    fpa_off: int
    fpa_div: float
    amount_pixels: int
    cal_00_offset: float
    cal_00_fpamul: float


class Camera:
    """Read and convert data from XTherm / HT301 / InfiRay thermal cameras."""

    supported_resolutions = {(240, 180), (256, 192), (384, 288), (640, 512)}
    ZEROC = 273.15

    distance_multiplier = 1.0
    offset_temp_shutter = 0.0
    offset_temp_fpa = 0.0

    range = 120
    correction_coefficient_m = 1.0
    correction_coefficient_b = 0.0

    def __init__(
        self,
        video_dev: Optional[cv2.VideoCapture] = None,
        camera_raw: bool = False,
        fixed_offset: float = 0.0,
    ) -> None:
        self.cap = video_dev or self.find_device()
        if self.cap is None:
            raise RuntimeError("No video device found")

        self.camera_raw = camera_raw
        self.user_offset = fixed_offset

        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - ROWS_SPECIAL_DATA
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

        self.frame_width = self.width
        self.frame_height = self.height
        self.four_line_para = self.width * self.height

        self.reference_frame: Optional[np.ndarray] = None
        self.offset_mean = 0.0
        self.dead_pixels_mask: Optional[np.ndarray] = None
        self.custom_coords: Optional[tuple[tuple[int, int], ...]] = None

        params = self._get_resolution_params(self.width)
        self.fpa_off = params.fpa_off
        self.fpa_div = params.fpa_div
        self.amountPixels = params.amount_pixels
        self.cal_00_offset = params.cal_00_offset
        self.cal_00_fpamul = params.cal_00_fpamul
        self.userArea = self.amountPixels + 127

        self.frame_raw_u16 = np.array([], dtype=np.uint16)

        self._configure_capture()
        self.wait_for_range_application()
        self.calibrate()

    def _configure_capture(self) -> None:
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8004)

    @classmethod
    def find_device(cls) -> cv2.VideoCapture:
        """Find the first supported thermal camera."""
        last_seen = None

        for idx in range(10):
            try:
                if platform.startswith("linux"):
                    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                else:
                    cap = cv2.VideoCapture(idx)

                cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                last_seen = (cap_width, cap_height)

                print(f"Found camera {idx}: {cap_width}x{cap_height}")

                if (
                    cap_width,
                    cap_height - ROWS_SPECIAL_DATA,
                ) in cls.supported_resolutions:
                    return cap

                cap.release()
            except Exception as exc:
                print(f"[warn] Failed opening camera index {idx}: {exc}")

        raise ValueError(
            f"Cannot find supported thermal camera. Last seen resolution: {last_seen}"
        )

    @staticmethod
    def _get_resolution_params(width: int) -> ResolutionParams:
        if width == 640:
            return ResolutionParams(6867, 33.8, width * 3, 390.0, 7.05)
        if width == 384:
            return ResolutionParams(7800, 36.0, width * 3, 390.0, 7.05)
        if width == 256:
            return ResolutionParams(8617, 37.682, width, 170.0, 0.0)
        if width == 240:
            return ResolutionParams(7800, 36.0, width, 390.0, 7.05)
        raise ValueError(f"Unsupported camera width: {width}")

    def get_resolution(self) -> tuple[int, int]:
        return self.width, self.height

    @staticmethod
    def bin_to_twos_complement(binary: str) -> int:
        if binary[0] == "1":
            return int(binary, 2) - 2 ** len(binary)
        return int(binary, 2)

    def _decode_shutter_core_temps(self) -> tuple[float, float]:
        shut_temper = read_u16(
            self.frame_raw_u16, self.four_line_para + self.amountPixels + 1
        )
        core_temper = read_u16(
            self.frame_raw_u16, self.four_line_para + self.amountPixels + 2
        )

        shutter_fix = 0.0

        if self.camera_raw:
            if shut_temper < 2049:
                float_shutter = float(shut_temper)
                corr_factor = 0.625
            else:
                float_shutter = float(0xFFF - shut_temper)
                corr_factor = -0.625

            shutter_fix = (
                self.bin_to_twos_complement(
                    bin(
                        read_u16(
                            self.frame_raw_u16,
                            self.four_line_para + self.amountPixels * 2 + 47,
                        )
                    )[2:].zfill(16)[:8]
                )
                / 10.0
            )
            float_shutter = (float_shutter * corr_factor + 2731.5) / 10.0 - 273.15
            float_shutter += shutter_fix
            float_core = float_shutter - shutter_fix
        else:
            float_shutter = shut_temper / 10.0 - self.ZEROC
            float_core = core_temper / 10.0 - self.ZEROC

        return float_shutter, float_core

    def _sample_coords(
        self,
        raw_image: np.ndarray,
        temperature_table: np.ndarray,
    ) -> dict[str, object]:
        if self.custom_coords is not None:
            coords = self.custom_coords
        else:
            coords = (
                (self.width // 4, self.height // 4),
                (self.width // 2, self.height // 4),
                (3 * self.width // 4, self.height // 4),
                (self.width // 3, 2 * self.height // 3),
                (2 * self.width // 3, 2 * self.height // 3),
            )

        result: dict[str, object] = {}
        for idx, (x, y) in enumerate(coords, start=1):
            if 0 <= x < self.width and 0 <= y < self.height:
                raw_val = raw_image[y, x]
                result[f"coord_{idx}_point"] = (x, y)
                result[f"coord_{idx}_C"] = float(temperature_table[raw_val])
        return result

    def info(self) -> tuple[dict, np.ndarray]:
        float_shut_temper, float_core_temper = self._decode_shutter_core_temps()

        cal_00 = float(
            read_u16(self.frame_raw_u16, self.four_line_para + self.amountPixels)
        )
        self.cal_01 = read_f32(
            self.frame_raw_u16, self.four_line_para + self.amountPixels + 3
        )
        cal_02 = read_f32(
            self.frame_raw_u16, self.four_line_para + self.amountPixels + 5
        )
        cal_03 = read_f32(
            self.frame_raw_u16, self.four_line_para + self.amountPixels + 7
        )
        cal_04 = read_f32(
            self.frame_raw_u16, self.four_line_para + self.amountPixels + 9
        )
        cal_05 = read_f32(
            self.frame_raw_u16, self.four_line_para + self.amountPixels + 11
        )

        camera_soft_version = (
            read_u8(
                self.frame_raw_u16, self.four_line_para + self.amountPixels + 24, step=8
            )
            .tobytes()
            .decode("ascii", errors="ignore")
            .rstrip("\x00")
        )

        sn = (
            read_u8(
                self.frame_raw_u16, self.four_line_para + self.amountPixels + 32, step=3
            )
            .tobytes()
            .decode("ascii", errors="ignore")
            .rstrip("\x00")
        )

        correction = read_f32(self.frame_raw_u16, self.four_line_para + self.userArea)
        refl_tmp = read_f32(self.frame_raw_u16, self.four_line_para + self.userArea + 2)
        air_tmp = read_f32(self.frame_raw_u16, self.four_line_para + self.userArea + 4)
        humidity = read_f32(self.frame_raw_u16, self.four_line_para + self.userArea + 6)
        emiss = read_f32(self.frame_raw_u16, self.four_line_para + self.userArea + 8)
        distance = read_u16(
            self.frame_raw_u16, self.four_line_para + self.userArea + 10
        )

        fpa_avg = read_u16(self.frame_raw_u16, self.four_line_para)
        fpa_tmp_raw = read_u16(self.frame_raw_u16, self.four_line_para + 1)
        max_x = read_u16(self.frame_raw_u16, self.four_line_para + 2)
        max_y = read_u16(self.frame_raw_u16, self.four_line_para + 3)
        self.max_raw = read_u16(self.frame_raw_u16, self.four_line_para + 4)
        min_x = read_u16(self.frame_raw_u16, self.four_line_para + 5)
        min_y = read_u16(self.frame_raw_u16, self.four_line_para + 6)
        self.min_raw = read_u16(self.frame_raw_u16, self.four_line_para + 7)
        self.avg_raw = read_u16(self.frame_raw_u16, self.four_line_para + 8)

        center_raw = read_u16(self.frame_raw_u16, self.four_line_para + 12)
        user_raw00 = read_u16(self.frame_raw_u16, self.four_line_para + 13)
        user_raw01 = read_u16(self.frame_raw_u16, self.four_line_para + 14)
        user_raw02 = read_u16(self.frame_raw_u16, self.four_line_para + 15)

        fpa_tmp_c = 20.0 - (float(fpa_tmp_raw) - self.fpa_off) / self.fpa_div

        distance_adjusted = min(float(distance), 20.0) * self.distance_multiplier
        atm = self.atmt(humidity, air_tmp, distance_adjusted)
        self.numerator_sub = (1.0 - emiss) * atm * math.pow(
            refl_tmp + self.ZEROC, 4
        ) + (1.0 - atm) * math.pow(air_tmp + self.ZEROC, 4)
        self.denominator = emiss * atm

        ts = float_shut_temper + self.offset_temp_shutter
        tfpa = fpa_tmp_c + self.offset_temp_fpa

        self.cal_a = cal_02 / (self.cal_01 * 2.0)
        self.cal_b = cal_02 * cal_02 / (self.cal_01 * self.cal_01 * 4.0)
        self.cal_c = self.cal_01 * math.pow(ts, 2) + ts * cal_02
        self.cal_d = cal_03 * math.pow(tfpa, 2) + cal_04 * tfpa + cal_05

        cal_00_corr = 0
        if self.range == 120:
            cal_00_corr = int(self.cal_00_offset - tfpa * self.cal_00_fpamul)

        table_offset = cal_00 - max(cal_00_corr, 0)

        temperature_table = self.get_temp_table(
            correction=correction,
            air_tmp=air_tmp,
            table_offset=table_offset,
            distance_adjusted=distance_adjusted,
        )
        temperature_table = temperature_table + self.user_offset

        raw_image = (
            self.frame_raw_u16[: self.four_line_para]
            .copy()
            .reshape(self.height, self.width)
        )
        coord_info = self._sample_coords(raw_image, temperature_table)

        info = {
            "temp_shutter": float_shut_temper,
            "temp_core": float_core_temper,
            "cameraSoftVersion": camera_soft_version,
            "sn": sn,
            "correction": correction,
            "temp_reflected": refl_tmp,
            "temp_air": air_tmp,
            "humidity": humidity,
            "emissivity": emiss,
            "distance": int(distance),
            "fpa_average": int(fpa_avg),
            "temp_fpa": fpa_tmp_c,
            "temp_max_x": int(max_x),
            "temp_max_y": int(max_y),
            "temp_max_raw": int(self.max_raw),
            "temp_max": float(temperature_table[self.max_raw]),
            "temp_min_x": int(min_x),
            "temp_min_y": int(min_y),
            "temp_min_raw": int(self.min_raw),
            "temp_min": float(temperature_table[self.min_raw]),
            "temp_average_raw": int(self.avg_raw),
            "temp_average": float(temperature_table[self.avg_raw]),
            "temp_center_raw": int(center_raw),
            "temp_center": float(temperature_table[center_raw]),
            "temp_user_00": float(temperature_table[user_raw00]),
            "temp_user_01": float(temperature_table[user_raw01]),
            "temp_user_02": float(temperature_table[user_raw02]),
            "Tmin_point": (int(min_x), int(min_y)),
            "Tmax_point": (int(max_x), int(max_y)),
            "Tcenter_point": (self.width // 2, self.height // 2),
            "Tmin_C": float(temperature_table[self.min_raw]),
            "Tmax_C": float(temperature_table[self.max_raw]),
            "Tcenter_C": float(temperature_table[center_raw]),
        }
        info.update(coord_info)

        return info, temperature_table

    def set_custom_coords(self, coords: tuple[tuple[int, int], ...]) -> None:
        self.custom_coords = tuple(coords) if coords else None

    def read(
        self,
        raw: bool = False,
        max_retries: int = 5,
        retry_delay: float = 0.5,
    ) -> tuple[bool, np.ndarray]:
        """Read one complete visible frame from the camera."""
        expected_size = self.height * self.width

        for attempt in range(max_retries):
            ret, frame_raw = self.cap.read()
            if not ret or frame_raw is None:
                time.sleep(retry_delay)
                continue

            frame_data = frame_raw.view(np.uint16).ravel()
            if frame_data.size < expected_size:
                print(
                    f"[warn] Incomplete frame ({frame_data.size} < {expected_size}), "
                    f"retry {attempt + 1}/{max_retries}"
                )
                time.sleep(retry_delay)
                continue

            try:
                visible = (
                    frame_data[: self.four_line_para]
                    .copy()
                    .reshape(self.height, self.width)
                )
            except ValueError:
                print(
                    f"[warn] Reshape failed (size={frame_data.size}), "
                    f"retry {attempt + 1}/{max_retries}"
                )
                time.sleep(retry_delay)
                continue

            self.frame_raw_u16 = frame_data
            break
        else:
            raise RuntimeError("Failed to read a complete frame from camera")

        if raw:
            return True, visible

        if self.reference_frame is not None:
            frame_float = visible.astype(np.float32)
            corrected = frame_float - self.reference_frame + self.offset_mean
            corrected = np.clip(corrected, 0, 65535)

            if self.dead_pixels_mask is not None:
                corrected = cv2.inpaint(
                    corrected,
                    self.dead_pixels_mask,
                    3,
                    cv2.INPAINT_TELEA,
                )

            visible = corrected.astype(np.uint16)

        return True, visible

    def get_frame(self) -> np.ndarray:
        _, frame = self.read()
        _, lut = self.info()
        return lut[frame]

    def convert_to_frame(self, frame_raw: np.ndarray, lut: np.ndarray) -> np.ndarray:
        return lut[frame_raw]

    def get_temperature_at(self, x: int, y: int) -> float:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Coordinate out of bounds: {(x, y)}")

        _, raw_frame = self.read(raw=True)
        _, temperature_table = self.info()
        raw_val = raw_frame[y, x]
        return float(temperature_table[raw_val])

    def get_temperatures_at(
        self,
        coords: tuple[tuple[int, int], ...],
    ) -> dict[tuple[int, int], float]:
        _, raw_frame = self.read(raw=True)
        _, temperature_table = self.info()

        out: dict[tuple[int, int], float] = {}
        for x, y in coords:
            if 0 <= x < self.width and 0 <= y < self.height:
                out[(x, y)] = float(temperature_table[raw_frame[y, x]])
        return out

    def set_correction(self, correction: float) -> None:
        self.send_float_command(SET_CORRECTION, correction)

    def set_reflection(self, reflection: float) -> None:
        self.send_float_command(SET_REFLECTION, reflection)

    def set_amb(self, amb: float) -> None:
        self.send_float_command(SET_AMB, amb)

    def set_humidity(self, humidity: float) -> None:
        self.send_float_command(SET_HUMIDITY, humidity)

    def set_emissivity(self, emiss: float) -> None:
        self.send_float_command(SET_EMISSIVITY, emiss)

    def set_distance(self, distance: int) -> None:
        self.send_ushort_command(SET_DISTANCE, distance)

    def send_float_command(self, position: int, value: float) -> None:
        bytes_ = np.array([value], dtype=np.float32).view(np.uint8)
        for i, byte in enumerate(bytes_):
            payload = ((position + i) << 8) | (0xFF & int(byte))
            if not self.cap.set(cv2.CAP_PROP_ZOOM, payload):
                print(f"Control failed: {payload}")

    def send_ushort_command(self, position: int, value: int) -> None:
        bytes_ = np.array([value], dtype=np.uint16).view(np.uint8)
        for i, byte in enumerate(bytes_):
            payload = ((position + i) << 8) | (0xFF & int(byte))
            if not self.cap.set(cv2.CAP_PROP_ZOOM, payload):
                print(f"Control failed: {payload}")

    def send_byte_command(self, position: int, value: int) -> None:
        byte = int(np.array([value], dtype=np.uint8)[0])
        payload = (position << 8) | (0xFF & byte)
        if not self.cap.set(cv2.CAP_PROP_ZOOM, payload):
            print(f"Control failed: {payload}")

    def save_parameters(self) -> None:
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x80FF)

    def set_point(self, x: int, y: int, index: int) -> None:
        if index == 0:
            x_cmd, y_cmd = 0xF000 + x, 0xF200 + y
        elif index == 1:
            x_cmd, y_cmd = 0xF400 + x, 0xF600 + y
        elif index == 2:
            x_cmd, y_cmd = 0xF800 + x, 0xFA00 + y
        else:
            raise ValueError("Index must be 0, 1, or 2")

        self.cap.set(cv2.CAP_PROP_ZOOM, x_cmd)
        self.cap.set(cv2.CAP_PROP_ZOOM, y_cmd)

    def calibrate_raw(self, quiet: bool = False) -> None:
        self.reference_frame = None
        self.offset_mean = 0.0
        self.dead_pixels_mask = None

        sleep(0.5)
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8000)
        sleep(0.3)
        self.flush_buffer()
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8000)

        ret, frame_visible = self.read(raw=True)
        if not ret:
            raise RuntimeError("Failed to capture reference frame")

        self.reference_frame = frame_visible.astype(np.float32)
        self.offset_mean = float(np.mean(self.reference_frame))

        frame_float = frame_visible.astype(np.float32)
        min_val = float(np.min(frame_float))
        max_val = float(np.max(frame_float))
        threshold = min_val + (max_val - min_val) * 0.05

        if np.count_nonzero(frame_float < threshold) != 0:
            self.dead_pixels_mask = cv2.inRange(
                frame_float, 0, float(threshold)
            ).astype(np.uint8)

        if not quiet:
            dead_count = (
                0
                if self.dead_pixels_mask is None
                else int(np.count_nonzero(self.dead_pixels_mask))
            )
            print(
                f"Calibration raw stats: min={min_val}, max={max_val}, avg={np.mean(frame_float)}"
            )
            print(f"Found {dead_count} dead pixels")

    def calibrate(self, quiet: bool = False) -> None:
        if self.camera_raw:
            self.calibrate_raw(quiet=quiet)
        else:
            self.cap.set(cv2.CAP_PROP_ZOOM, 0x8000)

    def release(self) -> None:
        self.cap.release()

    def wvc(self, humidity: float, t_atm: float) -> float:
        h1 = 1.5587
        h2 = 0.06939
        h3 = -2.7816e-4
        h4 = 6.8455e-7
        return humidity * math.exp(
            h1 + h2 * t_atm + h3 * math.pow(t_atm, 2) + h4 * math.pow(t_atm, 3)
        )

    def atmt(self, humidity: float, t_atm: float, distance: float) -> float:
        k_atm = 1.9
        nsqd = -math.sqrt(distance)
        sqw = math.sqrt(self.wvc(humidity, t_atm))

        a1 = 0.006569
        a2 = 0.01262
        b1 = -0.002276
        b2 = -0.00667

        return k_atm * math.exp(nsqd * (a1 + b1 * sqw)) + (1.0 - k_atm) * math.exp(
            nsqd * (a2 + b2 * sqw)
        )

    def get_temp_table(
        self,
        correction: float,
        air_tmp: float,
        table_offset: float,
        distance_adjusted: float,
    ) -> np.ndarray:
        n = np.sqrt(
            np.abs(
                (
                    (np.arange(MAX_RAW_VALUE, dtype=np.float32) - table_offset)
                    * self.cal_d
                    + self.cal_c
                )
                / self.cal_01
                + self.cal_b
            )
        )
        n[np.isnan(n)] = 0.0

        wtot = np.power(n - self.cal_a + self.ZEROC, 4)
        ttot = (
            np.power((wtot - self.numerator_sub) / self.denominator, 0.25) - self.ZEROC
        )
        temperature_table = (
            ttot
            + (distance_adjusted * 0.85 - 1.125) * (ttot - air_tmp) / 100.0
            + correction
        )

        return (
            self.correction_coefficient_m * temperature_table
            + self.correction_coefficient_b
        )

    def temperature_range_normal(self) -> None:
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8020)
        self.correction_coefficient_m = 1.0
        self.correction_coefficient_b = 0.0

    def temperature_range_high(self) -> None:
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8021)
        if self.camera_raw:
            self.correction_coefficient_m = 0.1
            self.correction_coefficient_b = 0.0
        else:
            self.correction_coefficient_m = 1.17
            self.correction_coefficient_b = -40.9

    def wait_for_range_application(self, timeout: float = 20.0) -> bool:
        print("Waiting for camera to stabilize...")
        start = time.time()
        done = False

        while time.time() - start < timeout:
            ret, frame_visible = self.read()
            if ret and np.std(frame_visible) > 0:
                done = True
                break
            time.sleep(0.1)

        if self.camera_raw:
            lowest = 1000.0
            margin = 0.1
            min_val = 0.01

            while time.time() - start < timeout:
                self.cap.set(cv2.CAP_PROP_ZOOM, 0x8000)
                self.calibrate(quiet=True)
                ret, frame_visible = self.read()
                if ret:
                    std = float(np.std(frame_visible))
                    self.cap.set(cv2.CAP_PROP_ZOOM, 0x8000)
                    sleep(0.1)

                    if std > min_val and lowest - std < margin:
                        print(f"Camera is stable with std={std}")
                        return True

                    if min_val < std < lowest:
                        lowest = std
        elif done:
            print("Camera is stable")
            return True

        return False

    def flush_buffer(self, num_reads: int = 16) -> None:
        for _ in range(num_reads):
            self.read(raw=True)


class MockVideoCapture:
    def set(self, prop_id: int, value) -> None:
        setattr(self, str(prop_id), value)

    def get(self, prop_id: int):
        return getattr(self, str(prop_id))

    def release(self) -> None:
        pass


class CameraEmulator(Camera):
    def __init__(self, filename: str) -> None:
        frame_raw_u16 = np.load(filename, allow_pickle=True)
        self.cap = MockVideoCapture()
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_raw_u16.shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_raw_u16.shape[1])
        self.frame_raw_u16 = frame_raw_u16.ravel()
        super().__init__(video_dev=self.cap)

    def read(
        self,
        raw: bool = False,
        max_retries: int = 5,
        retry_delay: float = 0.5,
    ) -> tuple[bool, np.ndarray]:
        frame_visible = (
            self.frame_raw_u16[: self.four_line_para]
            .copy()
            .reshape(self.height, self.width)
        )
        return True, frame_visible