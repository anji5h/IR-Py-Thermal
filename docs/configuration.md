# Configuration

IRThermal reads a TOML file. Start from the example and validate it before
deploying:

```bash
cp config.example.toml config.toml
irthermal check-config -c config.toml
```

## Where settings come from

Highest priority first:

1. Explicit command-line arguments, such as `--config`.
2. Environment variables, named `IRTHERMAL__<SECTION>__<KEY>`, for example
   `IRTHERMAL__SERVER__PORT=9000`.
3. The TOML file. The default path is `/etc/irthermal/config.toml`.

Unknown keys are rejected, so a typo fails loudly instead of being ignored.

## File layout

| Section              | Purpose                                                        | Required     |
| -------------------- | -------------------------------------------------------------- | ------------ |
| `[server]`           | HTTP server and logging                                        | No           |
| `[[cameras]]`        | One block per camera                                           | At least one |
| `[[cameras.points]]` | Fixed measurement points for the `[[cameras]]` block above it  | No           |

## `[server]`

| Key          | Default     | Allowed values                                                |
| ------------ | ----------- | ------------------------------------------------------------- |
| `host`       | `"0.0.0.0"` | Interface to bind. `"127.0.0.1"` listens locally only.        |
| `port`       | `8000`      | 1 to 65535                                                    |
| `log_level`  | `"info"`    | `"debug"`, `"info"`, `"warning"`, `"error"`                   |
| `log_format` | `"json"`    | `"json"` for log collectors, `"console"` for humans           |

## `[[cameras]]`

### Identity

| Key  | Default  | Description                                                                                                       |
| ---- | -------- | ----------------------------------------------------------------------------------------------------------------- |
| `id` | required | Unique name. Letters, digits, `-` and `_`, at most 64 characters. Used in API URLs and as a Prometheus label.     |

### Source

Set exactly one of these three.

| Key           | Default | Description                                                                                                                        |
| ------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `device`      | none    | The camera's device name for its transport. Linux: V4L2 path, e.g. `"/dev/video0"`. Windows: DirectShow name, e.g. `"T2SPro"`. macOS: avfoundation index, e.g. `"0"`. |
| `auto_detect` | `false` | `true` scans for a supported camera.                                                                                               |
| `replay_file` | none    | Path to a captured `.npy` file to replay instead of opening hardware.                                                              |

### Connection

| Key         | Default  | Description              |
| ----------- | -------- | ------------------------ |
| `transport` | `"auto"` | How to reach the camera. |

Transport values:

- `"auto"`: V4L2 on Linux, falling back to ffmpeg. ffmpeg on Windows and macOS.
- `"v4l2"`: Linux only. This is the only transport with the control channel, so
  calibration, range switching and parameter writes need it.
- `"ffmpeg"`: Any OS. Streaming and measurement only; control endpoints return
  501.

### Image and measurement

| Key                  | Default                          | Description                                                                                         |
| -------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------- |
| `raw_mode`           | `false`                          | `true` means the sensor sends unprocessed data and the correction is done in software.             |
| `calibration_file`   | `.temp/calibration/<id>.npz`     | Where the flat-field reference is stored between restarts. Only used with `raw_mode`.               |
| `temperature_offset` | `0.0`                            | Constant trim added to every reading, in °C.                                                        |
| `emissivity`         | the camera's value               | `0 < value <= 1`. Pushed to the camera on connect. Omit to keep the camera's value.                 |
| `colormap`           | `"inferno"`                      | `inferno`, `plasma`, `magma`, `viridis`, `cividis`, `turbo`, `jet`, `hot`, `bone`, `rainbow`, `gray` |

The calibration file is saved when you calibrate, reloaded on connect, and
ignored if it does not match the sensor. Without a control channel the shutter
can't close, so calibrate by hand: cover the lens, then `POST /calibrate`.

### Lock-in thermography (optional)

| Key               | Default  | Description                                                                         |
| ----------------- | -------- | ----------------------------------------------------------------------------------- |
| `lockin_port`     | none     | Serial port that switches the load, e.g. `"/dev/ttyUSB0"`. Writes `1\n` for on and `0\n` for off. |
| `lockin_baudrate` | `115200` | At least 300.                                                                       |
| `lockin_invert`   | `false`  | `true` for a load wired active-low.                                                 |

The serial port is set in the config rather than through the API so API
callers can't make the server open arbitrary devices. Omit `lockin_port` when
something else drives the load; the job then reports its modulator as
`"external"`.

## `[[cameras.points]]`

Each point is measured every frame and exported as the Prometheus gauge
`thermal_point_celsius`. A point belongs to the `[[cameras]]` block above it.

| Key      | Default  | Description                                                                                       |
| -------- | -------- | ------------------------------------------------------------------------------------------------- |
| `name`   | required | Unique within the camera. Same rules as the camera `id`.                                          |
| `x`, `y` | required | Pixel coordinates, at least 0.                                                                    |
| `labels` | `{}`     | Extra Prometheus labels, e.g. `{ group = "a", slot = "1" }`. Group them however you plan to query them. |

## Examples

See [`config.example.toml`](../config.example.toml) for a camera watching a
rack of SD cards with labelled points, an auto-detected bench camera, a
lock-in thermography setup and a replay of a recorded capture.
