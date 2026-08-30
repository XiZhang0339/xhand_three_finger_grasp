"""Off-screen MuJoCo rendering and verifiable MP4 encoding."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco


@dataclass(frozen=True)
class VideoSettings:
    width: int = 640
    height: int = 480
    fps: int = 30
    camera: str = "three_finger_camera"


def expected_video_frame_count(
    total_steps: int, timestep_s: float, fps: int
) -> int:
    """Count positive-time frame thresholds reached by a fixed-step run.

    ``VideoRecorder`` starts at ``1 / fps`` and does not render a time-zero
    frame, so a run ending exactly between two frame times must use floor, not
    round.  The small tolerance only absorbs binary representation error at an
    exact frame boundary.
    """

    if total_steps < 0 or timestep_s <= 0.0 or fps <= 0:
        raise ValueError("video frame schedule inputs must be positive")
    return int(math.floor(total_steps * timestep_s * fps + 1e-12))


def open_video_encoder(
    path: Path, width: int = 640, height: int = 480, fps: int = 30
) -> subprocess.Popen[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def probe_video(
    path: Path,
    expected_frames: int,
    settings: VideoSettings | None = None,
) -> dict[str, Any]:
    settings = settings or VideoSettings()
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    if decode.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}: {decode.stderr.strip()}")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames:format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    if len(payload.get("streams", [])) != 1:
        raise RuntimeError("ffprobe did not find exactly one video stream")
    stream = payload["streams"][0]
    frame_count = int(stream.get("nb_read_frames", -1))
    expected_rate = f"{settings.fps}/1"
    if (
        stream.get("codec_name") != "h264"
        or int(stream.get("width", -1)) != settings.width
        or int(stream.get("height", -1)) != settings.height
        or stream.get("avg_frame_rate") != expected_rate
        or frame_count != expected_frames
    ):
        raise RuntimeError(f"unexpected encoded video properties: {payload}")
    return {
        "decode_verified": True,
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream["avg_frame_rate"],
        "frame_count": frame_count,
        "duration_s": float(payload["format"]["duration"]),
        "size_bytes": int(payload["format"]["size"]),
    }


@dataclass
class VideoRecorder:
    """A one-way rendering sink; it never owns or advances simulation state."""

    model: mujoco.MjModel
    output: Path
    settings: VideoSettings = field(default_factory=VideoSettings)

    def __post_init__(self) -> None:
        self.renderer = mujoco.Renderer(
            self.model, height=self.settings.height, width=self.settings.width
        )
        self.encoder = open_video_encoder(
            self.output,
            self.settings.width,
            self.settings.height,
            self.settings.fps,
        )
        self.options = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.options)
        self.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        self.options.sitegroup[4] = 0
        self.next_frame_time = 1.0 / self.settings.fps
        self.frame_steps: list[int] = []
        self._closed = False

    def maybe_record(self, data: mujoco.MjData, step: int) -> None:
        if data.time + 1e-12 < self.next_frame_time:
            return
        self.renderer.update_scene(data, self.settings.camera, self.options)
        frame = self.renderer.render()
        if self.encoder.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self.encoder.stdin.write(frame.tobytes())
        self.frame_steps.append(step)
        self.next_frame_time += 1.0 / self.settings.fps

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.renderer.close()
        if self.encoder.stdin is not None:
            self.encoder.stdin.close()
        stderr = (
            self.encoder.stderr.read().decode("utf-8", errors="replace")
            if self.encoder.stderr
            else ""
        )
        return_code = self.encoder.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with {return_code}: {stderr.strip()}")

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
