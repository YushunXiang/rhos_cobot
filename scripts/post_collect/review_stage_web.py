#!/usr/bin/env python3
# coding=utf-8

"""Local web app for reviewing and correcting HDF5 stage annotations."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from textwrap import dedent
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import h5py
import numpy as np


DEFAULT_FPS = 25
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_FIXED_DIR_NAME = "fixed_stage"
JPEG_QUALITY = 85
EPISODE_RE = re.compile(r"episode_(\d+)\.hdf5$")


class ReviewError(RuntimeError):
    """Domain error surfaced to the frontend as a 4xx response."""


@dataclass
class EpisodeSummary:
    name: str
    n_frames: int | None
    fps: int
    cameras: list[str]
    editable: bool
    has_fixed_copy: bool
    source_has_stage: bool
    working_stage_source: str
    issue_flags: list[str]
    unique_stages: list[int]
    stage_changes: list[dict[str, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a local web UI for reviewing and correcting stage labels.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Directory containing episode_*.hdf5 files.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Host to bind the web server to. Default: {DEFAULT_HOST}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind the web server to. Default: {DEFAULT_PORT}",
    )
    parser.add_argument(
        "--fixed_dir_name",
        type=str,
        default=DEFAULT_FIXED_DIR_NAME,
        help=f"Subdirectory used for corrected copies. Default: {DEFAULT_FIXED_DIR_NAME}",
    )
    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Allow the UI to overwrite original HDF5 files in place.",
    )
    return parser.parse_args()


def episode_sort_key(path: Path) -> tuple[int, str]:
    match = EPISODE_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def decode_frame_to_bgr(frame: Any) -> np.ndarray:
    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        image = frame
    else:
        if isinstance(frame, np.ndarray):
            encoded = frame.tobytes()
        else:
            encoded = bytes(frame)
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ReviewError("Failed to decode image frame from dataset.")

    # Match the replay path: these recorded frames need a channel swap before
    # being re-encoded for the browser, otherwise red/blue appear inverted.
    return image[:, :, [2, 1, 0]]


def encode_frame_to_jpeg(frame: Any) -> bytes:
    image = decode_frame_to_bgr(frame)
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not success:
        raise ReviewError("Failed to encode image frame as JPEG.")
    return encoded.tobytes()


def first_existing_frame_count(root: h5py.File) -> int | None:
    candidates = [
        "/observations/qpos",
        "/action",
        "/base_action",
        "/action_eef",
    ]
    for name in candidates:
        if name in root:
            dataset = root[name]
            if dataset.shape:
                return int(dataset.shape[0])

    image_group = root.get("/observations/images")
    if isinstance(image_group, h5py.Group):
        for camera_name in image_group.keys():
            dataset = image_group[camera_name]
            if dataset.shape:
                return int(dataset.shape[0])
    return None


def build_segments(stage_values: list[int]) -> list[dict[str, int]]:
    if not stage_values:
        return []

    segments: list[dict[str, int]] = []
    start = 0
    current = int(stage_values[0])
    for idx in range(1, len(stage_values)):
        value = int(stage_values[idx])
        if value != current:
            segments.append({"start": start, "end": idx, "stage": current})
            start = idx
            current = value
    segments.append({"start": start, "end": len(stage_values), "stage": current})
    return segments


def stage_changes(stage_values: list[int]) -> list[dict[str, int]]:
    return [{"idx": segment["start"], "stage": segment["stage"]} for segment in build_segments(stage_values)]


def analyze_stage_flags(stage_values: list[int], n_frames: int | None) -> list[str]:
    flags: list[str] = []
    if n_frames is None:
        flags.append("missing_frame_count")
        return flags

    if not stage_values:
        flags.append("missing_stage")
        return flags

    if len(stage_values) != n_frames:
        flags.append("stage_length_mismatch")
        return flags

    if any(value < 0 for value in stage_values):
        flags.append("negative_stage")

    if all(value == 0 for value in stage_values):
        flags.append("all_zero_stage")

    if any(stage_values[idx] < stage_values[idx - 1] for idx in range(1, len(stage_values))):
        flags.append("stage_regression")

    compressed = [segment["stage"] for segment in build_segments(stage_values)]
    if any(compressed[idx] - compressed[idx - 1] > 1 for idx in range(1, len(compressed))):
        flags.append("stage_jump")

    return flags


def inspect_source_episode(path: Path) -> dict[str, Any]:
    structure_flags: list[str] = []
    with h5py.File(path, "r") as root:
        fps = int(root.attrs.get("fps", DEFAULT_FPS))
        image_group = root.get("/observations/images")
        cameras = list(image_group.keys()) if isinstance(image_group, h5py.Group) else []
        n_frames = first_existing_frame_count(root)
        source_stage = root["/stage"][()].astype(np.int64).tolist() if "/stage" in root else []

        if image_group is None:
            structure_flags.append("missing_images")
        elif not cameras:
            structure_flags.append("empty_images_group")

        if "/observations/qpos" not in root:
            structure_flags.append("missing_qpos")

    return {
        "fps": fps,
        "cameras": cameras,
        "n_frames": n_frames,
        "source_stage": source_stage,
        "structure_flags": structure_flags,
    }


def load_stage_from_file(path: Path, n_frames: int | None) -> list[int] | None:
    if not path.exists():
        return None
    with h5py.File(path, "r") as root:
        if "/stage" not in root:
            return None
        stage_values = root["/stage"][()].astype(np.int64).tolist()
    if n_frames is not None and len(stage_values) != n_frames:
        return None
    return stage_values


def summarize_episode(source_path: Path, fixed_path: Path) -> EpisodeSummary:
    source_meta = inspect_source_episode(source_path)
    n_frames = source_meta["n_frames"]
    source_stage = source_meta["source_stage"]
    fixed_stage = load_stage_from_file(fixed_path, n_frames)

    if fixed_stage is not None:
        working_stage = fixed_stage
        working_stage_source = "fixed"
    elif source_stage:
        working_stage = source_stage
        working_stage_source = "source"
    elif n_frames is not None:
        working_stage = [0] * n_frames
        working_stage_source = "generated_zero"
    else:
        working_stage = []
        working_stage_source = "none"

    issue_flags = list(source_meta["structure_flags"])
    issue_flags.extend(analyze_stage_flags(source_stage, n_frames))
    issue_flags = list(dict.fromkeys(issue_flags))

    return EpisodeSummary(
        name=source_path.name,
        n_frames=n_frames,
        fps=source_meta["fps"],
        cameras=source_meta["cameras"],
        editable=n_frames is not None and bool(source_meta["cameras"]),
        has_fixed_copy=fixed_path.exists(),
        source_has_stage=bool(source_stage),
        working_stage_source=working_stage_source,
        issue_flags=issue_flags,
        unique_stages=sorted(set(working_stage)) if working_stage else [],
        stage_changes=stage_changes(working_stage),
    )


def normalize_stage_payload(payload: dict[str, Any], n_frames: int) -> list[int]:
    if n_frames <= 0:
        raise ReviewError("Episode frame count must be positive before saving stage.")

    if "stage" in payload:
        values = payload["stage"]
        if not isinstance(values, list):
            raise ReviewError("Field 'stage' must be a JSON array.")
        if len(values) != n_frames:
            raise ReviewError(
                f"Stage length mismatch: expected {n_frames}, got {len(values)}.",
            )
        stage_values = [int(value) for value in values]
    elif "segments" in payload:
        segments = payload["segments"]
        if not isinstance(segments, list) or not segments:
            raise ReviewError("Field 'segments' must be a non-empty JSON array.")

        normalized = sorted(
            (
                int(segment["start"]),
                int(segment["end"]),
                int(segment["stage"]),
            )
            for segment in segments
        )

        if normalized[0][0] != 0:
            raise ReviewError("Segments must start at frame 0.")
        if normalized[-1][1] != n_frames:
            raise ReviewError("Segments must end at the final frame.")

        stage_values = [0] * n_frames
        cursor = 0
        for start, end, stage_value in normalized:
            if start != cursor:
                raise ReviewError("Segments must be contiguous without overlap or gaps.")
            if end <= start:
                raise ReviewError("Each segment must satisfy end > start.")
            if stage_value < 0:
                raise ReviewError("Stage values must be non-negative integers.")
            stage_values[start:end] = [stage_value] * (end - start)
            cursor = end
    else:
        raise ReviewError("Request must include either 'stage' or 'segments'.")

    if any(value < 0 for value in stage_values):
        raise ReviewError("Stage values must be non-negative integers.")
    return stage_values


def write_stage_array(
    source_path: Path,
    stage_values: list[int],
    save_mode: str,
    fixed_dir_name: str,
) -> Path:
    if save_mode not in {"fixed", "overwrite"}:
        raise ReviewError(f"Unsupported save_mode: {save_mode}")

    if save_mode == "fixed":
        target_dir = source_path.parent / fixed_dir_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
    else:
        target_path = source_path

    with h5py.File(target_path, "r+") as root:
        n_frames = first_existing_frame_count(root)
        if n_frames is None:
            raise ReviewError("Unable to infer frame count from HDF5 file.")
        if len(stage_values) != n_frames:
            raise ReviewError(
                f"Refusing to save stage length {len(stage_values)} into episode with {n_frames} frames.",
            )

        array = np.asarray(stage_values, dtype=np.int64)
        if "/stage" in root:
            dataset = root["/stage"]
            if dataset.shape != (n_frames,):
                del root["/stage"]
                root.create_dataset("stage", data=array, dtype=np.int64)
            else:
                dataset[...] = array
        else:
            root.create_dataset("stage", data=array, dtype=np.int64)
    return target_path


class StageReviewApp:
    def __init__(self, dataset_dir: Path, fixed_dir_name: str, allow_overwrite: bool):
        self.dataset_dir = dataset_dir.resolve()
        self.fixed_dir_name = fixed_dir_name
        self.allow_overwrite = allow_overwrite
        if not self.dataset_dir.is_dir():
            raise ReviewError(f"Dataset directory does not exist: {self.dataset_dir}")

    @property
    def fixed_dir(self) -> Path:
        return self.dataset_dir / self.fixed_dir_name

    def source_path(self, episode_name: str) -> Path:
        path = self.dataset_dir / episode_name
        if path.parent != self.dataset_dir or not path.name.endswith(".hdf5"):
            raise ReviewError(f"Invalid episode name: {episode_name}")
        if not path.exists():
            raise ReviewError(f"Episode not found: {episode_name}")
        return path

    def fixed_path(self, episode_name: str) -> Path:
        return self.fixed_dir / episode_name

    def list_summaries(self) -> list[dict[str, Any]]:
        paths = sorted(self.dataset_dir.glob("episode_*.hdf5"), key=episode_sort_key)
        return [asdict(summarize_episode(path, self.fixed_path(path.name))) for path in paths]

    def get_detail(self, episode_name: str) -> dict[str, Any]:
        source_path = self.source_path(episode_name)
        fixed_path = self.fixed_path(episode_name)
        summary = summarize_episode(source_path, fixed_path)

        working_stage = load_stage_from_file(fixed_path, summary.n_frames)
        if working_stage is not None:
            stage_values = working_stage
            stage_source = "fixed"
        else:
            source_meta = inspect_source_episode(source_path)
            if source_meta["source_stage"]:
                stage_values = source_meta["source_stage"]
                stage_source = "source"
            elif summary.n_frames is not None:
                stage_values = [0] * summary.n_frames
                stage_source = "generated_zero"
            else:
                stage_values = []
                stage_source = "none"

        return {
            **asdict(summary),
            "stage": stage_values,
            "segments": build_segments(stage_values),
            "stage_source": stage_source,
            "source_path": str(source_path),
            "fixed_path": str(fixed_path) if fixed_path.exists() else None,
        }

    def load_frame(self, episode_name: str, camera: str, index: int) -> bytes:
        source_path = self.source_path(episode_name)
        with h5py.File(source_path, "r") as root:
            image_group = root.get("/observations/images")
            if not isinstance(image_group, h5py.Group):
                raise ReviewError(f"Episode {episode_name} does not contain /observations/images.")
            if camera not in image_group:
                raise ReviewError(f"Camera '{camera}' not found in episode {episode_name}.")

            dataset = image_group[camera]
            if not dataset.shape:
                raise ReviewError(f"Camera '{camera}' has no frames.")
            max_index = int(dataset.shape[0]) - 1
            if index < 0 or index > max_index:
                raise ReviewError(f"Frame index {index} is out of range 0..{max_index}.")

            return encode_frame_to_jpeg(dataset[index])

    def save_stage(self, episode_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        source_path = self.source_path(episode_name)
        detail = self.get_detail(episode_name)
        if not detail["editable"]:
            raise ReviewError("This episode cannot be edited because the structure is incomplete.")

        save_mode = str(payload.get("save_mode", "fixed"))
        if save_mode == "overwrite" and not self.allow_overwrite:
            raise ReviewError("Overwrite mode is disabled. Relaunch with --allow_overwrite to enable it.")

        stage_values = normalize_stage_payload(payload, int(detail["n_frames"]))
        target_path = write_stage_array(
            source_path=source_path,
            stage_values=stage_values,
            save_mode=save_mode,
            fixed_dir_name=self.fixed_dir_name,
        )
        updated_detail = self.get_detail(episode_name)
        return {
            "ok": True,
            "save_mode": save_mode,
            "target_path": str(target_path),
            "detail": updated_detail,
        }


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], app: StageReviewApp):
        super().__init__(server_address, ReviewRequestHandler)
        self.app = app


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server: ReviewServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(INDEX_HTML)
                return

            if parsed.path == "/api/config":
                self._send_json(
                    {
                        "dataset_dir": str(self.server.app.dataset_dir),
                        "fixed_dir": str(self.server.app.fixed_dir),
                        "fixed_dir_name": self.server.app.fixed_dir_name,
                        "allow_overwrite": self.server.app.allow_overwrite,
                    }
                )
                return

            parts = [unquote(part) for part in parsed.path.split("/") if part]
            if parts == ["api", "episodes"]:
                self._send_json({"episodes": self.server.app.list_summaries()})
                return

            if len(parts) == 3 and parts[:2] == ["api", "episodes"]:
                self._send_json(self.server.app.get_detail(parts[2]))
                return

            if len(parts) == 4 and parts[:2] == ["api", "episodes"] and parts[3] == "frame":
                query = parse_qs(parsed.query)
                camera = query.get("camera", [None])[0]
                index_raw = query.get("index", [None])[0]
                if camera is None or index_raw is None:
                    raise ReviewError("Query parameters 'camera' and 'index' are required.")
                jpeg = self.server.app.load_frame(parts[2], camera, int(index_raw))
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return

            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except ReviewError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover
            self._send_json({"error": f"Unexpected server error: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            parts = [unquote(part) for part in parsed.path.split("/") if part]
            if len(parts) == 4 and parts[:2] == ["api", "episodes"] and parts[3] == "stage":
                payload = self._read_json_body()
                result = self.server.app.save_stage(parts[2], payload)
                self._send_json(result)
                return
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except ReviewError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover
            self._send_json({"error": f"Unexpected server error: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ReviewError("Expected a non-empty JSON request body.")
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ReviewError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise ReviewError("JSON request body must be an object.")
        return payload

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


INDEX_HTML = dedent(
    """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Stage Review</title>
      <style>
        :root {
          --bg: #f4efe6;
          --panel: rgba(255, 251, 244, 0.92);
          --panel-strong: #fffaf2;
          --ink: #182028;
          --muted: #5c6671;
          --line: rgba(24, 32, 40, 0.12);
          --accent: #b74d2b;
          --accent-soft: #f0c3b4;
          --good: #1f7a4d;
          --warn: #b86a19;
          --bad: #a23333;
          --shadow: 0 18px 40px rgba(60, 37, 16, 0.12);
          --radius: 18px;
          --mono: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
          --sans: "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
        }

        * {
          box-sizing: border-box;
        }

        body {
          margin: 0;
          min-height: 100vh;
          font-family: var(--sans);
          color: var(--ink);
          background:
            radial-gradient(circle at top left, rgba(231, 157, 85, 0.18), transparent 24rem),
            radial-gradient(circle at bottom right, rgba(183, 77, 43, 0.12), transparent 22rem),
            linear-gradient(145deg, #efe4d2 0%, #f8f4ed 45%, #f3efe8 100%);
        }

        .shell {
          display: grid;
          grid-template-columns: 320px 1fr;
          min-height: 100vh;
        }

        .sidebar {
          border-right: 1px solid var(--line);
          background: rgba(250, 243, 233, 0.74);
          backdrop-filter: blur(14px);
          padding: 24px 18px 18px;
        }

        .brand {
          margin-bottom: 18px;
        }

        .eyebrow {
          font-size: 12px;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: var(--accent);
          margin-bottom: 8px;
          font-weight: 700;
        }

        h1 {
          margin: 0 0 6px;
          font-size: 28px;
          line-height: 1;
        }

        .subtle {
          color: var(--muted);
          font-size: 13px;
          line-height: 1.5;
        }

        .stack {
          display: grid;
          gap: 14px;
        }

        .card {
          background: var(--panel);
          border: 1px solid rgba(255, 255, 255, 0.6);
          box-shadow: var(--shadow);
          border-radius: var(--radius);
          padding: 16px;
        }

        .card h2 {
          margin: 0 0 10px;
          font-size: 16px;
        }

        .filters {
          display: grid;
          gap: 10px;
        }

        input[type="text"],
        input[type="number"],
        select,
        button {
          width: 100%;
          border-radius: 12px;
          border: 1px solid rgba(24, 32, 40, 0.12);
          padding: 10px 12px;
          font: inherit;
          background: #fffdf9;
          color: var(--ink);
        }

        input[type="range"] {
          width: 100%;
        }

        button {
          cursor: pointer;
          background: linear-gradient(135deg, #fff6ea, #f6e7da);
          transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
        }

        button:hover {
          transform: translateY(-1px);
          border-color: rgba(183, 77, 43, 0.3);
          box-shadow: 0 10px 18px rgba(183, 77, 43, 0.12);
        }

        button.primary {
          background: linear-gradient(135deg, #c25b37, #a73f1e);
          color: white;
          border-color: transparent;
        }

        button.ghost {
          background: rgba(255, 255, 255, 0.6);
        }

        button.danger {
          background: linear-gradient(135deg, #d76e5c, #b83e32);
          color: white;
          border-color: transparent;
        }

        .episodes {
          display: grid;
          gap: 8px;
          max-height: calc(100vh - 290px);
          overflow: auto;
          padding-right: 4px;
        }

        .episode-row {
          border: 1px solid rgba(24, 32, 40, 0.08);
          background: rgba(255, 255, 255, 0.82);
          border-radius: 14px;
          padding: 10px 12px;
          display: grid;
          gap: 6px;
          cursor: pointer;
          transition: transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
        }

        .episode-row.active {
          border-color: rgba(183, 77, 43, 0.45);
          box-shadow: 0 14px 24px rgba(183, 77, 43, 0.14);
          transform: translateY(-1px);
        }

        .episode-row:hover {
          border-color: rgba(183, 77, 43, 0.28);
        }

        .episode-title {
          font-family: var(--mono);
          font-size: 13px;
          font-weight: 700;
        }

        .episode-meta {
          font-size: 12px;
          color: var(--muted);
        }

        .badges {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
        }

        .badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          border-radius: 999px;
          padding: 4px 9px;
          font-size: 12px;
          line-height: 1;
          background: rgba(24, 32, 40, 0.08);
        }

        .badge.issue {
          background: rgba(184, 106, 25, 0.12);
          color: var(--warn);
        }

        .badge.good {
          background: rgba(31, 122, 77, 0.12);
          color: var(--good);
        }

        .badge.bad {
          background: rgba(162, 51, 51, 0.12);
          color: var(--bad);
        }

        .main {
          padding: 24px;
          display: grid;
          gap: 18px;
          align-content: start;
        }

        .topbar,
        .viewer-grid,
        .editor-grid {
          display: grid;
          gap: 18px;
        }

        .topbar {
          grid-template-columns: minmax(360px, 1fr) auto;
          align-items: start;
        }

        .viewer-grid {
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        }

        .editor-grid {
          grid-template-columns: minmax(360px, 1.1fr) minmax(360px, 0.9fr);
        }

        .image-panel {
          display: grid;
          gap: 8px;
        }

        .image-panel img {
          width: 100%;
          aspect-ratio: 4 / 3;
          object-fit: contain;
          border-radius: 16px;
          background: linear-gradient(135deg, #fcf1e7, #f4e4d5);
          border: 1px solid rgba(24, 32, 40, 0.06);
        }

        .image-label {
          font-family: var(--mono);
          font-size: 12px;
          color: var(--muted);
        }

        .toolbar {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          align-items: center;
          justify-content: flex-end;
        }

        .toolbar button {
          width: auto;
        }

        .toolbar .status {
          min-width: 220px;
          text-align: right;
          font-size: 12px;
          color: var(--muted);
        }

        .kv {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 10px;
        }

        .kv-item {
          background: rgba(255, 255, 255, 0.72);
          border-radius: 14px;
          padding: 10px 12px;
          border: 1px solid rgba(24, 32, 40, 0.06);
        }

        .kv-label {
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--muted);
          margin-bottom: 6px;
        }

        .kv-value {
          font-family: var(--mono);
          font-size: 13px;
          word-break: break-all;
        }

        .scrubber {
          display: grid;
          gap: 12px;
        }

        .scrubber-row {
          display: grid;
          grid-template-columns: 1fr 130px 90px;
          gap: 10px;
          align-items: center;
        }

        .timeline-shell {
          display: grid;
          gap: 10px;
        }

        canvas {
          width: 100%;
          height: 150px;
          border-radius: 16px;
          background: linear-gradient(180deg, rgba(255, 255, 255, 0.72), rgba(242, 231, 217, 0.9));
          border: 1px solid rgba(24, 32, 40, 0.08);
        }

        .legend {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
        }

        .legend-item {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 4px 8px;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.72);
          font-size: 12px;
        }

        .swatch {
          width: 12px;
          height: 12px;
          border-radius: 999px;
        }

        .form-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 10px;
        }

        .form-grid button {
          align-self: end;
        }

        .segment-list {
          max-height: 420px;
          overflow: auto;
          display: grid;
          gap: 8px;
        }

        .segment-row {
          display: grid;
          grid-template-columns: 1fr 120px auto;
          gap: 10px;
          align-items: center;
          padding: 12px;
          border-radius: 14px;
          background: rgba(255, 255, 255, 0.72);
          border: 1px solid rgba(24, 32, 40, 0.07);
        }

        .segment-row.active {
          border-color: rgba(183, 77, 43, 0.4);
          box-shadow: 0 8px 16px rgba(183, 77, 43, 0.12);
        }

        .segment-actions {
          display: flex;
          gap: 8px;
        }

        .segment-actions button {
          width: auto;
          padding: 8px 10px;
        }

        .empty {
          color: var(--muted);
          font-size: 13px;
          padding: 8px 0;
        }

        .hidden {
          display: none !important;
        }

        @media (max-width: 1200px) {
          .shell {
            grid-template-columns: 1fr;
          }

          .sidebar {
            border-right: none;
            border-bottom: 1px solid var(--line);
          }

          .episodes {
            max-height: 320px;
          }

          .topbar,
          .editor-grid {
            grid-template-columns: 1fr;
          }
        }

        @media (max-width: 720px) {
          .main {
            padding: 16px;
          }

          .sidebar {
            padding: 16px;
          }

          .scrubber-row,
          .form-grid {
            grid-template-columns: 1fr;
          }
        }
      </style>
    </head>
    <body>
      <div class="shell">
        <aside class="sidebar">
          <div class="brand">
            <div class="eyebrow">Stage Review</div>
            <h1>人工复核台</h1>
            <div class="subtle" id="datasetInfo">正在加载数据目录...</div>
          </div>

          <div class="stack">
            <section class="card">
              <h2>筛选</h2>
              <div class="filters">
                <input id="searchInput" type="text" placeholder="搜索 episode 名称" />
                <select id="filterSelect">
                  <option value="all">全部</option>
                  <option value="issues">仅异常</option>
                  <option value="fixed">已有 fixed 副本</option>
                  <option value="editable">仅可编辑</option>
                </select>
                <button id="refreshBtn" class="ghost">刷新列表</button>
              </div>
            </section>

            <section class="card">
              <h2>Episode</h2>
              <div id="episodeCount" class="subtle">0 条</div>
              <div id="episodeList" class="episodes"></div>
            </section>
          </div>
        </aside>

        <main class="main">
          <section class="card topbar">
            <div class="stack">
              <div>
                <div class="eyebrow" id="episodeEyebrow">未选择</div>
                <h2 id="episodeTitle" style="font-size: 24px; margin: 0 0 10px;">选择一个 episode 开始复核</h2>
                <div id="episodeBadges" class="badges"></div>
              </div>
              <div id="episodeMeta" class="kv"></div>
            </div>

            <div class="toolbar">
              <button id="playBtn">播放</button>
              <button id="saveFixedBtn" class="primary">保存到 fixed</button>
              <button id="saveOverwriteBtn" class="danger hidden">覆盖原文件</button>
              <div id="statusLine" class="status">等待选择 episode</div>
            </div>
          </section>

          <section class="card">
            <div class="viewer-grid" id="viewerGrid"></div>
          </section>

          <section class="card scrubber">
            <div class="scrubber-row">
              <input id="frameRange" type="range" min="0" max="0" value="0" />
              <input id="frameInput" type="number" min="0" value="0" />
              <div id="frameStage" class="badge">stage -</div>
            </div>
            <div class="timeline-shell">
              <canvas id="timelineCanvas"></canvas>
              <div id="timelineLegend" class="legend"></div>
              <div class="subtle">点击时间轴跳转，拖动白色边界手柄调整阶段切换位置，左右方向键逐帧移动。</div>
            </div>
          </section>

          <section class="editor-grid">
            <section class="card stack">
              <div>
                <h2>快速编辑</h2>
                <div class="subtle">所有编辑都先发生在浏览器内，只有点击保存才会写入 HDF5。</div>
              </div>

              <div class="stack">
                <div>
                  <div class="subtle" style="margin-bottom: 8px;">当前 segment 改成指定 stage</div>
                  <div class="form-grid">
                    <input id="currentStageInput" type="number" min="0" value="0" />
                    <button id="fillCurrentBtn">覆盖当前段</button>
                    <button id="splitAtCurrentBtn">从当前帧切分</button>
                  </div>
                </div>

                <div>
                  <div class="subtle" style="margin-bottom: 8px;">按区间重写 stage</div>
                  <div class="form-grid">
                    <input id="rangeStartInput" type="number" min="0" value="0" placeholder="start" />
                    <input id="rangeEndInput" type="number" min="1" value="1" placeholder="end" />
                    <input id="rangeStageInput" type="number" min="0" value="0" placeholder="stage" />
                    <button id="applyRangeBtn">应用区间</button>
                    <button id="resetToSourceBtn" class="ghost">恢复到载入状态</button>
                    <button id="normalizeBtn" class="ghost">规范化段</button>
                  </div>
                </div>
              </div>
            </section>

            <section class="card stack">
              <div>
                <h2>Segments</h2>
                <div class="subtle" id="segmentSummary">0 段</div>
              </div>
              <div id="segmentList" class="segment-list"></div>
            </section>
          </section>
        </main>
      </div>

      <script>
        const STAGE_COLORS = [
          "#355c7d",
          "#c06c84",
          "#f67280",
          "#f8b195",
          "#6c5b7b",
          "#99b898",
          "#2a9d8f",
          "#e76f51",
          "#e9c46a",
          "#457b9d",
        ];

        const state = {
          config: null,
          episodes: [],
          selectedEpisode: null,
          detail: null,
          originalStage: [],
          dirty: false,
          playTimer: null,
          currentFrame: 0,
          dragBoundaryIndex: null,
        };

        const els = {
          datasetInfo: document.getElementById("datasetInfo"),
          episodeCount: document.getElementById("episodeCount"),
          episodeList: document.getElementById("episodeList"),
          episodeEyebrow: document.getElementById("episodeEyebrow"),
          episodeTitle: document.getElementById("episodeTitle"),
          episodeBadges: document.getElementById("episodeBadges"),
          episodeMeta: document.getElementById("episodeMeta"),
          viewerGrid: document.getElementById("viewerGrid"),
          frameRange: document.getElementById("frameRange"),
          frameInput: document.getElementById("frameInput"),
          frameStage: document.getElementById("frameStage"),
          timelineCanvas: document.getElementById("timelineCanvas"),
          timelineLegend: document.getElementById("timelineLegend"),
          currentStageInput: document.getElementById("currentStageInput"),
          rangeStartInput: document.getElementById("rangeStartInput"),
          rangeEndInput: document.getElementById("rangeEndInput"),
          rangeStageInput: document.getElementById("rangeStageInput"),
          applyRangeBtn: document.getElementById("applyRangeBtn"),
          fillCurrentBtn: document.getElementById("fillCurrentBtn"),
          splitAtCurrentBtn: document.getElementById("splitAtCurrentBtn"),
          resetToSourceBtn: document.getElementById("resetToSourceBtn"),
          normalizeBtn: document.getElementById("normalizeBtn"),
          saveFixedBtn: document.getElementById("saveFixedBtn"),
          saveOverwriteBtn: document.getElementById("saveOverwriteBtn"),
          playBtn: document.getElementById("playBtn"),
          statusLine: document.getElementById("statusLine"),
          segmentSummary: document.getElementById("segmentSummary"),
          segmentList: document.getElementById("segmentList"),
          refreshBtn: document.getElementById("refreshBtn"),
          searchInput: document.getElementById("searchInput"),
          filterSelect: document.getElementById("filterSelect"),
        };

        function stageColor(stage) {
          return STAGE_COLORS[Math.abs(Number(stage)) % STAGE_COLORS.length];
        }

        async function api(path, options = {}) {
          const response = await fetch(path, {
            headers: {"Content-Type": "application/json"},
            ...options,
          });
          const data = response.headers.get("content-type")?.includes("application/json")
            ? await response.json()
            : await response.text();
          if (!response.ok) {
            throw new Error(data?.error || data || `HTTP ${response.status}`);
          }
          return data;
        }

        function setStatus(text, isError = false) {
          els.statusLine.textContent = text;
          els.statusLine.style.color = isError ? "var(--bad)" : "var(--muted)";
        }

        function cloneStage(values) {
          return Array.isArray(values) ? values.map((value) => Number(value)) : [];
        }

        function buildSegments(stageValues) {
          if (!stageValues?.length) {
            return [];
          }
          const segments = [];
          let start = 0;
          let current = Number(stageValues[0]);
          for (let idx = 1; idx < stageValues.length; idx += 1) {
            const value = Number(stageValues[idx]);
            if (value !== current) {
              segments.push({start, end: idx, stage: current});
              start = idx;
              current = value;
            }
          }
          segments.push({start, end: stageValues.length, stage: current});
          return segments;
        }

        function markDirty(text = "有未保存编辑") {
          state.dirty = true;
          setStatus(text);
          renderEpisodeList();
        }

        function clearDirty(text) {
          state.dirty = false;
          setStatus(text || "已同步");
          renderEpisodeList();
        }

        function currentSegments() {
          return buildSegments(state.detail?.stage || []);
        }

        function currentSegmentIndex() {
          const segments = currentSegments();
          for (let idx = 0; idx < segments.length; idx += 1) {
            const segment = segments[idx];
            if (state.currentFrame >= segment.start && state.currentFrame < segment.end) {
              return idx;
            }
          }
          return -1;
        }

        function clampFrame(frame) {
          if (!state.detail?.n_frames) {
            return 0;
          }
          return Math.max(0, Math.min(state.detail.n_frames - 1, Number(frame) || 0));
        }

        function setCurrentFrame(frame) {
          if (!state.detail) {
            return;
          }
          state.currentFrame = clampFrame(frame);
          els.frameRange.value = String(state.currentFrame);
          els.frameInput.value = String(state.currentFrame);
          els.rangeStartInput.value = String(state.currentFrame);
          els.rangeEndInput.value = String(Math.min(state.detail.n_frames, state.currentFrame + 1));
          const stage = state.detail.stage[state.currentFrame];
          els.frameStage.textContent = `stage ${stage ?? "-"}`;
          els.frameStage.style.background = `${stageColor(stage || 0)}22`;
          els.frameStage.style.color = stageColor(stage || 0);
          renderFrames();
          renderTimeline();
          renderSegments();
        }

        function renderBadges(flags, target) {
          target.innerHTML = "";
          if (!flags?.length) {
            const clean = document.createElement("span");
            clean.className = "badge good";
            clean.textContent = "无异常";
            target.appendChild(clean);
            return;
          }
          for (const flag of flags) {
            const badge = document.createElement("span");
            badge.className = "badge issue";
            badge.textContent = flag;
            target.appendChild(badge);
          }
        }

        function renderEpisodeList() {
          const search = els.searchInput.value.trim().toLowerCase();
          const filter = els.filterSelect.value;
          const items = state.episodes.filter((episode) => {
            if (search && !episode.name.toLowerCase().includes(search)) {
              return false;
            }
            if (filter === "issues" && !episode.issue_flags?.length) {
              return false;
            }
            if (filter === "fixed" && !episode.has_fixed_copy) {
              return false;
            }
            if (filter === "editable" && !episode.editable) {
              return false;
            }
            return true;
          });

          els.episodeCount.textContent = `${items.length} / ${state.episodes.length} 条`;
          els.episodeList.innerHTML = "";

          if (!items.length) {
            const empty = document.createElement("div");
            empty.className = "empty";
            empty.textContent = "没有符合筛选条件的 episode。";
            els.episodeList.appendChild(empty);
            return;
          }

          for (const episode of items) {
            const row = document.createElement("button");
            row.type = "button";
            row.className = `episode-row ${episode.name === state.selectedEpisode ? "active" : ""}`;
            row.addEventListener("click", () => selectEpisode(episode.name));

            const title = document.createElement("div");
            title.className = "episode-title";
            title.textContent = episode.name;

            const meta = document.createElement("div");
            meta.className = "episode-meta";
            const segments = episode.stage_changes?.length || 0;
            const frames = episode.n_frames == null ? "?" : episode.n_frames;
            const source = episode.working_stage_source;
            meta.textContent = `${frames} frames · ${segments} segments · ${source}`;

            const badges = document.createElement("div");
            badges.className = "badges";
            if (episode.has_fixed_copy) {
              const saved = document.createElement("span");
              saved.className = "badge good";
              saved.textContent = "fixed";
              badges.appendChild(saved);
            }
            if (episode.name === state.selectedEpisode && state.dirty) {
              const dirty = document.createElement("span");
              dirty.className = "badge bad";
              dirty.textContent = "unsaved";
              badges.appendChild(dirty);
            }
            for (const flag of (episode.issue_flags || []).slice(0, 3)) {
              const badge = document.createElement("span");
              badge.className = "badge issue";
              badge.textContent = flag;
              badges.appendChild(badge);
            }

            row.appendChild(title);
            row.appendChild(meta);
            row.appendChild(badges);
            els.episodeList.appendChild(row);
          }
        }

        function renderMeta() {
          if (!state.detail) {
            els.episodeEyebrow.textContent = "未选择";
            els.episodeTitle.textContent = "选择一个 episode 开始复核";
            els.episodeMeta.innerHTML = "";
            els.episodeBadges.innerHTML = "";
            return;
          }

          els.episodeEyebrow.textContent = state.detail.stage_source === "fixed"
            ? "当前使用 fixed 副本"
            : state.detail.stage_source === "generated_zero"
              ? "当前使用临时零值 stage"
              : "当前使用源文件 stage";
          els.episodeTitle.textContent = state.detail.name;
          renderBadges(state.detail.issue_flags, els.episodeBadges);

          const pairs = [
            ["frames", state.detail.n_frames ?? "-"],
            ["fps", state.detail.fps ?? "-"],
            ["cameras", (state.detail.cameras || []).join(", ") || "-"],
            ["editable", state.detail.editable ? "yes" : "no"],
            ["source path", state.detail.source_path || "-"],
            ["fixed path", state.detail.fixed_path || "-"],
          ];

          els.episodeMeta.innerHTML = "";
          for (const [label, value] of pairs) {
            const item = document.createElement("div");
            item.className = "kv-item";
            item.innerHTML = `
              <div class="kv-label">${label}</div>
              <div class="kv-value">${value}</div>
            `;
            els.episodeMeta.appendChild(item);
          }
        }

        function renderFrames() {
          els.viewerGrid.innerHTML = "";
          if (!state.detail) {
            return;
          }

          const cameras = state.detail.cameras || [];
          if (!cameras.length) {
            const empty = document.createElement("div");
            empty.className = "empty";
            empty.textContent = "这个 episode 没有可显示的相机图像。";
            els.viewerGrid.appendChild(empty);
            return;
          }

          for (const camera of cameras) {
            const panel = document.createElement("div");
            panel.className = "image-panel";
            const label = document.createElement("div");
            label.className = "image-label";
            label.textContent = camera;
            const image = document.createElement("img");
            image.alt = camera;
            image.src = `/api/episodes/${encodeURIComponent(state.detail.name)}/frame?camera=${encodeURIComponent(camera)}&index=${state.currentFrame}&t=${Date.now()}`;
            panel.appendChild(label);
            panel.appendChild(image);
            els.viewerGrid.appendChild(panel);
          }
        }

        function renderLegend(segments) {
          const stages = [...new Set(segments.map((segment) => segment.stage))];
          els.timelineLegend.innerHTML = "";
          for (const stage of stages) {
            const item = document.createElement("div");
            item.className = "legend-item";
            item.innerHTML = `<span class="swatch" style="background:${stageColor(stage)}"></span>stage ${stage}`;
            els.timelineLegend.appendChild(item);
          }
        }

        function renderTimeline() {
          const canvas = els.timelineCanvas;
          const ctx = canvas.getContext("2d");
          const rect = canvas.getBoundingClientRect();
          const dpr = window.devicePixelRatio || 1;
          canvas.width = rect.width * dpr;
          canvas.height = rect.height * dpr;
          ctx.scale(dpr, dpr);

          ctx.clearRect(0, 0, rect.width, rect.height);
          ctx.fillStyle = "rgba(255,255,255,0.35)";
          ctx.fillRect(0, 0, rect.width, rect.height);

          if (!state.detail?.n_frames || !state.detail.stage?.length) {
            ctx.fillStyle = "#5c6671";
            ctx.font = "14px sans-serif";
            ctx.fillText("无可绘制的时间轴", 18, 32);
            return;
          }

          const segments = currentSegments();
          const width = rect.width;
          const barY = 34;
          const barHeight = 54;
          const total = state.detail.n_frames;

          for (const segment of segments) {
            const x = (segment.start / total) * width;
            const segWidth = Math.max(2, ((segment.end - segment.start) / total) * width);
            ctx.fillStyle = stageColor(segment.stage);
            ctx.fillRect(x, barY, segWidth, barHeight);
          }

          ctx.fillStyle = "rgba(24,32,40,0.7)";
          ctx.font = "12px monospace";
          ctx.fillText(`0`, 0, 20);
          ctx.fillText(`${total - 1}`, Math.max(0, width - 70), 20);

          ctx.strokeStyle = "rgba(24,32,40,0.2)";
          ctx.lineWidth = 1;
          ctx.strokeRect(0.5, barY + 0.5, width - 1, barHeight);

          for (let idx = 1; idx < segments.length; idx += 1) {
            const x = (segments[idx].start / total) * width;
            ctx.strokeStyle = "rgba(255,255,255,0.94)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x, barY - 8);
            ctx.lineTo(x, barY + barHeight + 8);
            ctx.stroke();

            ctx.fillStyle = "#fff";
            ctx.beginPath();
            ctx.arc(x, barY + barHeight + 18, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = "rgba(24,32,40,0.22)";
            ctx.stroke();
          }

          const currentX = (state.currentFrame / total) * width;
          ctx.strokeStyle = "#182028";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(currentX, barY - 18);
          ctx.lineTo(currentX, rect.height - 12);
          ctx.stroke();

          renderLegend(segments);
        }

        function renderSegments() {
          if (!state.detail) {
            els.segmentList.innerHTML = "";
            els.segmentSummary.textContent = "0 段";
            return;
          }

          const segments = currentSegments();
          els.segmentSummary.textContent = `${segments.length} 段`;
          els.segmentList.innerHTML = "";
          const activeIndex = currentSegmentIndex();

          if (!segments.length) {
            const empty = document.createElement("div");
            empty.className = "empty";
            empty.textContent = "没有 stage 段。";
            els.segmentList.appendChild(empty);
            return;
          }

          for (let idx = 0; idx < segments.length; idx += 1) {
            const segment = segments[idx];
            const row = document.createElement("div");
            row.className = `segment-row ${idx === activeIndex ? "active" : ""}`;

            const info = document.createElement("div");
            info.innerHTML = `
              <div class="episode-title">stage ${segment.stage}</div>
              <div class="episode-meta">${segment.start} → ${segment.end} (${segment.end - segment.start} 帧)</div>
            `;

            const stageInput = document.createElement("input");
            stageInput.type = "number";
            stageInput.min = "0";
            stageInput.value = String(segment.stage);
            stageInput.addEventListener("change", () => {
              applyRange(segment.start, segment.end, Number(stageInput.value));
            });

            const actions = document.createElement("div");
            actions.className = "segment-actions";

            const jumpBtn = document.createElement("button");
            jumpBtn.type = "button";
            jumpBtn.className = "ghost";
            jumpBtn.textContent = "跳转";
            jumpBtn.addEventListener("click", () => setCurrentFrame(segment.start));

            const mergeBtn = document.createElement("button");
            mergeBtn.type = "button";
            mergeBtn.textContent = "并到前段";
            mergeBtn.disabled = idx === 0;
            mergeBtn.addEventListener("click", () => {
              if (idx === 0) {
                return;
              }
              const prev = segments[idx - 1];
              applyRange(segment.start, segment.end, prev.stage);
            });

            actions.appendChild(jumpBtn);
            actions.appendChild(mergeBtn);

            row.appendChild(info);
            row.appendChild(stageInput);
            row.appendChild(actions);
            els.segmentList.appendChild(row);
          }
        }

        function applyRange(start, end, stage) {
          if (!state.detail) {
            return;
          }
          const nFrames = state.detail.n_frames;
          const rangeStart = Math.max(0, Math.min(nFrames - 1, Number(start)));
          const rangeEnd = Math.max(rangeStart + 1, Math.min(nFrames, Number(end)));
          const stageValue = Math.max(0, Number(stage) || 0);
          state.detail.stage.fill(stageValue, rangeStart, rangeEnd);
          markDirty(`已修改区间 ${rangeStart} → ${rangeEnd} 为 stage ${stageValue}`);
          setCurrentFrame(state.currentFrame);
        }

        function fillCurrentSegment(stageValue) {
          const segments = currentSegments();
          const index = currentSegmentIndex();
          if (index < 0) {
            return;
          }
          const segment = segments[index];
          applyRange(segment.start, segment.end, stageValue);
        }

        function splitAtCurrent(stageValue) {
          const segments = currentSegments();
          const index = currentSegmentIndex();
          if (index < 0) {
            return;
          }
          const segment = segments[index];
          const frame = clampFrame(state.currentFrame);
          if (frame <= segment.start || frame >= segment.end) {
            setStatus("当前帧不在可切分位置", true);
            return;
          }
          applyRange(frame, segment.end, stageValue);
        }

        function normalizeStage() {
          if (!state.detail?.stage?.length) {
            return;
          }
          state.detail.stage = cloneStage(state.detail.stage);
          markDirty("已按当前 stage 数组重新归并分段");
          setCurrentFrame(state.currentFrame);
        }

        function restoreOriginalStage() {
          if (!state.detail) {
            return;
          }
          state.detail.stage = cloneStage(state.originalStage);
          clearDirty("已恢复到当前载入状态");
          setCurrentFrame(state.currentFrame);
        }

        async function loadConfig() {
          state.config = await api("/api/config");
          els.datasetInfo.textContent = `${state.config.dataset_dir}`;
          els.saveOverwriteBtn.classList.toggle("hidden", !state.config.allow_overwrite);
        }

        async function loadEpisodes() {
          const data = await api("/api/episodes");
          state.episodes = data.episodes || [];
          renderEpisodeList();
        }

        async function selectEpisode(name) {
          if (state.selectedEpisode === name && state.detail) {
            return;
          }

          if (state.playTimer) {
            togglePlay(false);
          }

          setStatus(`加载 ${name} ...`);
          const detail = await api(`/api/episodes/${encodeURIComponent(name)}`);
          state.selectedEpisode = name;
          state.detail = {
            ...detail,
            stage: cloneStage(detail.stage),
          };
          state.originalStage = cloneStage(detail.stage);
          state.currentFrame = 0;
          state.dirty = false;

          els.frameRange.max = String(Math.max(0, (detail.n_frames || 1) - 1));
          els.frameInput.max = String(Math.max(0, (detail.n_frames || 1) - 1));
          els.rangeStartInput.max = String(Math.max(0, (detail.n_frames || 1) - 1));
          els.rangeEndInput.max = String(detail.n_frames || 1);

          renderEpisodeList();
          renderMeta();
          setCurrentFrame(0);
          clearDirty(`已载入 ${name}`);
        }

        async function saveStage(saveMode) {
          if (!state.detail) {
            return;
          }
          const payload = {
            save_mode: saveMode,
            stage: state.detail.stage,
          };
          setStatus(`保存中 (${saveMode}) ...`);
          const result = await api(`/api/episodes/${encodeURIComponent(state.detail.name)}/stage`, {
            method: "POST",
            body: JSON.stringify(payload),
          });

          state.detail = {
            ...result.detail,
            stage: cloneStage(result.detail.stage),
          };
          state.originalStage = cloneStage(result.detail.stage);
          state.currentFrame = clampFrame(state.currentFrame);
          await loadEpisodes();
          renderMeta();
          setCurrentFrame(state.currentFrame);
          clearDirty(`已保存到 ${result.target_path}`);
        }

        function togglePlay(forceState = null) {
          if (!state.detail?.n_frames) {
            return;
          }
          const nextState = forceState ?? !state.playTimer;
          if (!nextState) {
            clearInterval(state.playTimer);
            state.playTimer = null;
            els.playBtn.textContent = "播放";
            return;
          }
          const intervalMs = Math.max(30, Math.round(1000 / Math.max(1, state.detail.fps || 25)));
          state.playTimer = setInterval(() => {
            const nextFrame = state.currentFrame + 1;
            setCurrentFrame(nextFrame >= state.detail.n_frames ? 0 : nextFrame);
          }, intervalMs);
          els.playBtn.textContent = "暂停";
        }

        function boundaryIndexAt(x) {
          if (!state.detail?.n_frames) {
            return null;
          }
          const rect = els.timelineCanvas.getBoundingClientRect();
          const width = rect.width;
          const tolerance = 10;
          const segments = currentSegments();
          for (let idx = 1; idx < segments.length; idx += 1) {
            const boundaryX = (segments[idx].start / state.detail.n_frames) * width;
            if (Math.abs(boundaryX - x) <= tolerance) {
              return idx;
            }
          }
          return null;
        }

        function moveBoundary(boundaryIndex, targetFrame) {
          const segments = currentSegments();
          if (!segments.length || boundaryIndex == null || boundaryIndex <= 0 || boundaryIndex >= segments.length) {
            return;
          }
          const left = segments[boundaryIndex - 1];
          const right = segments[boundaryIndex];
          const minFrame = left.start + 1;
          const maxFrame = right.end - 1;
          const newBoundary = Math.max(minFrame, Math.min(maxFrame, targetFrame));
          const oldBoundary = right.start;
          if (newBoundary === oldBoundary) {
            return;
          }
          if (newBoundary > oldBoundary) {
            state.detail.stage.fill(left.stage, oldBoundary, newBoundary);
          } else {
            state.detail.stage.fill(right.stage, newBoundary, oldBoundary);
          }
          markDirty(`已移动切换点到 ${newBoundary}`);
          setCurrentFrame(newBoundary);
        }

        function frameFromCanvasX(clientX) {
          const rect = els.timelineCanvas.getBoundingClientRect();
          if (!state.detail?.n_frames) {
            return 0;
          }
          const x = Math.max(0, Math.min(rect.width, clientX - rect.left));
          const ratio = rect.width === 0 ? 0 : x / rect.width;
          return clampFrame(Math.round(ratio * (state.detail.n_frames - 1)));
        }

        function bindEvents() {
          els.frameRange.addEventListener("input", () => setCurrentFrame(Number(els.frameRange.value)));
          els.frameInput.addEventListener("change", () => setCurrentFrame(Number(els.frameInput.value)));
          els.applyRangeBtn.addEventListener("click", () => {
            applyRange(
              Number(els.rangeStartInput.value),
              Number(els.rangeEndInput.value),
              Number(els.rangeStageInput.value),
            );
          });
          els.fillCurrentBtn.addEventListener("click", () => fillCurrentSegment(Number(els.currentStageInput.value)));
          els.splitAtCurrentBtn.addEventListener("click", () => splitAtCurrent(Number(els.currentStageInput.value)));
          els.resetToSourceBtn.addEventListener("click", restoreOriginalStage);
          els.normalizeBtn.addEventListener("click", normalizeStage);
          els.saveFixedBtn.addEventListener("click", () => saveStage("fixed").catch((error) => setStatus(error.message, true)));
          els.saveOverwriteBtn.addEventListener("click", () => saveStage("overwrite").catch((error) => setStatus(error.message, true)));
          els.playBtn.addEventListener("click", () => togglePlay());
          els.refreshBtn.addEventListener("click", () => loadEpisodes().catch((error) => setStatus(error.message, true)));
          els.searchInput.addEventListener("input", renderEpisodeList);
          els.filterSelect.addEventListener("change", renderEpisodeList);

          els.timelineCanvas.addEventListener("mousedown", (event) => {
            const rect = els.timelineCanvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const boundary = boundaryIndexAt(x);
            if (boundary != null) {
              state.dragBoundaryIndex = boundary;
              return;
            }
            setCurrentFrame(frameFromCanvasX(event.clientX));
          });

          window.addEventListener("mousemove", (event) => {
            if (state.dragBoundaryIndex == null) {
              return;
            }
            moveBoundary(state.dragBoundaryIndex, frameFromCanvasX(event.clientX));
          });

          window.addEventListener("mouseup", () => {
            state.dragBoundaryIndex = null;
          });

          window.addEventListener("resize", renderTimeline);

          window.addEventListener("keydown", (event) => {
            if (!state.detail?.n_frames) {
              return;
            }
            if (event.key === "ArrowRight") {
              event.preventDefault();
              setCurrentFrame(state.currentFrame + 1);
            } else if (event.key === "ArrowLeft") {
              event.preventDefault();
              setCurrentFrame(state.currentFrame - 1);
            } else if (event.key === " ") {
              event.preventDefault();
              togglePlay();
            }
          });
        }

        async function boot() {
          try {
            bindEvents();
            await loadConfig();
            await loadEpisodes();
            if (state.episodes.length) {
              await selectEpisode(state.episodes[0].name);
            } else {
              setStatus("当前目录下没有 episode_*.hdf5 文件", true);
            }
          } catch (error) {
            setStatus(error.message, true);
          }
        }

        boot();
      </script>
    </body>
    </html>
    """
).strip()


def main() -> None:
    args = parse_args()
    app = StageReviewApp(
        dataset_dir=args.dataset_dir,
        fixed_dir_name=args.fixed_dir_name,
        allow_overwrite=args.allow_overwrite,
    )
    server = ReviewServer((args.host, args.port), app)
    print(f"Stage review UI serving {app.dataset_dir}")
    print(f"Open http://{args.host}:{args.port}")
    print(f"Fixed copies will be written to {app.fixed_dir}")
    if args.allow_overwrite:
        print("Overwrite mode is enabled.")
    server.serve_forever()


if __name__ == "__main__":
    main()
