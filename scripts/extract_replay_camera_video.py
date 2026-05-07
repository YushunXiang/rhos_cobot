#!/usr/bin/env python3
"""Extract a replay camera stream from an HDF5 episode into a video and contact sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import ImageFont

from rhos_cobot.pillow_overlay import (
    bgr_to_pil,
    draw_text_box,
    load_font,
    pil_to_bgr,
    resolve_font_path,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", help="Path to the replay HDF5 episode file.")
    parser.add_argument(
        "--camera",
        default="cam_high",
        help="Camera name under /observations/images. Default: cam_high",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for generated artifacts. Default: /tmp/<episode_stem>_<camera>_extract",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=12,
        help="Number of sampled frames in the contact sheet. Default: 12",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="Number of columns in the contact sheet. Default: 4",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=320,
        help="Contact-sheet tile width. Default: 320",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=240,
        help="Contact-sheet tile height. Default: 240",
    )
    parser.add_argument(
        "--input-color-space",
        choices=("rgb", "bgr"),
        default="rgb",
        help=(
            "Color space stored in the replay frames before exporting. "
            "OpenCV writers expect BGR; datasets from this ROS replay pipeline are typically RGB. "
            "Default: rgb"
        ),
    )
    parser.add_argument(
        "--video-font-path",
        default=None,
        type=Path,
        help="Path to a TrueType font for tile labels. If omitted, auto-discover "
        "CJK -> Latin system fonts, or honor $RHOS_COBOT_VIDEO_FONT.",
    )
    return parser.parse_args()


def _decode_frame(frame_data: np.ndarray | bytes, *, compressed: bool) -> np.ndarray:
    if compressed:
        if isinstance(frame_data, np.ndarray):
            encoded = np.frombuffer(frame_data.tobytes(), np.uint8)
        else:
            encoded = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    else:
        frame = np.asarray(frame_data)
    if frame is None:
        raise RuntimeError("failed to decode frame")
    return frame


def _default_output_dir(dataset_path: Path, camera: str) -> Path:
    return Path("/tmp") / f"{dataset_path.stem}_{camera}_extract"


def _prepare_frame_for_output(frame: np.ndarray, *, input_color_space: str) -> np.ndarray:
    if input_color_space == "rgb" and frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def _annotate_tile(
    tile: np.ndarray,
    label: str,
    font: ImageFont.FreeTypeFont,
    *,
    tile_width: int,
) -> np.ndarray:
    """Paint a black header strip and white label text on a contact-sheet tile."""
    pil_tile = bgr_to_pil(tile)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(pil_tile)
    draw_text_box(
        draw,
        (8, 6),
        label,
        font,
        padding=(6, 4),
        fg=(255, 255, 255),
        bg=(0, 0, 0),
        box_width=tile_width,
    )
    return pil_to_bgr(pil_tile)


def main() -> int:
    args = _parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir(dataset_path, args.camera)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / f"{dataset_path.stem}_{args.camera}.mp4"
    contact_path = output_dir / f"{dataset_path.stem}_{args.camera}_contact_{args.samples}.jpg"
    font_path = resolve_font_path(args.video_font_path)
    tile_font = load_font(18, font_path)

    with h5py.File(dataset_path, "r") as handle:
        images_group = handle["/observations/images"]
        cameras = list(images_group.keys())
        if args.camera not in cameras:
            raise KeyError(f"camera {args.camera!r} not found; available cameras: {cameras}")

        compressed = bool(handle.attrs.get("compress", False))
        fps = float(handle.attrs.get("fps", 25))
        dataset = images_group[args.camera]
        num_frames = len(dataset)
        if num_frames == 0:
            raise RuntimeError(f"camera {args.camera!r} has no frames")

        first_frame = _decode_frame(dataset[0], compressed=compressed)
        height, width = first_frame.shape[:2]

        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer for {video_path}")

        try:
            for frame_idx in range(num_frames):
                frame = _decode_frame(dataset[frame_idx], compressed=compressed)
                frame = _prepare_frame_for_output(frame, input_color_space=args.input_color_space)
                writer.write(frame)
        finally:
            writer.release()

        sample_count = max(1, min(args.samples, num_frames))
        sample_indices = np.linspace(0, num_frames - 1, sample_count, dtype=int)
        tiles: list[np.ndarray] = []
        for frame_idx in sample_indices:
            frame = _decode_frame(dataset[frame_idx], compressed=compressed)
            frame = _prepare_frame_for_output(frame, input_color_space=args.input_color_space)
            frame = cv2.resize(frame, (args.tile_width, args.tile_height))
            label = f"frame {frame_idx}  t={frame_idx / fps:.1f}s"
            frame = _annotate_tile(frame, label, tile_font, tile_width=args.tile_width)
            tiles.append(frame)

    rows: list[np.ndarray] = []
    columns = max(1, args.columns)
    for row_start in range(0, len(tiles), columns):
        row_tiles = tiles[row_start : row_start + columns]
        while len(row_tiles) < columns:
            row_tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(row_tiles, axis=1))
    contact_sheet = np.concatenate(rows, axis=0)
    if not cv2.imwrite(str(contact_path), contact_sheet):
        raise RuntimeError(f"failed to write contact sheet to {contact_path}")

    duration_sec = num_frames / fps if fps > 0 else 0.0
    print(f"dataset_path={dataset_path}")
    print(f"camera={args.camera}")
    print(f"available_cameras={','.join(cameras)}")
    print(f"fps={fps}")
    print(f"num_frames={num_frames}")
    print(f"duration_sec={duration_sec:.2f}")
    print(f"video_path={video_path}")
    print(f"contact_path={contact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
