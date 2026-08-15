"""Adapter de I/O: leitura de metadados de video via OpenCV. Sem logica de negocio aqui."""
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoProbeResult:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


def probe_video(path: Path) -> VideoProbeResult:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Nao foi possivel abrir o video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = frame_count / fps if fps > 0 else 0.0
        return VideoProbeResult(
            path=path, fps=fps, frame_count=frame_count,
            width=width, height=height, duration_sec=duration_sec,
        )
    finally:
        cap.release()
