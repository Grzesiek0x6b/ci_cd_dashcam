"""Wczytywanie klatek z wideo."""

from pathlib import Path

import cv2
import numpy as np


def load_frames(
    path: str | Path, max_frames: int = 180, stride: int = 4, width: int = 960
) -> tuple[list[np.ndarray], float]:
    """Zwraca (klatki RGB przeskalowane do `width`, fps po decymacji)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps > 0:  # 0, -1 lub NaN, gdy kontener nie podaje fps
        fps = 30.0
    frames: list[np.ndarray] = []
    i = 0
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            h, w = bgr.shape[:2]
            bgr = cv2.resize(bgr, (width, int(h * width / w)))
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames, fps / stride
