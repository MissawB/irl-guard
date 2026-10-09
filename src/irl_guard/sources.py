"""Sources d'images. V1 : fichier vidéo lu avec OpenCV."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from irl_guard.types import Frame


class VideoSource:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {self.path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        index = 0
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                yield Frame(index, index / self.fps, image)
                index += 1
        finally:
            cap.release()
