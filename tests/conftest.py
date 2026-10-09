import cv2
import numpy as np
import pytest

from irl_guard.config import SessionConfig


@pytest.fixture
def cfg() -> SessionConfig:
    return SessionConfig(charte="Tout public. Jamais de nom de rue.")


@pytest.fixture
def make_video(tmp_path):
    """Écrit une vidéo mp4 blanche ; `draw(image, index)` peut dessiner sur chaque image."""

    def _make(name="clip.mp4", n=20, fps=10.0, size=(320, 240), draw=None):
        path = tmp_path / name
        w, h = size
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i in range(n):
            image = np.full((h, w, 3), 255, np.uint8)
            if draw is not None:
                draw(image, i)
            writer.write(image)
        writer.release()
        return path

    return _make
