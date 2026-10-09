"""Critère de réussite V1 n° 2 : aucune image de sortie ne laisse relire le texte sensible."""

import cv2
import pytest

from irl_guard.config import SessionConfig
from irl_guard.decision.rules import RulesDecider
from irl_guard.perception.base import Perception
from irl_guard.perception.text import TextDetector
from irl_guard.pipeline import traiter_video

APPARITION = 11  # entre deux analyses (pas de 2 images)


def _dessiner(image, i):
    if i >= APPARITION:
        x = 20 + 6 * (i - APPARITION)
        cv2.putText(image, "Rue Nationale", (x, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)


def _textes_lus(ocr, image):
    return " ".join(d.texte or "" for d in ocr.detect(image)).lower()


@pytest.mark.slow
def test_aucune_image_ne_laisse_lire_le_texte(make_video, tmp_path):
    entree = make_video(n=40, fps=10.0, size=(640, 240), draw=_dessiner)
    ocr = TextDetector(conf_min=0.0)

    cap = cv2.VideoCapture(str(entree))
    for _ in range(20):
        ok, image = cap.read()
    cap.release()
    assert "nation" in _textes_lus(ocr, image)  # le test sait détecter une fuite

    cfg = SessionConfig(charte="Tout public. Jamais de nom de rue.", fps_analyse=5.0)
    res = traiter_video(entree, tmp_path / "out.mp4", cfg, Perception([TextDetector(0.5)]), RulesDecider(cfg))

    cap = cv2.VideoCapture(str(res.sortie))
    fuites, n = [], 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        lus = _textes_lus(ocr, image)
        if "nation" in lus or "rue" in lus:
            fuites.append((n, lus))
        n += 1
    cap.release()
    assert n == 40
    assert fuites == []
