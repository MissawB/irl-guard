import json

import cv2
import numpy as np

from irl_guard.config import SessionConfig
from irl_guard.decision.rules import RulesDecider
from irl_guard.pipeline import chemin_debug, traiter_video
from irl_guard.types import Box, Detection

TEXTE = "Rue Nationale"


def _x(i):
    return 20 + 4 * i


def _boite(i):
    return Box(_x(i), 100, _x(i) + 260, 145)


def _dessiner(image, i):
    cv2.putText(image, TEXTE, (_x(i) + 5, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)


class TexteQuiBouge:
    """Perception factice : renvoie la boîte exacte du texte dessiné."""

    def detect(self, frame_index, image):
        return [Detection("texte", 0.95, _boite(frame_index), TEXTE)]


def _images(chemin):
    cap = cv2.VideoCapture(str(chemin))
    images = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        images.append(img)
    cap.release()
    return images


def _nettete(img, boite):
    x1, y1, x2, y2 = (int(v) for v in boite.to_tuple())
    return cv2.Laplacian(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def test_texte_masque_sur_toutes_les_images(make_video, tmp_path):
    entree = make_video(n=30, fps=10.0, size=(640, 240), draw=_dessiner)
    cfg = SessionConfig(charte="Jamais de nom de rue.", fps_analyse=5.0)
    res = traiter_video(entree, tmp_path / "out.mp4", cfg, TexteQuiBouge(), RulesDecider(cfg), debug=True)

    originales, sorties = _images(entree), _images(res.sortie)
    assert len(sorties) == 30 and res.images == 30
    for i, (orig, out) in enumerate(zip(originales, sorties)):
        assert _nettete(out, _boite(i)) < 0.3 * _nettete(orig, _boite(i)), f"image {i} lisible"

    lignes = res.journal.read_text(encoding="utf-8").splitlines()
    assert len(lignes) == res.analyses == 15
    assert json.loads(lignes[0])["analyse"]["pistes"][0]["action"] == "flouter"
    assert res.sortie_debug == chemin_debug(res.sortie) and res.sortie_debug.exists()
    assert res.debit > 0


def test_mode_domicile_met_en_pause(make_video, tmp_path):
    entree = make_video(n=20, fps=10.0, size=(640, 240), draw=_dessiner)
    cfg = SessionConfig(charte="Jamais de nom de rue.", fps_analyse=5.0, mode_domicile=True)
    res = traiter_video(entree, tmp_path / "out.mp4", cfg, TexteQuiBouge(), RulesDecider(cfg))
    sorties = _images(res.sortie)
    assert all(np.mean(img) < 60 for img in sorties)  # écran de pause sombre
