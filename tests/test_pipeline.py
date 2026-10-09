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
    assert len(lignes) == res.analyses == 16  # 15 échantillonnées + la dernière image
    assert json.loads(lignes[0])["analyse"]["pistes"][0]["action"] == "flouter"
    assert res.sortie_debug == chemin_debug(res.sortie) and res.sortie_debug.exists()
    assert res.debit > 0


def test_mode_domicile_met_en_pause(make_video, tmp_path):
    entree = make_video(n=20, fps=10.0, size=(640, 240), draw=_dessiner)
    cfg = SessionConfig(charte="Jamais de nom de rue.", fps_analyse=5.0, mode_domicile=True)
    res = traiter_video(entree, tmp_path / "out.mp4", cfg, TexteQuiBouge(), RulesDecider(cfg))
    sorties = _images(res.sortie)
    assert all(np.mean(img) < 60 for img in sorties)  # écran de pause sombre


class Rien:
    def detect(self, frame_index, image):
        return []


def test_derniere_image_toujours_analysee(make_video, tmp_path):
    entree = make_video(n=25, fps=10.0, size=(320, 240))
    cfg = SessionConfig(charte="x", fps_analyse=1.0)
    res = traiter_video(entree, tmp_path / "out.mp4", cfg, Rien(), RulesDecider(cfg))
    derniere = json.loads(res.journal.read_text(encoding="utf-8").splitlines()[-1])
    assert derniere["frame"] == 24


def _clip_vfr(tmp_path):
    import subprocess

    from irl_guard.media import ffmpeg_exe
    noir, blanc = tmp_path / "noir.png", tmp_path / "blanc.png"
    cv2.imwrite(str(noir), np.zeros((120, 160, 3), np.uint8))
    cv2.imwrite(str(blanc), np.full((120, 160, 3), 255, np.uint8))
    liste = tmp_path / "liste.txt"
    liste.write_text(f"file '{noir.as_posix()}'\nduration 0.2\nfile '{blanc.as_posix()}'\nduration 1.5\n"
                     f"file '{noir.as_posix()}'\nduration 0.2\nfile '{noir.as_posix()}'\n", encoding="utf-8")
    sortie = tmp_path / "vfr.mp4"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(liste),
                    "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(sortie)], check=True)
    return sortie


def test_frequence_variable_signalee(tmp_path, caplog):
    import logging
    cfg = SessionConfig(charte="x", fps_analyse=1.0)
    with caplog.at_level(logging.WARNING):
        traiter_video(_clip_vfr(tmp_path), tmp_path / "out.mp4", cfg, Rien(), RulesDecider(cfg))
    assert "fréquence d'images variable" in caplog.text


def test_dimensions_impaires(tmp_path):
    import subprocess

    from irl_guard.media import ffmpeg_exe
    entree = tmp_path / "impair.avi"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=duration=1:size=321x241:rate=10",
                    "-c:v", "mjpeg", "-pix_fmt", "yuvj444p", str(entree)], check=True)
    cfg = SessionConfig(charte="x", fps_analyse=5.0)
    res = traiter_video(entree, tmp_path / "out.mp4", cfg, Rien(), RulesDecider(cfg))
    assert len(_images(res.sortie)) == 10
