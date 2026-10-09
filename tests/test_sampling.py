import cv2
import numpy as np

from irl_guard.sampling import Sampler


def _image(valeur=128, w=1280, h=720):
    return np.full((h, w, 3), valeur, np.uint8)


def test_pas_d_echantillonnage():
    s = Sampler(video_fps=30, fps_analyse=3, diff_max=6, max_reutilisations=2)
    assert s.pas == 10
    assert [i for i in range(35) if s.is_sample(i)] == [0, 10, 20, 30]


def test_fps_analyse_superieur_au_fps_video():
    assert Sampler(video_fps=10, fps_analyse=30, diff_max=6, max_reutilisations=2).pas == 1


def test_image_identique_puis_differente():
    s = Sampler(30, 3, 6, 5)
    assert s.is_unchanged(_image()) is False  # première image : référence
    assert s.is_unchanged(_image()) is True
    assert s.is_unchanged(_image(200)) is False


def test_bruit_leger_reste_inchange():
    s = Sampler(30, 3, 6, 5)
    s.is_unchanged(_image())
    rng = np.random.default_rng(0)
    bruit = np.clip(_image().astype(np.int16) + rng.normal(0, 3, (720, 1280, 3)), 0, 255).astype(np.uint8)
    assert s.is_unchanged(bruit) is True


def test_petit_texte_rend_l_image_differente():
    s = Sampler(30, 3, 6, 5)
    s.is_unchanged(_image(255))
    avec_texte = _image(255)
    cv2.putText(avec_texte, "12", (600, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    assert s.is_unchanged(avec_texte) is False


def test_reanalyse_forcee_apres_max_reutilisations():
    s = Sampler(30, 3, 6, max_reutilisations=2)
    resultats = [s.is_unchanged(_image()) for _ in range(7)]
    assert resultats == [False, True, True, False, True, True, False]


def test_petit_texte_peu_contraste_en_4k():
    s = Sampler(30, 3, 6, 5)
    s.is_unchanged(_image(80, w=3840, h=2160))
    avec_texte = _image(80, w=3840, h=2160)
    cv2.putText(avec_texte, "06 12 34 56 78", (1800, 1080), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    assert s.is_unchanged(avec_texte) is False
