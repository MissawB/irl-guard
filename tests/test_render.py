import cv2
import numpy as np

from irl_guard.analysis import Analyse, PisteAnalysee
from irl_guard.planning import PlanImage
from irl_guard.render import ecran_pause, masquer, rendre
from irl_guard.types import Box


def _image_texte():
    img = np.full((240, 640, 3), 255, np.uint8)
    cv2.putText(img, "Rue Nationale", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


def _nettete(img, boite):
    x1, y1, x2, y2 = map(int, boite.to_tuple())
    gris = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gris, cv2.CV_64F).var()


def test_pixellisation_detruit_le_texte_et_ne_touche_pas_le_reste():
    img = _image_texte()
    origine = img.copy()
    boite = Box(10, 100, 300, 145)
    masquer(img, boite, "pixel")
    assert _nettete(img, boite) < 0.3 * _nettete(origine, boite)
    assert np.array_equal(img[160:, :], origine[160:, :])


def test_flou_detruit_le_texte():
    img = _image_texte()
    origine = img.copy()
    boite = Box(10, 100, 300, 145)
    masquer(img, boite, "flou")
    assert _nettete(img, boite) < 0.3 * _nettete(origine, boite)


def test_boite_hors_image_ne_plante_pas():
    img = _image_texte()
    origine = img.copy()
    masquer(img, Box(-50, -50, -10, -10), "pixel")
    masquer(img, Box(700, 300, 800, 400), "flou")
    masquer(img, Box(100, 100, 100, 100), "pixel")
    assert np.array_equal(img, origine)
    masquer(img, Box(600, 200, 900, 400), "pixel")  # déborde en partie


def test_rendre_pause_et_debug():
    img = _image_texte()
    ecran = ecran_pause(640, 240)
    assert ecran.shape == (240, 640, 3)
    sortie = rendre(img, PlanImage((), True), "pixel", ecran)
    assert np.array_equal(sortie, ecran)
    analyse = Analyse(0, (PisteAnalysee(1, "texte", Box(10, 100, 300, 145), "Rue", "flouter", False),), 0.0, False)
    plan = PlanImage((Box(10, 100, 300, 145),), False, analyse)
    sans_debug = rendre(img, plan, "pixel", ecran)
    avec_debug = rendre(img, plan, "pixel", ecran, debug=True)
    assert not np.array_equal(sans_debug, avec_debug)
    assert np.array_equal(img, _image_texte())  # l'image d'entrée n'est pas modifiée
