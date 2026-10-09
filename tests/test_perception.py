import cv2
import numpy as np
import pytest

from irl_guard.config import SessionConfig
from irl_guard.perception import factory
from irl_guard.perception.base import (
    CachedPerception, DetecteurIndisponible, JournalIncompatible, Perception,
)
from irl_guard.types import Box, Detection


class Fixe:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, image):
        return list(self.dets)


def test_perception_concatene():
    a = Detection("visage", 0.9, Box(0, 0, 1, 1))
    b = Detection("texte", 0.9, Box(0, 0, 1, 1), "x")
    assert Perception([Fixe([a]), Fixe([b])]).detect(0, None) == [a, b]


def test_cache_et_journal_incompatible():
    d = Detection("texte", 0.9, Box(0, 0, 1, 1), "x")
    cache = CachedPerception({0: [d]})
    assert cache.detect(0, None) == [d]
    with pytest.raises(JournalIncompatible, match="fps_analyse"):
        cache.detect(3, None)


def test_echec_de_chargement_nomme_le_detecteur(monkeypatch):
    def poids_en_echec(p):
        raise OSError("réseau coupé")

    monkeypatch.setattr(factory, "YoloDetector", lambda *a, **k: object())
    monkeypatch.setattr(factory, "poids_hf", poids_en_echec)
    with pytest.raises(DetecteurIndisponible, match="visages"):
        factory.construire_perception(SessionConfig(charte="x"))


@pytest.mark.gpu
def test_yolo_objets_et_visages_sur_bus():
    from ultralytics.utils import ASSETS

    from irl_guard.perception.yolo import COCO_VERS_CLASSES, YoloDetector, poids_hf
    image = cv2.imread(str(ASSETS / "bus.jpg"))
    cfg = SessionConfig(charte="x")
    objets = YoloDetector(cfg.detecteurs.objets, COCO_VERS_CLASSES, 0.3, None).detect(image)
    assert {"personne", "bus"} <= {d.classe for d in objets}
    visages = YoloDetector(poids_hf(cfg.detecteurs.visages), {"*": "visage"}, 0.3, None).detect(image)
    assert visages and all(d.classe == "visage" for d in visages)
    plaques = YoloDetector(poids_hf(cfg.detecteurs.plaques), {"*": "plaque"}, 0.3, None)
    assert isinstance(plaques.detect(image), list)


@pytest.mark.gpu
def test_nudite_sur_image_vide():
    from irl_guard.perception.nudity import NudityDetector
    assert NudityDetector(0.3).detect(np.zeros((320, 320, 3), np.uint8)) == []


@pytest.mark.slow
def test_ocr_lit_un_texte():
    from irl_guard.perception.text import TextDetector
    image = np.full((200, 600, 3), 255, np.uint8)
    cv2.putText(image, "Rue Nationale", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    dets = TextDetector(0.5).detect(image)
    assert len(dets) == 1 and dets[0].texte == "Rue Nationale" and dets[0].classe == "texte"
    assert dets[0].boite.x1 < 40 and dets[0].boite.x2 > 400
    assert TextDetector(0.5).detect(np.full((100, 100, 3), 255, np.uint8)) == []
