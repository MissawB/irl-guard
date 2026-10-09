"""Construction de la perception complète à partir de la configuration."""

from __future__ import annotations

from collections.abc import Callable

from irl_guard.config import SessionConfig
from irl_guard.perception.base import DetecteurIndisponible, Detector, Perception
from irl_guard.perception.nudity import NudityDetector
from irl_guard.perception.text import TextDetector
from irl_guard.perception.yolo import COCO_VERS_CLASSES, YoloDetector, poids_hf


def _charger(nom: str, fabrique: Callable[[], Detector]) -> Detector:
    try:
        return fabrique()
    except Exception as e:  # le message doit nommer le modèle en cause
        raise DetecteurIndisponible(f"Impossible de charger le détecteur « {nom} » : {e}") from e


def construire_perception(cfg: SessionConfig) -> Perception:
    d = cfg.detecteurs
    return Perception([
        _charger(f"objets ({d.objets})",
                 lambda: YoloDetector(d.objets, COCO_VERS_CLASSES, d.conf_min, d.device)),
        _charger(f"visages ({d.visages.repo})",
                 lambda: YoloDetector(poids_hf(d.visages), {"*": "visage"}, d.conf_min, d.device)),
        _charger(f"plaques ({d.plaques.repo})",
                 lambda: YoloDetector(poids_hf(d.plaques), {"*": "plaque"}, d.conf_min, d.device)),
        _charger("nudité (NudeNet)", lambda: NudityDetector(d.conf_min)),
        _charger("texte (RapidOCR)", lambda: TextDetector(d.ocr_conf_min)),
    ])
