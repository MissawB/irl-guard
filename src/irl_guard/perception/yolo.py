"""Détecteurs Ultralytics YOLO : objets COCO, visages et plaques."""

from __future__ import annotations

import numpy as np

from irl_guard.config import PoidsHF
from irl_guard.types import Box, Detection

COCO_VERS_CLASSES = {
    "person": "personne", "knife": "couteau", "scissors": "ciseaux", "car": "voiture",
    "motorcycle": "moto", "bus": "bus", "truck": "camion", "cell phone": "telephone",
    "laptop": "ordinateur", "stop sign": "panneau_stop",
}


def poids_hf(p: PoidsHF) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(p.repo, p.fichier)


class YoloDetector:
    """`correspondance` traduit les noms du modèle ; la clé "*" s'applique à toutes les classes."""

    def __init__(self, poids: str, correspondance: dict[str, str], conf_min: float, device: str | None):
        from ultralytics import YOLO
        self.modele = YOLO(poids)
        self.correspondance = correspondance
        self.conf_min = conf_min
        self.device = device

    def detect(self, image: np.ndarray) -> list[Detection]:
        r = self.modele.predict(image, conf=self.conf_min, device=self.device, verbose=False)[0]
        detections = []
        for (x1, y1, x2, y2), conf, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            nom = r.names[int(k)]
            classe = self.correspondance.get(nom, self.correspondance.get("*"))
            if classe:
                detections.append(Detection(classe, float(conf), Box(x1, y1, x2, y2)))
        return detections
