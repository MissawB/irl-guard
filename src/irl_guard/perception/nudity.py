"""Détection de nudité avec NudeNet (onnxruntime CPU)."""

from __future__ import annotations

import numpy as np

from irl_guard.types import CLASSE_NUDITE, Box, Detection

CLASSES_EXPOSEES = frozenset({
    "FEMALE_GENITALIA_EXPOSED", "FEMALE_BREAST_EXPOSED", "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED", "ANUS_EXPOSED",
})


class NudityDetector:
    def __init__(self, conf_min: float):
        from nudenet import NudeDetector
        self.modele = NudeDetector()
        self.conf_min = conf_min

    def detect(self, image: np.ndarray) -> list[Detection]:
        detections = []
        for d in self.modele.detect(image):
            if d["class"] in CLASSES_EXPOSEES and d["score"] >= self.conf_min:
                x, y, w, h = d["box"]
                detections.append(Detection(CLASSE_NUDITE, float(d["score"]), Box(x, y, x + w, y + h)))
        return detections
