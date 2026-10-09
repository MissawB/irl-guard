"""Lecture du texte visible avec RapidOCR (modèles PaddleOCR en ONNX, CPU)."""

from __future__ import annotations

import numpy as np

from irl_guard.types import Box, Detection


class TextDetector:
    def __init__(self, conf_min: float):
        from rapidocr import RapidOCR
        self.moteur = RapidOCR(params={"Global.log_level": "critical"})
        self.conf_min = conf_min

    def detect(self, image: np.ndarray) -> list[Detection]:
        r = self.moteur(image)
        if r.boxes is None or r.txts is None:
            return []
        detections = []
        for quad, texte, score in zip(r.boxes, r.txts, r.scores):
            if score < self.conf_min or not texte.strip():
                continue
            xs, ys = quad[:, 0], quad[:, 1]
            boite = Box(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            detections.append(Detection("texte", float(score), boite, texte.strip()))
        return detections
