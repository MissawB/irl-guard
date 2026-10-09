"""Choix des images à analyser et détection des images inchangées.

On compare des empreintes réduites (moyennes locales de CELLULE×CELLULE pixels) par différence
maximale : un petit texte qui apparaît modifie fortement quelques cellules, alors que le bruit
de compression s'annule dans les moyennes. La taille de cellule est fixe en pixels pour que
la sensibilité ne baisse pas quand la résolution augmente.
"""

from __future__ import annotations

import cv2
import numpy as np

CELLULE = 8


def empreinte(image: np.ndarray) -> np.ndarray:
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gris.shape[:2]
    taille = (max(1, w // CELLULE), max(1, h // CELLULE))
    return cv2.resize(gris, taille, interpolation=cv2.INTER_AREA).astype(np.int16)


def difference_max(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.abs(a - b).max())


class Sampler:
    def __init__(self, video_fps: float, fps_analyse: float, diff_max: int, max_reutilisations: int):
        self.pas = max(1, round(video_fps / fps_analyse))
        self.diff_max = diff_max
        self.max_reutilisations = max_reutilisations
        self._reference: np.ndarray | None = None
        self._reutilisations = 0

    def is_sample(self, index: int) -> bool:
        return index % self.pas == 0

    def is_unchanged(self, image: np.ndarray) -> bool:
        courante = empreinte(image)
        if (
            self._reference is not None
            and self._reutilisations < self.max_reutilisations
            and difference_max(courante, self._reference) <= self.diff_max
        ):
            self._reutilisations += 1
            return True
        self._reference = courante
        self._reutilisations = 0
        return False
