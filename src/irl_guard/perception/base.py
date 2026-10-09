"""Interfaces de perception, composition de détecteurs et rejeu depuis le journal."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from irl_guard.types import Detection


class DetecteurIndisponible(RuntimeError):
    pass


class JournalIncompatible(RuntimeError):
    pass


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


class Perception:
    def __init__(self, detecteurs: list[Detector]):
        self.detecteurs = detecteurs

    def detect(self, frame_index: int, image: np.ndarray) -> list[Detection]:
        return [d for detecteur in self.detecteurs for d in detecteur.detect(image)]


class CachedPerception:
    def __init__(self, cache: dict[int, list[Detection]]):
        self.cache = cache

    def detect(self, frame_index: int, image: np.ndarray) -> list[Detection]:
        if frame_index not in self.cache:
            raise JournalIncompatible(
                f"L'image {frame_index} n'est pas dans le journal : "
                "rejoue avec le même fps_analyse que l'analyse d'origine.")
        return self.cache[frame_index]
