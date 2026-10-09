"""Types de base partagés par tout le pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASSES_REGLE = frozenset({"visage", "plaque"})
CLASSES_CONTEXTE = frozenset({"texte", "couteau", "ciseaux", "telephone", "ordinateur"})
CLASSE_NUDITE = "nudite"


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: Box) -> float:
        inter = Box(
            max(self.x1, other.x1), max(self.y1, other.y1),
            min(self.x2, other.x2), min(self.y2, other.y2),
        ).area
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def expand(self, ratio: float, width: int, height: int) -> Box:
        dw = (self.x2 - self.x1) * ratio / 2
        dh = (self.y2 - self.y1) * ratio / 2
        return Box(
            max(0.0, self.x1 - dw), max(0.0, self.y1 - dh),
            min(float(width), self.x2 + dw), min(float(height), self.y2 + dh),
        )

    def lerp(self, other: Box, t: float) -> Box:
        return Box(*(a + (b - a) * t for a, b in zip(self.to_tuple(), other.to_tuple())))

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_list(self) -> list[float]:
        return [round(v, 1) for v in self.to_tuple()]

    @classmethod
    def from_list(cls, values) -> Box:
        return cls(*(float(v) for v in values))


@dataclass(frozen=True)
class Detection:
    classe: str
    conf: float
    boite: Box
    texte: str | None = None

    def to_dict(self) -> dict:
        return {"classe": self.classe, "conf": round(self.conf, 3),
                "boite": self.boite.to_list(), "texte": self.texte}

    @classmethod
    def from_dict(cls, d: dict) -> Detection:
        return cls(d["classe"], float(d["conf"]), Box.from_list(d["boite"]), d.get("texte"))


@dataclass(frozen=True)
class Frame:
    index: int
    t: float
    image: np.ndarray
