"""Journal JSONL : une ligne par image analysée. Il sert au débogage et au rejeu sans perception."""

from __future__ import annotations

import json
from pathlib import Path

from irl_guard.analysis import Analyse
from irl_guard.types import Detection


class Journal:
    def __init__(self, chemin: str | Path):
        self.chemin = Path(chemin)
        self._fichier = None

    def __enter__(self) -> Journal:
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        self._fichier = self.chemin.open("w", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        self._fichier.close()

    def ecrire(self, analyse: Analyse, detections: list[Detection]) -> None:
        ligne = {"frame": analyse.frame_index,
                 "detections": [d.to_dict() for d in detections],
                 "analyse": analyse.to_dict()}
        self._fichier.write(json.dumps(ligne, ensure_ascii=False) + "\n")


def lire_detections(chemin: str | Path) -> dict[int, list[Detection]]:
    resultat: dict[int, list[Detection]] = {}
    with Path(chemin).open(encoding="utf-8") as f:
        for ligne in f:
            if ligne.strip():
                d = json.loads(ligne)
                resultat[int(d["frame"])] = [Detection.from_dict(x) for x in d["detections"]]
    return resultat
