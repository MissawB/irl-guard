"""Plan de rendu d'une image à partir des analyses qui l'encadrent (le tampon donne l'analyse suivante)."""

from __future__ import annotations

from dataclasses import dataclass

from irl_guard.analysis import Analyse
from irl_guard.types import Box


@dataclass(frozen=True)
class PlanImage:
    masques: tuple[Box, ...]
    pause: bool
    analyse: Analyse | None = None


def _boites_masquees(analyse: Analyse | None) -> dict[int, Box]:
    if analyse is None:
        return {}
    return {p.id: p.boite for p in analyse.pistes if p.masquer}


def planifier(frame_index: int, avant: Analyse | None, apres: Analyse | None,
              marge: float, largeur: int, hauteur: int) -> PlanImage:
    if avant is not None and avant.frame_index == frame_index:
        apres = None
    if avant is None and apres is None:
        return PlanImage((), False, None)

    a, b = _boites_masquees(avant), _boites_masquees(apres)
    t = 0.0
    if avant is not None and apres is not None:
        t = (frame_index - avant.frame_index) / (apres.frame_index - avant.frame_index)

    masques = []
    for pid in sorted(a.keys() | b.keys()):
        if pid in a and pid in b:
            boite = a[pid].lerp(b[pid], t)
        else:
            boite = a[pid] if pid in a else b[pid]
        masques.append(boite.expand(marge, largeur, hauteur))

    pause = bool((avant is not None and avant.en_pause) or (apres is not None and apres.en_pause))
    return PlanImage(tuple(masques), pause, avant if avant is not None else apres)
