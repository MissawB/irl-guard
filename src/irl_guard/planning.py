"""Plan de rendu d'une image à partir des analyses qui l'encadrent (le tampon donne l'analyse suivante)."""

from __future__ import annotations

from dataclasses import dataclass

from irl_guard.analysis import Analyse, PisteAnalysee
from irl_guard.types import Box


@dataclass(frozen=True)
class PlanImage:
    masques: tuple[Box, ...]
    pause: bool
    analyse: Analyse | None = None


def _pistes_masquees(analyse: Analyse | None) -> dict[int, PisteAnalysee]:
    if analyse is None:
        return {}
    return {p.id: p for p in analyse.pistes if p.masquer}


def _centre(boite: Box) -> tuple[float, float]:
    return ((boite.x1 + boite.x2) / 2, (boite.y1 + boite.y2) / 2)


def _ponts(a: dict[int, PisteAnalysee], b: dict[int, PisteAnalysee], t: float) -> list[Box]:
    """Quand ByteTrack change d'id sur un mouvement rapide, relie l'ancienne piste à la nouvelle.

    Chaque piste apparue dans `b` est reliée à la piste de même classe la plus proche
    qui a disparu ou est perdue dans `b`. Le masque interpolé couvre le trajet.
    """
    orphelines = [p for pid, p in a.items() if pid not in b or b[pid].perdu]
    ponts = []
    for pid, nouvelle in b.items():
        if pid in a:
            continue
        candidates = [p for p in orphelines if p.classe == nouvelle.classe]
        if not candidates:
            continue
        cx, cy = _centre(nouvelle.boite)
        origine = min(candidates, key=lambda p: (_centre(p.boite)[0] - cx) ** 2 + (_centre(p.boite)[1] - cy) ** 2)
        ponts.append(origine.boite.lerp(nouvelle.boite, t))
    return ponts


def planifier(frame_index: int, avant: Analyse | None, apres: Analyse | None,
              marge: float, largeur: int, hauteur: int) -> PlanImage:
    if avant is not None and avant.frame_index == frame_index:
        apres = None
    if avant is None and apres is None:
        return PlanImage((), False, None)

    a, b = _pistes_masquees(avant), _pistes_masquees(apres)
    t = 0.0
    if avant is not None and apres is not None:
        t = (frame_index - avant.frame_index) / (apres.frame_index - avant.frame_index)

    boites = []
    for pid in sorted(a.keys() | b.keys()):
        if pid in a and pid in b:
            boites.append(a[pid].boite.lerp(b[pid].boite, t))
        else:
            boites.append(a[pid].boite if pid in a else b[pid].boite)
    boites += _ponts(a, b, t)
    masques = [boite.expand(marge, largeur, hauteur) for boite in boites]

    pause = bool((avant is not None and avant.en_pause) or (apres is not None and apres.en_pause))
    return PlanImage(tuple(masques), pause, avant if avant is not None else apres)
