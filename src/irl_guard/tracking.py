"""Suivi des détections d'une analyse à l'autre et accumulation de l'OCR par piste."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from irl_guard.types import Box, Detection

TEXTE_MAX = 200
IOU_ADOPTION = 0.3


def fusionner_texte(cumul: str, nouveau: str, max_len: int = TEXTE_MAX) -> str:
    nouveau = nouveau.strip()
    if not cumul:
        return nouveau[:max_len]
    if not nouveau or nouveau.lower() in cumul.lower():
        return cumul
    if cumul.lower() in nouveau.lower():
        return nouveau[:max_len]
    return f"{cumul} | {nouveau}"[-max_len:]


@dataclass
class Piste:
    id: int
    classe: str
    boite: Box
    conf: float
    texte: str | None = None
    nouveau: bool = True
    modifie: bool = True
    perdu: bool = False
    analyses_sans_detection: int = 0


def _vers_sv(detections: list[Detection]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([d.boite.to_tuple() for d in detections], dtype=float),
        confidence=np.array([d.conf for d in detections], dtype=float),
        class_id=np.zeros(len(detections), dtype=int),
        data={"idx": np.arange(len(detections))},
    )


class TrackManager:
    """Un ByteTrack par classe, ids globaux stables, pistes perdues gardées `maintien` analyses."""

    def __init__(self, maintien: int):
        self.maintien = maintien
        self._trackers: dict[str, ByteTrackTracker] = {}
        self._ids: dict[tuple[str, int], int] = {}
        self._pistes: dict[int, Piste] = {}
        self._ephemeres: dict[str, list[int]] = defaultdict(list)
        self._prochain_id = 1

    def update(self, detections: list[Detection]) -> list[Piste]:
        par_classe: dict[str, list[Detection]] = defaultdict(list)
        for d in detections:
            par_classe[d.classe].append(d)

        vus: set[int] = set()
        ephemeres: dict[str, list[int]] = defaultdict(list)
        for classe in sorted(set(par_classe) | set(self._trackers)):
            dets = par_classe.get(classe, [])
            tracker = self._tracker(classe)
            if not dets:
                tracker.update(sv.Detections.empty())
                continue
            sortie = tracker.update(_vers_sv(dets))
            paires = list(zip(sortie.tracker_id.tolist(), sortie.data["idx"].tolist()))
            rendus = {idx for _, idx in paires}
            # Une détection non renvoyée par ByteTrack doit quand même être masquée.
            paires += [(-1, i) for i in range(len(dets)) if i not in rendus]
            for tracker_id, idx in paires:
                det = dets[idx]
                if tracker_id < 0:
                    pid = self._adopter(classe, det.boite, vus) or self._nouvel_id()
                    ephemeres[classe].append(pid)
                else:
                    cle = (classe, tracker_id)
                    if cle not in self._ids:
                        self._ids[cle] = self._adopter(classe, det.boite, vus) or self._nouvel_id()
                    pid = self._ids[cle]
                self._mettre_a_jour(pid, det)
                vus.add(pid)

        self._ephemeres = ephemeres
        self._vieillir(vus)
        return [replace(p) for p in self._pistes.values()]

    def _tracker(self, classe: str) -> ByteTrackTracker:
        if classe not in self._trackers:
            # frame_rate=30 : lost_track_buffer s'exprime alors directement en nombre d'analyses.
            self._trackers[classe] = ByteTrackTracker(
                lost_track_buffer=max(1, self.maintien),
                frame_rate=30.0,
                track_activation_threshold=0.25,
                minimum_consecutive_frames=1,
                high_conf_det_threshold=0.25,
            )
        return self._trackers[classe]

    def _nouvel_id(self) -> int:
        pid = self._prochain_id
        self._prochain_id += 1
        return pid

    def _adopter(self, classe: str, boite: Box, deja_pris: set[int]) -> int | None:
        """Reprend l'id d'une piste éphémère de l'analyse précédente située au même endroit."""
        meilleur, meilleur_iou = None, IOU_ADOPTION
        for pid in self._ephemeres[classe]:
            piste = self._pistes.get(pid)
            if piste is None or pid in deja_pris:
                continue
            iou = piste.boite.iou(boite)
            if iou >= meilleur_iou:
                meilleur, meilleur_iou = pid, iou
        return meilleur

    def _mettre_a_jour(self, pid: int, det: Detection) -> None:
        texte = det.texte.strip() if det.texte else None
        piste = self._pistes.get(pid)
        if piste is None:
            self._pistes[pid] = Piste(pid, det.classe, det.boite, det.conf, texte=texte or None)
            return
        ancien = piste.texte
        if texte:
            piste.texte = fusionner_texte(piste.texte or "", texte)
        piste.boite, piste.conf = det.boite, det.conf
        piste.nouveau, piste.modifie = False, piste.texte != ancien
        piste.perdu, piste.analyses_sans_detection = False, 0

    def _vieillir(self, vus: set[int]) -> None:
        for pid in list(self._pistes):
            if pid in vus:
                continue
            piste = self._pistes[pid]
            piste.analyses_sans_detection += 1
            piste.perdu, piste.nouveau, piste.modifie = True, False, False
            if piste.analyses_sans_detection > self.maintien:
                del self._pistes[pid]
