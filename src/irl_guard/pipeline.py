"""Pipeline hors ligne en deux passes : analyse des images échantillonnées, puis rendu de toutes les images."""

from __future__ import annotations

import bisect
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2

from irl_guard.analysis import Analyse, Analyzer
from irl_guard.config import SessionConfig
from irl_guard.decision.base import Decider
from irl_guard.journal import Journal
from irl_guard.media import finaliser, intervalles_pause
from irl_guard.planning import planifier
from irl_guard.render import ecran_pause, rendre
from irl_guard.sampling import Sampler
from irl_guard.sources import VideoSource


@dataclass(frozen=True)
class Resultat:
    sortie: Path
    sortie_debug: Path | None
    journal: Path
    images: int
    analyses: int
    duree_video_s: float
    duree_traitement_s: float

    @property
    def debit(self) -> float:
        """Multiple du temps réel : 2.0 = deux fois plus rapide que la lecture."""
        if self.duree_traitement_s <= 0:
            return float("inf")
        return self.duree_video_s / self.duree_traitement_s


def analyser_video(source: VideoSource, perception, analyzer: Analyzer, sampler: Sampler,
                   journal: Journal) -> list[Analyse]:
    analyses: list[Analyse] = []
    derniere, detections_derniere = None, []
    for frame in source:
        if not sampler.is_sample(frame.index):
            continue
        inchangee = sampler.is_unchanged(frame.image)
        if inchangee and derniere is not None:
            analyse = replace(derniere, frame_index=frame.index, appels=(), reutilisee=True)
            detections = detections_derniere
        else:
            detections = perception.detect(frame.index, frame.image)
            analyse = analyzer.analyser(frame.index, detections)
        journal.ecrire(analyse, detections)
        analyses.append(analyse)
        derniere, detections_derniere = analyse, detections
    return analyses


def rendre_video(source: VideoSource, analyses: list[Analyse], cfg: SessionConfig,
                 sorties: list[tuple[Path, bool]]) -> list[bool]:
    indices = [a.frame_index for a in analyses]
    ecran = ecran_pause(source.width, source.height, cfg.ecran_pause)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = [(cv2.VideoWriter(str(p), fourcc, source.fps, (source.width, source.height)), debug)
               for p, debug in sorties]
    pauses: list[bool] = []
    try:
        for frame in source:
            k = bisect.bisect_right(indices, frame.index)
            avant = analyses[k - 1] if k > 0 else None
            apres = analyses[k] if k < len(analyses) else None
            plan = planifier(frame.index, avant, apres, cfg.marge, source.width, source.height)
            pauses.append(plan.pause)
            for writer, debug in writers:
                writer.write(rendre(frame.image, plan, cfg.masquage, ecran, debug))
    finally:
        for writer, _ in writers:
            writer.release()
    return pauses


def chemin_debug(sortie: Path) -> Path:
    return sortie.with_name(f"{sortie.stem}_debug{sortie.suffix}")


def traiter_video(entree: str | Path, sortie: str | Path, cfg: SessionConfig, perception, decider: Decider,
                  debug: bool = False, journal: str | Path | None = None) -> Resultat:
    entree, sortie = Path(entree), Path(sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    chemin_journal = Path(journal) if journal else sortie.with_suffix(".jsonl")
    source = VideoSource(entree)
    sampler = Sampler(source.fps, cfg.fps_analyse, cfg.seuils.diff_max, cfg.seuils.max_reutilisations)
    analyzer = Analyzer(cfg, decider, source.width, source.height)

    debut = time.perf_counter()
    with Journal(chemin_journal) as j:
        analyses = analyser_video(source, perception, analyzer, sampler, j)

    sortie_debug = chemin_debug(sortie) if debug else None
    with tempfile.TemporaryDirectory() as tmp:
        muettes = [(Path(tmp) / "protegee.mp4", False)]
        if debug:
            muettes.append((Path(tmp) / "debug.mp4", True))
        pauses = rendre_video(source, analyses, cfg, muettes)
        intervalles = intervalles_pause(pauses, source.fps)
        finaliser(muettes[0][0], entree, sortie, intervalles, cfg.couper_audio_en_pause)
        if debug:
            finaliser(muettes[1][0], entree, sortie_debug, intervalles, cfg.couper_audio_en_pause)

    return Resultat(sortie, sortie_debug, chemin_journal, len(pauses), len(analyses),
                    len(pauses) / source.fps, time.perf_counter() - debut)
