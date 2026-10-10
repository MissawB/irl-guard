"""Analyse d'une image échantillonnée : suivi, décisions par piste, score et hystérésis."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from irl_guard.config import SessionConfig
from irl_guard.decision.base import Decider, Decision, decision_repli
from irl_guard.state import construire_state
from irl_guard.temporal import Hysteresis
from irl_guard.tracking import Piste, TrackManager
from irl_guard.types import CLASSE_NUDITE, CLASSES_CONTEXTE, CLASSES_REGLE, Box, Detection

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PisteAnalysee:
    id: int
    classe: str
    boite: Box
    texte: str | None
    action: str
    perdu: bool
    decision: Decision | None = None

    @property
    def masquer(self) -> bool:
        return self.action in ("flouter", "pause")

    def to_dict(self) -> dict:
        return {"id": self.id, "classe": self.classe, "boite": self.boite.to_list(), "texte": self.texte,
                "action": self.action, "perdu": self.perdu,
                "decision": self.decision.to_dict() if self.decision else None}


@dataclass(frozen=True)
class Analyse:
    frame_index: int
    pistes: tuple[PisteAnalysee, ...]
    score: float
    en_pause: bool
    appels: tuple[dict, ...] = ()
    reutilisee: bool = False

    def to_dict(self) -> dict:
        return {"frame": self.frame_index, "score": round(self.score, 4), "en_pause": self.en_pause,
                "reutilisee": self.reutilisee, "pistes": [p.to_dict() for p in self.pistes],
                "appels": list(self.appels)}


class Analyzer:
    def __init__(self, cfg: SessionConfig, decider: Decider, largeur: int, hauteur: int):
        self.cfg, self.decider = cfg, decider
        self.largeur, self.hauteur = largeur, hauteur
        self.pistes = TrackManager(maintien=cfg.maintien)
        self.hysteresis = Hysteresis(cfg.seuils.hysteresis_haut, cfg.seuils.hysteresis_bas, cfg.seuils.sortie_k)
        self._cache: dict[tuple[int, str | None], Decision] = {}

    def analyser(self, frame_index: int, detections: list[Detection]) -> Analyse:
        pistes = self.pistes.update(detections)
        score = 0.0
        appels: list[dict] = []
        resultats: list[PisteAnalysee] = []
        for p in pistes:
            decision = None
            if p.classe == CLASSE_NUDITE:
                action = "flouter"
                if not p.perdu and p.conf >= self.cfg.seuils.nudite:
                    score = 1.0
            elif p.classe in CLASSES_REGLE:
                action = "flouter"
            elif p.classe in CLASSES_CONTEXTE:
                decision = self._decision(p, pistes, appels)
                action = decision.action
                score = max(score, decision.p_pause)
            else:
                action = "diffuser"
            resultats.append(PisteAnalysee(p.id, p.classe, p.boite, p.texte, action, p.perdu, decision))
        en_pause = self.hysteresis.update(score)
        return Analyse(frame_index, tuple(resultats), score, en_pause, tuple(appels))

    def _decision(self, piste: Piste, pistes: list[Piste], appels: list[dict]) -> Decision:
        cle = (piste.id, piste.texte)
        if cle in self._cache:
            return self._cache[cle]
        state = construire_state(piste, pistes, self.cfg, self.largeur, self.hauteur)
        try:
            decision = self.decider.decide(state)
        except Exception as e:  # échec du côté sûr, quel que soit le décideur
            log.warning("Décideur en erreur, piste floutée par défaut : %s", e)
            decision = decision_repli()
        appels.append({"piste": piste.id, "state": state, "decision": decision.to_dict()})
        if "fallback" not in decision.source:  # repli seul ou dans le mode combiné : réessayer
            self._cache[cle] = decision
        return decision
