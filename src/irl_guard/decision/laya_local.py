"""Laya (ou un Laya fine-tuné) chargé dans le processus avec le paquet `laya` : CPU, sans serveur."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace

from irl_guard.config import SessionConfig
from irl_guard.decision.base import QUESTIONS, Decision, decision_repli
from irl_guard.decision.laya import LayaIndisponible, parser_reponse

log = logging.getLogger(__name__)


def _charger_laya(modele: str):
    import laya
    return laya.load(modele)


class LayaLocalDecider:
    def __init__(self, cfg: SessionConfig, modele: str, chargeur: Callable[[str], object] | None = None):
        self.cfg, self.modele = cfg, modele
        self._chargeur = chargeur or _charger_laya
        self._agent = None

    def _agent_charge(self):
        if self._agent is None:
            self._agent = self._chargeur(self.modele)
        return self._agent

    def _appeler(self, state: dict) -> Decision:
        reponse = self._agent_charge().predict(state, QUESTIONS)
        decision = parser_reponse(reponse, self.cfg.seuils.pause, self.cfg.seuils.flou)
        return replace(decision, source="laya-local")

    def decide(self, state: dict) -> Decision:
        try:
            return self._appeler(state)
        except Exception as e:  # échec du côté sûr, quelle que soit l'erreur du modèle local
            log.warning("Laya local en erreur, piste floutée par défaut : %s", e)
            return decision_repli()

    def verifier(self) -> None:
        try:
            self._appeler({"cible": {"type": "texte", "texte": "test"}})
        except Exception as e:
            raise LayaIndisponible(f"Modèle Laya local « {self.modele} » inutilisable : {e}. "
                                   "Vérifie l'identifiant du modèle et la connexion au Hub, "
                                   "ou relance avec --decider rules.") from e
