"""Client de la Decision API d'Unsloth (Laya ou modèle fine-tuné servi par Unsloth Studio)."""

from __future__ import annotations

import logging

import httpx

from irl_guard.config import SessionConfig
from irl_guard.decision.base import (
    ACTIONS, NIVEAUX_GRAVITE, QUESTIONS, RISQUES, Decision, action_depuis_probas, decision_repli,
)

log = logging.getLogger(__name__)
ERREURS_REPONSE = (KeyError, ValueError, TypeError)


class LayaIndisponible(RuntimeError):
    pass


def _probas(reponse: dict, options: tuple[str, ...]) -> dict[str, float]:
    brutes = reponse["probabilities"]
    probas = {o: float(brutes[o]) for o in options}
    if any(not 0.0 <= v <= 1.0 for v in probas.values()):
        raise ValueError(f"probabilité hors de [0, 1] : {probas}")
    return probas


def _gravite(reponse: dict) -> int:
    score = reponse["score"]
    if isinstance(score, str) and score in NIVEAUX_GRAVITE:
        return NIVEAUX_GRAVITE.index(score)
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return min(max(int(score), 0), len(NIVEAUX_GRAVITE) - 1)
    raise ValueError(f"score de gravité inattendu : {score!r}")


def parser_reponse(data: dict, seuil_pause: float, seuil_flou: float) -> Decision:
    reponses = data["answers"]
    p_action = _probas(reponses["action"], ACTIONS)
    p_risque = _probas(reponses["risque"], RISQUES)
    return Decision(
        action=action_depuis_probas(p_action, seuil_pause, seuil_flou),
        risque=max(p_risque, key=p_risque.get),
        gravite=_gravite(reponses["gravite"]),
        probabilites={"action": p_action, "risque": p_risque},
        source="laya",
    )


class LayaDecider:
    def __init__(self, cfg: SessionConfig, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        laya = cfg.laya
        en_tetes = {"Authorization": f"Bearer {laya.cle}"} if laya.cle else {}
        self._client = httpx.Client(base_url=laya.url, timeout=laya.timeout_s,
                                    headers=en_tetes, transport=transport)

    def _appeler(self, state: dict) -> Decision:
        corps = {"model": self.cfg.laya.modele, "state": state, "questions": QUESTIONS}
        reponse = self._client.post("/v1/systemone", json=corps)
        reponse.raise_for_status()
        return parser_reponse(reponse.json(), self.cfg.seuils.pause, self.cfg.seuils.flou)

    def decide(self, state: dict) -> Decision:
        try:
            return self._appeler(state)
        except (httpx.HTTPError, *ERREURS_REPONSE) as e:
            log.warning("Décision Laya impossible, piste floutée par défaut : %s", e)
            return decision_repli()

    def verifier(self) -> None:
        try:
            self._appeler({"cible": {"type": "texte", "texte": "test"}})
        except (httpx.HTTPError, *ERREURS_REPONSE) as e:
            raise LayaIndisponible(
                f"Laya injoignable ou réponse invalide sur {self.cfg.laya.url} : {e}. "
                "Lance l'app Unsloth et active Settings → API → Decision API, "
                "ou relance avec --decider rules."
            ) from e
