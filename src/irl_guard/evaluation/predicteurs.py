"""Prédicteurs communs : décideur du pipeline (règles, Laya) et fichiers exportés par Colab."""

from __future__ import annotations

import json
from pathlib import Path

from irl_guard.decision.base import ACTIONS, RISQUES, Decider, Decision


def _one_hot(options: tuple[str, ...], valeur: str) -> dict[str, float]:
    if valeur not in options:
        return {o: 1.0 / len(options) for o in options}
    return {o: 1.0 if o == valeur else 0.0 for o in options}


def reponses_depuis_decision(decision: Decision) -> dict:
    return {
        "action": {"probabilities": decision.probabilites.get("action") or _one_hot(ACTIONS, decision.action)},
        "risque": {"probabilities": decision.probabilites.get("risque") or _one_hot(RISQUES, decision.risque)},
        "gravite": {"score": float(decision.gravite)},
    }


def predire_avec_decider(decider: Decider, lignes: list[dict]) -> dict[str, dict]:
    return {l["id"]: reponses_depuis_decision(decider.decide(json.loads(l["state"]))) for l in lignes}


def charger_predictions(chemin: str | Path) -> dict[str, dict]:
    with Path(chemin).open(encoding="utf-8") as f:
        return {d["id"]: d["answers"] for d in (json.loads(l) for l in f if l.strip())}
