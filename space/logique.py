"""Logique du Space IRL Guard, sans Gradio (testable en local)."""

from __future__ import annotations

import logging
import sys

from irl_guard.config import SessionConfig
from irl_guard.decision.base import QUESTIONS, Decision
from irl_guard.decision.combine import CombinedDecider
from irl_guard.decision.laya import parser_reponse

log = logging.getLogger(__name__)
REPO_QWEN = "MissawB/irl-guard-decision-qwen"
REPO_LAYA = "MissawB/irl-guard-decision-laya"
CHARTE_DEFAUT = ("Live tout public. Floute les insultes. En mode domicile, coupe le live si on voit les adresses. "
                 "Sauf en live cuisine, masque les objets tranchants.")
EXEMPLES = [
    [CHARTE_DEFAUT, "IRL", True, "texte", "Rue Nationale", "couteau", "grande"],
    [CHARTE_DEFAUT, "IRL", False, "texte", "Rue Nationale", "couteau", "grande"],
    [CHARTE_DEFAUT, "Cuisine", False, "objet", "", "couteau", "moyenne"],
    [CHARTE_DEFAUT, "IRL", False, "objet", "", "couteau", "moyenne"],
    ["Ambiance détendue. Je n'ai aucun problème avec les gros mots.", "Gaming", False, "texte", "Sale connard",
     "couteau", "moyenne"],
]


def construire_state(charte: str, categorie: str, mode_domicile: bool, type_cible: str, texte: str,
                     classe: str, taille: str) -> dict:
    cible: dict = {"type": type_cible}
    if type_cible == "objet":
        cible["classe"] = classe
    elif texte:
        cible["texte"] = texte
    cible["conf"] = 0.9
    cible["taille"] = taille
    return {"categorie": categorie, "mode_domicile": bool(mode_domicile), "charte": charte, "cible": cible,
            "contexte": []}


def ligne(nom: str, decision: Decision) -> list[str]:
    p = decision.probabilites.get("action") or {}
    probas = " / ".join(f"{p.get(a, 0):.2f}" for a in ("diffuser", "flouter", "pause")) if p else "règle"
    return [nom, decision.action, probas, decision.risque, str(decision.gravite)]


def decider_tout(state: dict, regles, laya, qwen) -> list[list[str]]:
    lignes = [ligne("Règles", regles.decide(state)),
              ligne("Laya fine-tuné", laya.decide(state)),
              ligne("Règles + Laya", CombinedDecider(regles, laya).decide(state))]
    if qwen is None:
        lignes.append(["Qwen fine-tuné", "indisponible en ligne", "", "", ""])
    else:
        lignes += [ligne("Qwen fine-tuné", qwen.decide(state)),
                   ligne("Règles + Qwen", CombinedDecider(regles, qwen).decide(state))]
    return lignes


class DeciderQwen:
    """Qwen fine-tuné chargé avec `joint_schema_model.py` du dépôt (transformers seul, sans Unsloth)."""

    def __init__(self, cfg: SessionConfig, device: str = "cuda", repo: str = REPO_QWEN):
        from huggingface_hub import snapshot_download
        chemin = snapshot_download(repo)
        sys.path.insert(0, chemin)
        from joint_schema_model import load_release_model, systemone
        self._systemone = systemone
        self.model, self.processor = load_release_model(chemin, device=device)
        self.cfg = cfg

    def decide(self, state: dict) -> Decision:
        reponse = self._systemone(self.model, self.processor,
                                  {"model": "irl-guard-qwen", "state": state, "questions": QUESTIONS})
        decision = parser_reponse(reponse, self.cfg.seuils.pause, self.cfg.seuils.flou)
        return Decision(decision.action, decision.risque, decision.gravite, decision.probabilites, "qwen")


def charger_qwen(cfg: SessionConfig, device: str = "cuda") -> DeciderQwen | None:
    try:
        return DeciderQwen(cfg, device)
    except Exception as e:  # le Space reste utile sans Qwen
        log.warning("Qwen indisponible dans le Space : %s", e)
        return None
