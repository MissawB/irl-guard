"""Qwen fine-tuné chargé dans le processus avec le `joint_schema_model.py` de son dépôt (transformers seul)."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import replace

from irl_guard.config import SessionConfig
from irl_guard.decision.base import QUESTIONS, Decision, decision_repli
from irl_guard.decision.laya import LayaIndisponible, parser_reponse

log = logging.getLogger(__name__)
REPO_QWEN = "MissawB/irl-guard-decision-qwen"


def _charger_qwen(repo: str = REPO_QWEN, revision: str | None = None,
                  device: str | None = None) -> Callable[[dict], dict]:
    import torch
    from huggingface_hub import snapshot_download
    chemin = snapshot_download(repo, revision=revision)
    sys.path.insert(0, chemin)
    from joint_schema_model import load_release_model, systemone
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_release_model(chemin, device=device)
    return lambda requete: systemone(model, processor, requete)


class QwenLocalDecider:
    def __init__(self, cfg: SessionConfig, chargeur: Callable[[], Callable[[dict], dict]] | None = None,
                 repo: str = REPO_QWEN, revision: str | None = None):
        self.cfg, self.repo = cfg, repo
        self._chargeur = chargeur or (lambda: _charger_qwen(repo, revision=revision))
        self._systemone = None

    def decide(self, state: dict) -> Decision:
        try:
            if self._systemone is None:
                self._systemone = self._chargeur()
            reponse = self._systemone({"model": "irl-guard-qwen", "state": state, "questions": QUESTIONS})
            decision = parser_reponse(reponse, self.cfg.seuils.pause, self.cfg.seuils.flou)
            return replace(decision, source="qwen-local")
        except Exception as e:
            log.warning("Qwen local en erreur, piste floutée par défaut : %s", e)
            return decision_repli()

    def verifier(self) -> None:
        try:
            if self._systemone is None:
                self._systemone = self._chargeur()
            self._systemone({"model": "irl-guard-qwen", "state": {"cible": {"type": "texte", "texte": "test"}},
                             "questions": QUESTIONS})
        except Exception as e:
            raise LayaIndisponible(f"Modèle Qwen local « {self.repo} » inutilisable : {e}. "
                                   "Vérifie le GPU et la connexion au Hub, ou règle decideur.modele: laya-local, "
                                   "ou relance avec --decider rules.") from e
