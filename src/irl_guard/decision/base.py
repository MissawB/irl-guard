"""Types communs aux décideurs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

ACTIONS = ("diffuser", "flouter", "pause")
RISQUES = ("localisation", "donnee_perso", "arme", "propos", "aucun")
NIVEAUX_GRAVITE = ("aucune", "faible", "élevée", "critique")

QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "Que faire de cet élément dans le live ?",
        "criteria": {
            "diffuser": "rien de problématique",
            "flouter": "problème localisé et masquable",
            "pause": "problème grave ou impossible à masquer",
        },
    },
    "risque": {
        "type": "choice",
        "instructions": "Quel est le risque principal ?",
        "criteria": {
            "localisation": "nom de rue, numéro, enseigne, repère identifiable",
            "donnee_perso": "document, écran, nom, adresse, e-mail",
            "arme": "arme réelle ou objet menaçant",
            "propos": "texte injurieux ou haineux",
            "aucun": "aucun risque",
        },
    },
    "gravite": {
        "type": "score",
        "instructions": "Quelle est la gravité ?",
        "criteria": list(NIVEAUX_GRAVITE),
    },
}


@dataclass(frozen=True)
class Decision:
    action: str
    risque: str
    gravite: int
    probabilites: dict[str, dict[str, float]] = field(default_factory=dict)
    source: str = "rules"

    @property
    def p_pause(self) -> float:
        p = self.probabilites.get("action", {})
        if "pause" in p:
            return float(p["pause"])
        return 1.0 if self.action == "pause" else 0.0

    def to_dict(self) -> dict:
        return {"action": self.action, "risque": self.risque, "gravite": self.gravite,
                "probabilites": self.probabilites, "source": self.source}


def action_depuis_probas(p: dict[str, float], seuil_pause: float, seuil_flou: float) -> str:
    """Biais vers la protection : flouter dès que flouter + pause dépasse le seuil."""
    if p["pause"] >= seuil_pause:
        return "pause"
    if p["flouter"] + p["pause"] >= seuil_flou:
        return "flouter"
    return "diffuser"


def decision_repli() -> Decision:
    """Décision appliquée quand le décideur échoue : on masque."""
    return Decision("flouter", "inconnu", 2, {}, "fallback")


class Decider(Protocol):
    def decide(self, state: dict) -> Decision: ...
