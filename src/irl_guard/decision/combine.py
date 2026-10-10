"""Mode combiné : règles et modèle, la décision la plus protectrice l'emporte."""

from __future__ import annotations

from dataclasses import replace

from irl_guard.decision.base import ACTIONS, Decider, Decision


def plus_protectrice(regle: str, modele: str) -> str:
    """Nom du gagnant ; à égalité, le modèle (il porte des probabilités)."""
    return "modele" if ACTIONS.index(modele) >= ACTIONS.index(regle) else "regle"


class CombinedDecider:
    def __init__(self, regles: Decider, modele: Decider):
        self.regles, self.modele = regles, modele

    def decide(self, state: dict) -> Decision:
        regle = self.regles.decide(state)
        modele = self.modele.decide(state)
        if modele.source == "fallback":
            # Le repli vaut « flouter » : il ne peut que relever une règle qui diffuserait.
            gagnant = modele if regle.action == "diffuser" else regle
        else:
            gagnant = modele if plus_protectrice(regle.action, modele.action) == "modele" else regle
        return replace(gagnant, source=f"combine:{gagnant.source}")
