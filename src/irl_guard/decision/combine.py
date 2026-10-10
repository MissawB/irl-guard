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
            # Le repli vaut « flouter » : il ne peut que relever une règle qui diffuserait. La source garde la
            # trace du repli pour que l'Analyzer ne le mette pas en cache (le modèle sera réessayé).
            if regle.action == "diffuser":
                return replace(modele, source="combine:fallback")
            return replace(regle, source=f"combine:{regle.source}+fallback")
        gagnant = modele if plus_protectrice(regle.action, modele.action) == "modele" else regle
        if regle.action == modele.action and regle.p_pause > modele.p_pause:
            gagnant = regle  # à égalité, ne jamais affaiblir le score de pause (hystérésis)
        return replace(gagnant, source=f"combine:{gagnant.source}")
