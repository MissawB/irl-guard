"""Décideur à base de règles : plancher de référence et repli quand Laya n'est pas utilisé."""

from __future__ import annotations

import re
import unicodedata

from irl_guard.config import SessionConfig
from irl_guard.decision.base import ACTIONS, Decision

VOIE = re.compile(r"\b(rue|avenue|av|boulevard|bd|place|chemin|impasse|allee|quai|route|square"
                  r"|cours|passage|residence|lotissement)\b")
NUMERO = re.compile(r"^\s*\d{1,4}\s*(bis|ter)?\s*$")
CODE_POSTAL = re.compile(r"\b\d{5}\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
TELEPHONE = re.compile(r"(\+33\s?|\b0)[1-9]([\s.-]?\d{2}){4}\b")
MOTS_OFFENSANTS_BASE = ("connard", "connasse", "salope", "salaud", "encule", "pute", "batard", "fdp", "ntm")


def normaliser(texte: str) -> str:
    sans_accents = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode()
    return sans_accents.lower()


class RulesDecider:
    def __init__(self, cfg: SessionConfig):
        mots = [normaliser(m) for m in (*MOTS_OFFENSANTS_BASE, *cfg.regles.mots_offensants)]
        self._offensant = re.compile(r"\b(" + "|".join(re.escape(m) for m in mots) + r")\b")
        self._categories_couteau_ok = {c.lower() for c in cfg.regles.categories_couteau_ok}

    def decide(self, state: dict) -> Decision:
        cible = state["cible"]
        if cible["type"] == "texte":
            return self._texte(normaliser(cible.get("texte") or ""), state, cible.get("taille", "petite"))
        if cible.get("classe") in ("couteau", "ciseaux"):
            if str(state.get("categorie", "")).lower() in self._categories_couteau_ok:
                return self._d("diffuser", "aucun", 0)
            return self._d("flouter", "arme", 2)
        return self._d("diffuser", "aucun", 0)

    def _texte(self, texte: str, state: dict, taille: str) -> Decision:
        if self._offensant.search(texte):
            return self._d("flouter", "propos", 2)
        if EMAIL.search(texte) or TELEPHONE.search(texte):
            return self._d("flouter", "donnee_perso", 2)
        if VOIE.search(texte) or NUMERO.match(texte) or CODE_POSTAL.search(texte):
            if state.get("mode_domicile") and taille != "petite":
                return self._d("pause", "localisation", 3)
            return self._d("flouter", "localisation", 2)
        return self._d("diffuser", "aucun", 0)

    @staticmethod
    def _d(action: str, risque: str, gravite: int) -> Decision:
        probas = {a: 1.0 if a == action else 0.0 for a in ACTIONS}
        return Decision(action, risque, gravite, {"action": probas}, "rules")
