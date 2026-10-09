"""Clauses de charte structurées et règle d'application déterministe (spec §2.1)."""

from __future__ import annotations

from dataclasses import dataclass

RISQUES_CHARTE = ("localisation", "donnee_perso", "arme", "propos")
SANCTIONS = ("diffuser", "flouter", "pause")
TYPES_CONDITION = ("toujours", "domicile", "sauf_categorie")
CATEGORIES_CLAUSE = ("cuisine", "gaming", "musique", "sport")
SEVERITE = {"diffuser": 0, "flouter": 1, "pause": 2}


@dataclass(frozen=True)
class Clause:
    risque: str
    sanction: str
    condition: str  # "toujours", "domicile" ou "sauf_categorie:<nom>"

    @property
    def type_condition(self) -> str:
        return self.condition.split(":", 1)[0]

    @property
    def categorie(self) -> str:
        return self.condition.split(":", 1)[1] if ":" in self.condition else ""

    @property
    def signature(self) -> tuple[str, str, str]:
        return (self.risque, self.sanction, self.type_condition)

    def to_dict(self) -> dict:
        return {"risque": self.risque, "sanction": self.sanction, "condition": self.condition}


def condition_remplie(clause: Clause, categorie: str, mode_domicile: bool) -> bool:
    t = clause.type_condition
    if t == "toujours":
        return True
    if t == "domicile":
        return mode_domicile
    if t == "sauf_categorie":
        return categorie.lower() != clause.categorie.lower()
    raise ValueError(f"condition inconnue : {clause.condition}")


def appliquer_charte(clauses: list[Clause], risque: str, categorie: str, mode_domicile: bool) -> str:
    """Sanction de la clause applicable la plus protectrice ; diffuser si aucune ne s'applique."""
    if risque == "aucun":
        return "diffuser"
    retenues = [c.sanction for c in clauses
                if c.risque == risque and condition_remplie(c, categorie, mode_domicile)]
    return max(retenues, key=SEVERITE.__getitem__, default="diffuser")


def charte_valide(clauses: list[Clause]) -> bool:
    paires = [(c.risque, c.type_condition) for c in clauses]
    return 2 <= len(clauses) <= 5 and len(paires) == len(set(paires))
