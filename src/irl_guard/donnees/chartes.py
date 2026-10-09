"""Tirage des chartes (ensembles de clauses) et rendu en texte."""

from __future__ import annotations

import random

from irl_guard.donnees.clauses import (
    CATEGORIES_CLAUSE, RISQUES_CHARTE, SANCTIONS, TYPES_CONDITION, Clause, charte_valide,
)
from irl_guard.donnees.gabarits import Composants, choisir_composants, composer

# Paires de signatures (risque, sanction, type_condition) réservées au test « chartes non vues ».
# Les deux clauses de chaque paire sont protectrices : chacune peut changer un label.
RESERVEES = (
    (("arme", "pause", "toujours"), ("propos", "flouter", "domicile")),
    (("localisation", "pause", "domicile"), ("donnee_perso", "flouter", "sauf_categorie")),
    (("arme", "flouter", "sauf_categorie"), ("localisation", "pause", "toujours")),
)
PREAMBULES = ("", "Live tout public.", "Ambiance détendue.", "Merci de respecter ces règles.", "Règles du stream :")


def contient_reservee(clauses: list[Clause]) -> bool:
    signatures = {c.signature for c in clauses}
    return any(a in signatures and b in signatures for a, b in RESERVEES)


def _clause(risque: str, sanction: str, type_condition: str, rng: random.Random) -> Clause:
    condition = f"sauf_categorie:{rng.choice(CATEGORIES_CLAUSE)}" if type_condition == "sauf_categorie" \
        else type_condition
    return Clause(risque, sanction, condition)


def _clause_aleatoire(rng: random.Random) -> Clause:
    sanction = rng.choice(SANCTIONS)
    type_condition = "toujours" if sanction == "diffuser" else rng.choice(TYPES_CONDITION)
    return _clause(rng.choice(RISQUES_CHARTE), sanction, type_condition, rng)


def tirer_clauses(rng: random.Random, reservee: bool) -> list[Clause]:
    while True:
        n = rng.randint(2, 5)
        clauses: list[Clause] = []
        if reservee:
            a, b = rng.choice(RESERVEES)
            clauses = [_clause(*a, rng), _clause(*b, rng)]
        for _ in range(50):  # borne : une charte peut n'admettre aucune clause de plus (autorisations partout)
            if len(clauses) >= n:
                break
            candidate = _clause_aleatoire(rng)
            if not clauses or charte_valide([*clauses, candidate]):
                clauses.append(candidate)
        rng.shuffle(clauses)
        if charte_valide(clauses) and contient_reservee(clauses) == reservee:
            return clauses


def rendre_charte(clauses: list[Clause], rng: random.Random, train: Composants, test: Composants,
                  mode: str) -> str:
    phrases = [rng.choice(PREAMBULES)]
    for clause in clauses:
        sujet, verbe, prefixe = choisir_composants(clause, rng, train, test, mode)
        phrases.append(composer(sujet, verbe, prefixe, clause.categorie))
    return " ".join(p for p in phrases if p)
