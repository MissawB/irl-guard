import random

from irl_guard.donnees.chartes import RESERVEES, contient_reservee, rendre_charte, tirer_clauses
from irl_guard.donnees.clauses import Clause, charte_valide
from irl_guard.donnees.gabarits import partitionner


def test_contient_reservee():
    (a, b) = RESERVEES[0]
    clauses = [Clause(a[0], a[1], a[2]), Clause(b[0], b[1], b[2])]
    assert contient_reservee(clauses)
    assert not contient_reservee(clauses[:1] + [Clause("arme", "flouter", "domicile")])


def test_tirage_non_reserve():
    rng = random.Random(0)
    for _ in range(300):
        clauses = tirer_clauses(rng, reservee=False)
        assert charte_valide(clauses) and not contient_reservee(clauses)


def test_tirage_reserve():
    rng = random.Random(0)
    for _ in range(300):
        clauses = tirer_clauses(rng, reservee=True)
        assert charte_valide(clauses) and contient_reservee(clauses)


def test_sauf_categorie_porte_une_categorie():
    rng = random.Random(3)
    for _ in range(200):
        for c in tirer_clauses(rng, reservee=False):
            if c.type_condition == "sauf_categorie":
                assert c.categorie in ("cuisine", "gaming", "musique", "sport")


def test_rendu_contient_chaque_clause():
    train, test = partitionner(0)
    rng = random.Random(4)
    clauses = [Clause("arme", "flouter", "sauf_categorie:cuisine"), Clause("propos", "pause", "toujours")]
    texte = rendre_charte(clauses, rng, train, test, "train")
    assert "cuisine" in texte
    assert texte.count(".") >= 2
    assert texte == texte.strip() and "  " not in texte


def test_paires_reservees_toutes_protectrices():
    for a, b in RESERVEES:
        assert a[1] != "diffuser" and b[1] != "diffuser"
