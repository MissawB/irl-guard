import pytest

from irl_guard.donnees.clauses import Clause, appliquer_charte, charte_valide, condition_remplie

ROUGE = Clause("localisation", "flouter", "toujours")


def test_proprietes_de_clause():
    c = Clause("arme", "flouter", "sauf_categorie:cuisine")
    assert c.type_condition == "sauf_categorie"
    assert c.categorie == "cuisine"
    assert c.signature == ("arme", "flouter", "sauf_categorie")
    assert c.to_dict() == {"risque": "arme", "sanction": "flouter", "condition": "sauf_categorie:cuisine"}
    assert ROUGE.categorie == ""


@pytest.mark.parametrize("condition, categorie, domicile, attendu", [
    ("toujours", "IRL", False, True),
    ("domicile", "IRL", False, False),
    ("domicile", "IRL", True, True),
    ("sauf_categorie:cuisine", "Cuisine", False, False),
    ("sauf_categorie:cuisine", "IRL", False, True),
])
def test_condition_remplie(condition, categorie, domicile, attendu):
    assert condition_remplie(Clause("arme", "flouter", condition), categorie, domicile) is attendu


def test_condition_inconnue():
    with pytest.raises(ValueError):
        condition_remplie(Clause("arme", "flouter", "parfois"), "IRL", False)


def test_couteau_interdit_sauf_cuisine():
    clauses = [Clause("arme", "flouter", "sauf_categorie:cuisine")]
    assert appliquer_charte(clauses, "arme", "IRL", False) == "flouter"
    assert appliquer_charte(clauses, "arme", "Cuisine", False) == "diffuser"


def test_la_plus_protectrice_l_emporte():
    clauses = [Clause("localisation", "flouter", "toujours"), Clause("localisation", "pause", "domicile")]
    assert appliquer_charte(clauses, "localisation", "IRL", False) == "flouter"
    assert appliquer_charte(clauses, "localisation", "IRL", True) == "pause"


def test_risque_non_mentionne_ou_aucun_est_diffuse():
    assert appliquer_charte([ROUGE], "propos", "IRL", True) == "diffuser"
    assert appliquer_charte([ROUGE], "aucun", "IRL", True) == "diffuser"


def test_autorisation_explicite():
    assert appliquer_charte([Clause("propos", "diffuser", "toujours")], "propos", "IRL", False) == "diffuser"


def test_charte_valide():
    assert charte_valide([ROUGE, Clause("arme", "pause", "toujours")])
    assert not charte_valide([ROUGE])  # moins de 2 clauses
    assert not charte_valide([ROUGE, Clause("localisation", "pause", "toujours")])  # même (risque, condition)
    assert not charte_valide([Clause(r, "flouter", t) for r in ("localisation", "arme", "propos")
                              for t in ("toujours", "domicile")])  # plus de 5 clauses
