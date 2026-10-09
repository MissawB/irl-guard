import pytest

from irl_guard.config import SessionConfig
from irl_guard.decision.base import Decision, action_depuis_probas, decision_repli
from irl_guard.decision.rules import RulesDecider


def _state(texte=None, classe=None, domicile=False, taille="moyenne", categorie="IRL"):
    cible = {"type": "texte" if classe is None else "objet", "conf": 0.9, "taille": taille}
    if classe:
        cible["classe"] = classe
    if texte:
        cible["texte"] = texte
    return {"categorie": categorie, "mode_domicile": domicile, "charte": "x", "cible": cible, "contexte": []}


@pytest.fixture
def rules():
    return RulesDecider(SessionConfig(charte="x", regles={"mots_offensants": ["boloss"]}))


@pytest.mark.parametrize("state, action, risque", [
    (_state("Rue Nationale"), "flouter", "localisation"),
    (_state("Rue Nationale", domicile=True, taille="grande"), "pause", "localisation"),
    (_state("Rue Nationale", domicile=True, taille="petite"), "flouter", "localisation"),
    (_state("Allée des Érables"), "flouter", "localisation"),
    (_state("12"), "flouter", "localisation"),
    (_state("Paris 75011"), "flouter", "localisation"),
    (_state("SOLDES -50 %"), "diffuser", "aucun"),
    (_state("contact : jean.dupont@mail.fr"), "flouter", "donnee_perso"),
    (_state("06 12 34 56 78"), "flouter", "donnee_perso"),
    (_state("Sale connard"), "flouter", "propos"),
    (_state("t'es un boloss"), "flouter", "propos"),
    (_state("une dispute"), "diffuser", "aucun"),
    (_state(classe="couteau"), "flouter", "arme"),
    (_state(classe="couteau", categorie="Cuisine"), "diffuser", "aucun"),
    (_state(classe="telephone"), "diffuser", "aucun"),
])
def test_regles(rules, state, action, risque):
    d = rules.decide(state)
    assert (d.action, d.risque, d.source) == (action, risque, "rules")
    assert d.probabilites["action"][action] == 1.0


def test_p_pause():
    assert Decision("pause", "localisation", 3, {"action": {"diffuser": 0, "flouter": 0.2, "pause": 0.8}}).p_pause == 0.8
    assert Decision("pause", "localisation", 3).p_pause == 1.0
    assert decision_repli().p_pause == 0.0
    assert decision_repli().action == "flouter"


@pytest.mark.parametrize("p, attendu", [
    ({"diffuser": 0.1, "flouter": 0.3, "pause": 0.6}, "pause"),
    ({"diffuser": 0.5, "flouter": 0.3, "pause": 0.2}, "flouter"),
    ({"diffuser": 0.8, "flouter": 0.15, "pause": 0.05}, "diffuser"),
])
def test_action_depuis_probas(p, attendu):
    assert action_depuis_probas(p, seuil_pause=0.5, seuil_flou=0.4) == attendu
