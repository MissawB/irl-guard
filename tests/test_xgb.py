import json

import pytest

from irl_guard.config import SessionConfig
from irl_guard.decision.base import Decision
from irl_guard.decision.rules import RulesDecider
from irl_guard.donnees.lignes import generer_jeux
from irl_guard.evaluation.predicteurs import charger_predictions, predire_avec_decider, reponses_depuis_decision
from irl_guard.evaluation.xgb import PredicteurXGBoost, features_structurees


def test_reponses_depuis_decision():
    rep = reponses_depuis_decision(Decision("pause", "propos", 3))
    assert rep["action"]["probabilities"] == {"diffuser": 0.0, "flouter": 0.0, "pause": 1.0}
    assert rep["risque"]["probabilities"]["propos"] == 1.0
    assert rep["gravite"]["score"] == 3.0
    repli = reponses_depuis_decision(Decision("flouter", "inconnu", 2, {}, "fallback"))
    assert sum(repli["risque"]["probabilities"].values()) == pytest.approx(1.0)


def test_predire_avec_decider_regles():
    lignes = generer_jeux(0, {"train": 0, "val": 0, "test_vues": 5, "test_non_vues": 0})["test_vues"]
    preds = predire_avec_decider(RulesDecider(SessionConfig(charte="x")), lignes)
    assert set(preds) == {l["id"] for l in lignes}
    assert all("probabilities" in p["action"] for p in preds.values())


def test_charger_predictions(tmp_path):
    chemin = tmp_path / "p.jsonl"
    chemin.write_text(json.dumps({"id": "x", "answers": {"gravite": {"score": 1}}}) + "\n", encoding="utf-8")
    assert charger_predictions(chemin) == {"x": {"gravite": {"score": 1}}}


def test_features_structurees():
    state = {"categorie": "Cuisine", "mode_domicile": True, "charte": "x",
             "cible": {"type": "texte", "texte": "Rue Nationale", "conf": 0.9, "taille": "grande"}, "contexte": []}
    f = features_structurees(state)
    assert f["voie"] == 1 and f["email"] == 0 and f["domicile"] == 1
    assert f["categorie=cuisine"] == 1 and f["type=texte"] == 1 and f["taille=grande"] == 1


@pytest.fixture(scope="module")
def jeux():
    return generer_jeux(0, {"train": 400, "val": 0, "test_vues": 50, "test_non_vues": 0})


def test_xgboost_entraine_et_predit(jeux):
    modele = PredicteurXGBoost().entrainer(jeux["train"])
    assert modele.tfidf.transform(["test"]).shape[1] <= 500
    preds = modele.predire(jeux["test_vues"])
    assert set(preds) == {l["id"] for l in jeux["test_vues"]}
    for p in preds.values():
        assert set(p["action"]["probabilities"]) == {"diffuser", "flouter", "pause"}
        assert sum(p["action"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
        assert 0 <= p["gravite"]["score"] <= 3


def test_xgboost_classe_absente_du_train(jeux):
    # Train sans aucune ligne « pause » : les prédictions couvrent quand même les trois actions.
    sans_pause = [l for l in jeux["train"] if json.loads(l["gold"])["action"]["label"] != "pause"]
    preds = PredicteurXGBoost().entrainer(sans_pause).predire(jeux["test_vues"][:5])
    for p in preds.values():
        assert p["action"]["probabilities"]["pause"] == 0.0
        assert set(p["risque"]["probabilities"]) == {"localisation", "donnee_perso", "arme", "propos", "aucun"}
