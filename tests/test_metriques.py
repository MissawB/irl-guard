import json

import pytest

from irl_guard.config import SessionConfig
from irl_guard.donnees.lignes import construire_gold
from irl_guard.evaluation.metriques import (
    action_predite, ece_et_fiabilite, evaluer_predictions, niveau_gravite,
)

CFG = SessionConfig(charte="x")


def _ligne(id_, action, risque):
    return {"id": id_, "gold": json.dumps(construire_gold(action, risque))}


def _rep(p_action, risque, score):
    return {"action": {"probabilities": dict(zip(("diffuser", "flouter", "pause"), p_action))},
            "risque": {"probabilities": {risque: 1.0}}, "gravite": {"score": score}}


def test_couts_par_defaut():
    c = CFG.evaluation.couts
    assert c["diffuser"]["flouter"] == 10 and c["diffuser"]["pause"] == 20
    assert c["pause"]["diffuser"] == 3 and c["flouter"]["flouter"] == 0


def test_action_predite_et_gravite():
    assert action_predite(_rep((0.5, 0.3, 0.2), "aucun", 0), CFG.seuils) == "flouter"
    assert niveau_gravite(_rep((1, 0, 0), "aucun", 1.6)) == 2
    assert niveau_gravite(_rep((1, 0, 0), "aucun", 7)) == 3


def test_ece_calcule_a_la_main():
    ece, points = ece_et_fiabilite([0.95, 0.95, 0.55, 0.55], [True, False, True, True])
    # classe 0.9-1.0 : confiance 0.95, précision 0.5 ; classe 0.5-0.6 : confiance 0.55, précision 1.0
    assert ece == pytest.approx(0.5 * 0.45 + 0.5 * 0.45)
    assert points == [(pytest.approx(0.55), 1.0, 2), (pytest.approx(0.95), 0.5, 2)]


def test_evaluer_predictions():
    lignes = [_ligne("a", "flouter", "localisation"), _ligne("b", "pause", "propos"),
              _ligne("c", "diffuser", "aucun"), _ligne("d", "flouter", "arme")]
    predictions = {
        "a": _rep((0.1, 0.8, 0.1), "localisation", 2.0),  # juste
        "b": _rep((0.9, 0.05, 0.05), "propos", 0.0),      # fuite grave : coût 20
        "c": _rep((0.0, 0.0, 1.0), "aucun", 3.0),         # pause injustifiée : coût 3
        "d": _rep((0.1, 0.9, 0.0), "donnee_perso", 2.0),  # action juste, risque faux
    }
    r = evaluer_predictions(lignes, predictions, CFG.seuils, CFG.evaluation.couts)
    assert r.n == 4
    assert r.precision_action == 0.5
    assert r.precision_risque == 0.75
    assert r.precision_gravite == 0.5
    assert r.taux_fuite == pytest.approx(1 / 3)  # 1 fuite sur 3 lignes à protéger
    assert r.cout_moyen == pytest.approx((0 + 20 + 3 + 0) / 4)
    assert r.f1_macro_action == pytest.approx(1 / 3)  # flouter F1=1, pause 0, diffuser 0


def test_prediction_manquante():
    with pytest.raises(ValueError, match="b"):
        evaluer_predictions([_ligne("a", "flouter", "arme"), _ligne("b", "pause", "arme")],
                            {"a": _rep((0, 1, 0), "arme", 2)}, CFG.seuils, CFG.evaluation.couts)


def test_gravite_par_argmax_des_probabilites():
    # Espérance 1.2 (niveau 1 jamais attendu) : l'argmax des probabilités donne le bon niveau 0.
    assert niveau_gravite({"gravite": {"score": 1.2, "probabilities": {"0": 0.6, "1": 0, "2": 0, "3": 0.4}}}) == 0
    assert niveau_gravite({"gravite": {"score": 2.0, "probabilities": {"aucune": 0.1, "critique": 0.9}}}) == 3
