import json

import httpx
import pytest

from irl_guard.config import SessionConfig
from irl_guard.decision.laya import LayaDecider, LayaIndisponible

STATE = {"categorie": "IRL", "mode_domicile": False, "charte": "x",
         "cible": {"type": "texte", "texte": "Rue Nationale", "conf": 0.9, "taille": "grande"}, "contexte": []}


def _reponse(p_action=(0.1, 0.3, 0.6), score=2):
    return {
        "model": "laya-multilingual",
        "answers": {
            "action": {"choice": "pause", "confidence": 0.6,
                       "probabilities": dict(zip(("diffuser", "flouter", "pause"), p_action))},
            "risque": {"choice": "localisation", "confidence": 0.8,
                       "probabilities": {"localisation": 0.8, "donnee_perso": 0.05, "arme": 0.05,
                                         "propos": 0.05, "aucun": 0.05}},
            "gravite": {"score": score, "confidence": 0.5, "legend": [], "probabilities": {}},
        },
        "usage": {"input_tokens": 120, "output_tokens": 0},
    }


def _decider(handler, **laya):
    cfg = SessionConfig(charte="x", laya=laya)
    return LayaDecider(cfg, transport=httpx.MockTransport(handler))


def test_decision_normale_et_requete():
    requetes = []

    def handler(request):
        requetes.append(request)
        return httpx.Response(200, json=_reponse())

    d = _decider(handler, cle="secret").decide(STATE)
    assert (d.action, d.risque, d.gravite, d.source) == ("pause", "localisation", 2, "laya")
    assert d.p_pause == 0.6
    corps = json.loads(requetes[0].content)
    assert requetes[0].url.path == "/v1/systemone"
    assert corps["model"] == "laya-multilingual"
    assert corps["state"] == STATE
    assert set(corps["questions"]) == {"action", "risque", "gravite"}
    assert requetes[0].headers["Authorization"] == "Bearer secret"


@pytest.mark.parametrize("p_action, attendu", [
    ((0.5, 0.3, 0.2), "flouter"),
    ((0.8, 0.15, 0.05), "diffuser"),
])
def test_seuils_sur_probabilites(p_action, attendu):
    d = _decider(lambda r: httpx.Response(200, json=_reponse(p_action))).decide(STATE)
    assert d.action == attendu


def test_timeout_donne_repli():
    def handler(request):
        raise httpx.ReadTimeout("lent", request=request)

    d = _decider(handler).decide(STATE)
    assert (d.action, d.source) == ("flouter", "fallback")


def test_erreur_http_donne_repli():
    d = _decider(lambda r: httpx.Response(500, text="boom")).decide(STATE)
    assert (d.action, d.source) == ("flouter", "fallback")


def test_option_manquante_donne_repli():
    rep = _reponse()
    del rep["answers"]["action"]["probabilities"]["pause"]
    d = _decider(lambda r: httpx.Response(200, json=rep)).decide(STATE)
    assert d.source == "fallback"


def test_probabilite_hors_bornes_donne_repli():
    d = _decider(lambda r: httpx.Response(200, json=_reponse((1.5, 0.0, 0.0)))).decide(STATE)
    assert d.source == "fallback"


def test_gravite_libelle_et_bornes():
    assert _decider(lambda r: httpx.Response(200, json=_reponse(score="critique"))).decide(STATE).gravite == 3
    assert _decider(lambda r: httpx.Response(200, json=_reponse(score=7))).decide(STATE).gravite == 3
    assert _decider(lambda r: httpx.Response(200, json=_reponse(score="bizarre"))).decide(STATE).source == "fallback"


def test_verifier_leve_si_injoignable():
    def handler(request):
        raise httpx.ConnectError("refusé", request=request)

    with pytest.raises(LayaIndisponible, match="--decider rules"):
        _decider(handler).verifier()


def test_verifier_ok():
    _decider(lambda r: httpx.Response(200, json=_reponse())).verifier()


def test_score_infini_donne_repli_sans_planter():
    corps = json.dumps(_reponse()).replace('"score": 2', '"score": Infinity')
    d = _decider(lambda r: httpx.Response(200, content=corps.encode(),
                                          headers={"content-type": "application/json"})).decide(STATE)
    assert d.source == "fallback"


def test_probabilites_toutes_nulles_donnent_repli():
    d = _decider(lambda r: httpx.Response(200, json=_reponse((0.0, 0.0, 0.0)))).decide(STATE)
    assert (d.action, d.source) == ("flouter", "fallback")
