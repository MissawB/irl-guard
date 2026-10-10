import pytest

from irl_guard.config import SessionConfig
from irl_guard.decision.laya import LayaIndisponible
from irl_guard.decision.laya_local import LayaLocalDecider

CFG = SessionConfig(charte="x")
STATE = {"categorie": "IRL", "mode_domicile": True, "charte": "x",
         "cible": {"type": "texte", "texte": "Rue Nationale", "conf": 0.9, "taille": "grande"}, "contexte": []}


def _reponse(p_action=(0.2, 0.2, 0.6)):
    return {"answers": {
        "action": {"probabilities": dict(zip(("diffuser", "flouter", "pause"), p_action))},
        "risque": {"probabilities": {"localisation": 0.9, "donnee_perso": 0.025, "arme": 0.025,
                                     "propos": 0.025, "aucun": 0.025}},
        "gravite": {"score": 3, "probabilities": {"0": 0.1, "1": 0, "2": 0, "3": 0.9}},
    }}


class FauxAgent:
    def __init__(self, reponse=None, erreur=None):
        self.reponse, self.erreur, self.appels = reponse or _reponse(), erreur, []

    def predict(self, state, questions):
        self.appels.append((state, questions))
        if self.erreur:
            raise self.erreur
        return self.reponse


def test_decision_et_source():
    agent = FauxAgent()
    d = LayaLocalDecider(CFG, "MissawB/x", chargeur=lambda m: agent).decide(STATE)
    assert (d.action, d.risque, d.gravite, d.source) == ("pause", "localisation", 3, "laya-local")
    assert set(agent.appels[0][1]) == {"action", "risque", "gravite"}


def test_chargement_paresseux_et_unique():
    charges = []

    def chargeur(modele):
        charges.append(modele)
        return FauxAgent()

    decider = LayaLocalDecider(CFG, "MissawB/x", chargeur=chargeur)
    assert charges == []
    decider.decide(STATE)
    decider.decide(STATE)
    assert charges == ["MissawB/x"]


def test_erreur_de_prediction_donne_repli():
    d = LayaLocalDecider(CFG, "m", chargeur=lambda m: FauxAgent(erreur=RuntimeError("boum"))).decide(STATE)
    assert (d.action, d.source) == ("flouter", "fallback")


def test_reponse_invalide_donne_repli():
    d = LayaLocalDecider(CFG, "m", chargeur=lambda m: FauxAgent(reponse={"answers": {}})).decide(STATE)
    assert d.source == "fallback"


def test_verifier_leve_si_modele_introuvable():
    def chargeur(modele):
        raise OSError("dépôt introuvable")

    with pytest.raises(LayaIndisponible, match="MissawB/absent"):
        LayaLocalDecider(CFG, "MissawB/absent", chargeur=chargeur).verifier()


@pytest.mark.gpu
def test_vrai_modele_depuis_le_hub():
    d = LayaLocalDecider(CFG, "MissawB/irl-guard-decision-laya").decide(
        {**STATE, "charte": "Floute les insultes.", "cible": {**STATE["cible"], "texte": "Sale connard"},
         "mode_domicile": False})
    assert d.source == "laya-local" and d.action in ("flouter", "pause")


def test_chargement_par_defaut_sur_cpu_a_une_version_figee(monkeypatch):
    import sys
    import types

    appels = {}
    faux = types.ModuleType("laya")
    faux.load = lambda modele, **kw: appels.update(modele=modele, **kw) or FauxAgent()
    monkeypatch.setitem(sys.modules, "laya", faux)
    LayaLocalDecider(CFG, "MissawB/x", revision="abc123").decide(STATE)
    assert appels == {"modele": "MissawB/x", "device": "cpu", "revision": "abc123"}


def test_config_fige_la_version_du_modele_par_defaut():
    import re
    assert re.fullmatch(r"[0-9a-f]{40}", CFG.decideur.laya_revision)
