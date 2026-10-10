from irl_guard.config import SessionConfig
from irl_guard.decision.qwen_local import QwenLocalDecider

CFG = SessionConfig(charte="x")
STATE = {"categorie": "IRL", "mode_domicile": False, "charte": "x",
         "cible": {"type": "texte", "texte": "Sale connard", "conf": 0.9, "taille": "grande"}, "contexte": []}
REPONSE = {"answers": {
    "action": {"probabilities": {"diffuser": 0.05, "flouter": 0.9, "pause": 0.05}},
    "risque": {"probabilities": {"localisation": 0.0, "donnee_perso": 0.0, "arme": 0.0, "propos": 1.0, "aucun": 0.0}},
    "gravite": {"score": 2, "probabilities": {"0": 0, "1": 0, "2": 1.0, "3": 0}},
}}


def test_decision_via_systemone():
    requetes = []

    def systemone(requete):
        requetes.append(requete)
        return REPONSE

    d = QwenLocalDecider(CFG, chargeur=lambda: systemone).decide(STATE)
    assert (d.action, d.risque, d.source) == ("flouter", "propos", "qwen-local")
    assert requetes[0]["state"] == STATE and set(requetes[0]["questions"]) == {"action", "risque", "gravite"}


def test_erreur_donne_repli():
    def systemone(requete):
        raise RuntimeError("CUDA out of memory")

    assert QwenLocalDecider(CFG, chargeur=lambda: systemone).decide(STATE).source == "fallback"


def test_creer_decider_modele_selon_la_version(monkeypatch):
    from irl_guard import cli
    from irl_guard.decision import laya_local, qwen_local
    v2 = cli.creer_decider_modele("qwen-ft-v2", CFG)
    assert isinstance(v2, QwenLocalDecider) and v2.repo == "MissawB/irl-guard-decision-qwen-v2"
    assert cli.creer_decider_modele("qwen-ft", CFG).repo == "MissawB/irl-guard-decision-qwen"
    laya_v2 = cli.creer_decider_modele("laya-ft-v2", CFG)
    assert isinstance(laya_v2, laya_local.LayaLocalDecider) and laya_v2.modele == "MissawB/irl-guard-decision-laya-v2"
    assert cli.creer_decider_modele("laya-ft", CFG).modele == "MissawB/irl-guard-decision-laya"


def test_version_figee_transmise_au_chargeur(monkeypatch):
    from irl_guard.decision import qwen_local
    appels = []
    monkeypatch.setattr(qwen_local, "_charger_qwen",
                        lambda repo, revision=None: appels.append((repo, revision)) or (lambda r: REPONSE))
    QwenLocalDecider(CFG, repo="MissawB/q", revision="abc").decide(STATE)
    assert appels == [("MissawB/q", "abc")]


def test_verifier_leve_si_le_modele_ne_charge_pas():
    import pytest

    from irl_guard.decision.laya import LayaIndisponible

    def chargeur():
        raise OSError("pas de GPU")

    with pytest.raises(LayaIndisponible, match="Qwen"):
        QwenLocalDecider(CFG, chargeur=chargeur).verifier()
