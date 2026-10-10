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
