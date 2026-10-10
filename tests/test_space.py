import ast
import sys
from pathlib import Path

from irl_guard.config import SessionConfig
from irl_guard.decision.base import Decision
from irl_guard.decision.rules import RulesDecider

sys.path.insert(0, str(Path("space").resolve()))
import logique  # noqa: E402


class Fixe:
    def __init__(self, action):
        self.action = action

    def decide(self, state):
        return Decision(self.action, "localisation", 2,
                        {"action": {"diffuser": 0.1, "flouter": 0.2, "pause": 0.7}}, "laya-local")


def test_construire_state():
    s = logique.construire_state("Floute les insultes.", "IRL", True, "texte", "Sale connard", "couteau", "grande")
    assert s == {"categorie": "IRL", "mode_domicile": True, "charte": "Floute les insultes.",
                 "cible": {"type": "texte", "texte": "Sale connard", "conf": 0.9, "taille": "grande"},
                 "contexte": []}
    o = logique.construire_state("x", "Cuisine", False, "objet", "", "couteau", "moyenne")
    assert o["cible"] == {"type": "objet", "classe": "couteau", "conf": 0.9, "taille": "moyenne"}


def test_decider_tout_sans_qwen():
    regles = RulesDecider(SessionConfig(charte="x"))
    state = logique.construire_state("x", "IRL", False, "texte", "Rue Nationale", "", "grande")
    lignes = logique.decider_tout(state, regles, Fixe("pause"), None)
    noms = [l[0] for l in lignes]
    assert noms == ["Règles", "Laya fine-tuné", "Règles + Laya", "Qwen fine-tuné"]
    assert lignes[2][1] == "pause"
    assert lignes[3][1] == "indisponible en ligne"


def test_decider_tout_avec_qwen():
    regles = RulesDecider(SessionConfig(charte="x"))
    state = logique.construire_state("x", "IRL", False, "texte", "SOLDES -50 %", "", "petite")
    lignes = logique.decider_tout(state, regles, Fixe("diffuser"), Fixe("flouter"))
    assert [l[0] for l in lignes][-2:] == ["Qwen fine-tuné", "Règles + Qwen"]
    assert lignes[-1][1] == "flouter"


def test_exemples_complets():
    assert len(logique.EXEMPLES) >= 4
    assert all(len(e) == 7 for e in logique.EXEMPLES)


def test_app_syntaxe_et_readme():
    ast.parse(Path("space/app.py").read_text(encoding="utf-8"))
    readme = Path("space/README.md").read_text(encoding="utf-8")
    assert "sdk: gradio" in readme and "app_file: app.py" in readme
    exigences = Path("space/requirements.txt").read_text(encoding="utf-8")
    for paquet in ("gradio", "spaces", "laya", "transformers", "torchvision", "accelerate", "pydantic"):
        assert paquet in exigences
