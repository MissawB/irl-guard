import pytest

from irl_guard.decision.base import Decision, decision_repli
from irl_guard.decision.combine import CombinedDecider, plus_protectrice


class Fixe:
    def __init__(self, decision):
        self.decision = decision

    def decide(self, state):
        return self.decision


def _d(action, source):
    return Decision(action, "localisation", 2, {}, source)


@pytest.mark.parametrize("regle, modele, gagnant", [
    ("diffuser", "diffuser", "modele"), ("diffuser", "flouter", "modele"), ("diffuser", "pause", "modele"),
    ("flouter", "diffuser", "regle"), ("flouter", "flouter", "modele"), ("flouter", "pause", "modele"),
    ("pause", "diffuser", "regle"), ("pause", "flouter", "regle"), ("pause", "pause", "modele"),
])
def test_plus_protectrice(regle, modele, gagnant):
    assert plus_protectrice(regle, modele) == gagnant


def test_source_du_gagnant():
    d = CombinedDecider(Fixe(_d("pause", "rules")), Fixe(_d("flouter", "laya-local"))).decide({})
    assert (d.action, d.source) == ("pause", "combine:rules")
    d = CombinedDecider(Fixe(_d("diffuser", "rules")), Fixe(_d("flouter", "laya-local"))).decide({})
    assert (d.action, d.source) == ("flouter", "combine:laya-local")


def test_repli_du_modele_jamais_diffuser():
    d = CombinedDecider(Fixe(_d("diffuser", "rules")), Fixe(decision_repli())).decide({})
    assert d.action == "flouter" and d.source == "combine:fallback"
    d = CombinedDecider(Fixe(_d("pause", "rules")), Fixe(decision_repli())).decide({})
    assert d.action == "pause" and d.source == "combine:rules"
