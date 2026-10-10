from irl_guard.analysis import Analyzer
from irl_guard.config import SessionConfig
from irl_guard.decision.base import Decision, decision_repli
from irl_guard.types import Box, Detection

FLOU = Decision("flouter", "localisation", 2,
                {"action": {"diffuser": 0.1, "flouter": 0.8, "pause": 0.1}}, "laya")
PAUSE = Decision("pause", "localisation", 3,
                 {"action": {"diffuser": 0.0, "flouter": 0.2, "pause": 0.8}}, "laya")


class FakeDecider:
    def __init__(self, decision):
        self.decision = decision
        self.states = []

    def decide(self, state):
        self.states.append(state)
        return self.decision


def _analyzer(decision=FLOU, **cfg):
    fake = FakeDecider(decision)
    return Analyzer(SessionConfig(charte="x", **cfg), fake, 1000, 1000), fake


def _texte(t="Rue Nationale"):
    return Detection("texte", 0.95, Box(100, 100, 300, 140), t)


def test_piste_contexte_decidee_puis_cachee():
    an, fake = _analyzer()
    for i in range(3):
        a = an.analyser(i, [_texte()])
    assert len(fake.states) == 1
    assert a.pistes[0].action == "flouter" and a.pistes[0].masquer
    assert fake.states[0]["cible"]["texte"] == "Rue Nationale"


def test_nouvel_appel_quand_le_texte_change():
    an, fake = _analyzer()
    an.analyser(0, [_texte("Rue Nationale")])
    an.analyser(1, [_texte("Rue Nationale 12")])
    assert len(fake.states) == 2


def test_repli_non_mis_en_cache():
    an, fake = _analyzer(decision_repli())
    for i in range(3):
        a = an.analyser(i, [_texte()])
    assert len(fake.states) == 3
    assert a.pistes[0].action == "flouter"


def test_classes_regle_et_scene_sans_decideur():
    an, fake = _analyzer()
    a = an.analyser(0, [Detection("visage", 0.9, Box(0, 0, 50, 50)),
                        Detection("plaque", 0.9, Box(500, 500, 600, 540)),
                        Detection("personne", 0.9, Box(0, 0, 200, 600))])
    actions = {p.classe: p.action for p in a.pistes}
    assert actions == {"visage": "flouter", "plaque": "flouter", "personne": "diffuser"}
    assert fake.states == []
    assert a.score == 0.0 and a.en_pause is False


def test_score_et_pause_depuis_p_pause():
    an, _ = _analyzer(PAUSE)
    a = an.analyser(0, [_texte()])
    assert a.score == 0.8 and a.en_pause is True
    assert a.appels[0]["piste"] == a.pistes[0].id
    assert a.appels[0]["decision"]["action"] == "pause"


def test_nudite_au_dessus_du_seuil_met_en_pause():
    an, _ = _analyzer()
    a = an.analyser(0, [Detection("nudite", 0.9, Box(0, 0, 100, 100))])
    assert a.score == 1.0 and a.en_pause is True and a.pistes[0].masquer


def test_nudite_sous_le_seuil_masquee_sans_pause():
    an, _ = _analyzer()
    a = an.analyser(0, [Detection("nudite", 0.5, Box(0, 0, 100, 100))])
    assert a.score == 0.0 and a.en_pause is False and a.pistes[0].masquer


def test_to_dict_serialisable():
    import json
    an, _ = _analyzer()
    json.dumps(an.analyser(0, [_texte()]).to_dict(), ensure_ascii=False)


class DeciderQuiPlante:
    def decide(self, state):
        raise RuntimeError("bug inattendu")


def test_exception_du_decideur_donne_repli():
    an = Analyzer(SessionConfig(charte="x"), DeciderQuiPlante(), 1000, 1000)
    a = an.analyser(0, [_texte()])
    assert a.pistes[0].action == "flouter"
    assert a.pistes[0].decision.source == "fallback"


def test_repli_dans_le_mode_combine_non_mis_en_cache():
    an, fake = _analyzer(Decision("flouter", "localisation", 2, {}, "combine:rules+fallback"))
    for i in range(3):
        an.analyser(i, [_texte()])
    assert len(fake.states) == 3
