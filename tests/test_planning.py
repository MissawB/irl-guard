from irl_guard.analysis import Analyse, PisteAnalysee
from irl_guard.planning import planifier
from irl_guard.types import Box


def _a(frame, pistes, pause=False):
    return Analyse(frame, tuple(pistes), 1.0 if pause else 0.0, pause)


def _p(pid, boite, action="flouter"):
    return PisteAnalysee(pid, "texte", boite, "Rue", action, False)


def test_image_analysee_utilise_sa_propre_analyse():
    a = _a(10, [_p(1, Box(100, 100, 200, 200))])
    b = _a(20, [_p(1, Box(300, 100, 400, 200))], pause=True)
    plan = planifier(10, a, b, marge=0.0, largeur=1000, hauteur=1000)
    assert plan.masques == (Box(100, 100, 200, 200),) and plan.pause is False


def test_interpolation_et_marge():
    a = _a(10, [_p(1, Box(100, 100, 200, 200))])
    b = _a(20, [_p(1, Box(300, 100, 400, 200))])
    plan = planifier(15, a, b, marge=0.0, largeur=1000, hauteur=1000)
    assert plan.masques == (Box(200, 100, 300, 200),)
    plan = planifier(15, a, b, marge=0.2, largeur=1000, hauteur=1000)
    assert plan.masques == (Box(190, 90, 310, 210),)


def test_apparition_entre_deux_analyses():
    a = _a(10, [])
    b = _a(20, [_p(1, Box(300, 100, 400, 200))])
    assert planifier(11, a, b, 0.0, 1000, 1000).masques == (Box(300, 100, 400, 200),)


def test_disparition_garde_la_boite_precedente():
    a = _a(10, [_p(1, Box(100, 100, 200, 200))])
    b = _a(20, [])
    assert planifier(19, a, b, 0.0, 1000, 1000).masques == (Box(100, 100, 200, 200),)


def test_pistes_diffusees_non_masquees():
    a = _a(10, [_p(1, Box(0, 0, 10, 10), action="diffuser")])
    assert planifier(12, a, None, 0.0, 1000, 1000).masques == ()


def test_pause_si_l_une_des_deux_analyses_est_en_pause():
    assert planifier(15, _a(10, []), _a(20, [], pause=True), 0.0, 100, 100).pause is True
    assert planifier(15, _a(10, [], pause=True), _a(20, []), 0.0, 100, 100).pause is True
    assert planifier(15, _a(10, []), _a(20, []), 0.0, 100, 100).pause is False


def test_sans_analyse():
    plan = planifier(0, None, None, 0.15, 100, 100)
    assert plan.masques == () and plan.pause is False and plan.analyse is None
