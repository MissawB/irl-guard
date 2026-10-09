from irl_guard.config import SessionConfig, TailleConfig
from irl_guard.state import construire_state, estimer_tokens, resume_contexte, taille_relative
from irl_guard.tracking import Piste
from irl_guard.types import Box

W, H = 1000, 1000


def _p(pid, classe, texte=None, boite=Box(0, 0, 100, 100), perdu=False):
    return Piste(pid, classe, boite, 0.9, texte=texte, perdu=perdu)


def test_taille_relative():
    t = TailleConfig()
    assert taille_relative(Box(0, 0, 100, 100), W, H, t) == "petite"    # 1 %
    assert taille_relative(Box(0, 0, 200, 200), W, H, t) == "moyenne"   # 4 %
    assert taille_relative(Box(0, 0, 300, 300), W, H, t) == "grande"    # 9 %


def test_resume_contexte_exclut_cible_et_pistes_perdues():
    cible = _p(1, "texte", "Rue Nationale")
    pistes = [cible, _p(2, "personne"), _p(3, "personne"), _p(4, "voiture"),
              _p(5, "texte", "SOLDES -50 %"), _p(6, "voiture", perdu=True)]
    assert resume_contexte(cible, pistes) == ["personne x2", "voiture x1", "texte: SOLDES -50 %"]


def test_state_texte():
    cfg = SessionConfig(charte="Pas de rue.", mode_domicile=True)
    cible = _p(1, "texte", "Rue Nationale", boite=Box(0, 0, 300, 300))
    state = construire_state(cible, [cible, _p(2, "personne")], cfg, W, H)
    assert state == {
        "categorie": "IRL",
        "mode_domicile": True,
        "charte": "Pas de rue.",
        "cible": {"type": "texte", "texte": "Rue Nationale", "conf": 0.9, "taille": "grande"},
        "contexte": ["personne x1"],
    }


def test_state_objet_porte_sa_classe():
    cfg = SessionConfig(charte="x")
    cible = _p(1, "couteau")
    assert construire_state(cible, [cible], cfg, W, H)["cible"] == {
        "type": "objet", "classe": "couteau", "conf": 0.9, "taille": "petite"}


def test_troncature_du_contexte_seulement():
    charte = "Règle très importante. " * 10
    cfg = SessionConfig(charte=charte, budget_tokens_state=170)
    cible = _p(1, "texte", "Rue Nationale")
    autres = [_p(i, "texte", f"enseigne numero {i} tres longue") for i in range(2, 60)]
    state = construire_state(cible, [cible, *autres], cfg, W, H)
    assert estimer_tokens(state) <= 170
    assert state["charte"] == charte
    assert state["cible"]["texte"] == "Rue Nationale"
    assert 0 < len(state["contexte"]) < 58
