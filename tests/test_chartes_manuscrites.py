import json

from irl_guard.donnees.chartes_manuscrites import CHARTES
from irl_guard.donnees.clauses import RISQUES_CHARTE, SANCTIONS, TYPES_CONDITION, Clause, appliquer_charte
from irl_guard.donnees.gabarits import PREFIXES, SUJETS, VERBES
from irl_guard.donnees.lignes import generer_jeu_manuscrit


def test_vingt_chartes_uniques_et_bien_annotees():
    assert len(CHARTES) == 20
    assert len({c.texte for c in CHARTES}) == 20
    for c in CHARTES:
        assert c.clauses, c.texte
        for clause in c.clauses:
            assert clause.risque in RISQUES_CHARTE and clause.sanction in SANCTIONS
            assert clause.type_condition in TYPES_CONDITION


def test_couverture_des_risques_et_conditions():
    clauses = [cl for c in CHARTES for cl in c.clauses]
    assert {cl.risque for cl in clauses} == set(RISQUES_CHARTE)
    assert {cl.type_condition for cl in clauses} == set(TYPES_CONDITION)
    assert {cl.sanction for cl in clauses} == set(SANCTIONS)


def test_aucun_morceau_de_gabarit_dans_les_chartes():
    fragments = [s for t in SUJETS.values() for s in t]
    fragments += [v.split("{")[0] for t in VERBES.values() for v in t if not v.startswith("{")]
    fragments += [p.split("{")[0] for t in PREFIXES.values() for p in t if p]
    for c in CHARTES:
        for fragment in fragments:
            assert fragment.lower() not in c.texte.lower(), (fragment, c.texte)


def test_jeu_manuscrit():
    lignes = generer_jeu_manuscrit(0, 200)
    assert len(lignes) == 200 and len({l["id"] for l in lignes}) == 200
    textes = {c.texte for c in CHARTES}
    vus = set()
    for l in lignes:
        state, gold = json.loads(l["state"]), json.loads(l["gold"])
        assert state["charte"] in textes
        vus.add(state["charte"])
        clauses = [Clause(**c) for c in json.loads(l["clauses"])]
        attendu = appliquer_charte(clauses, gold["risque"]["label"], state["categorie"], state["mode_domicile"])
        assert gold["action"]["label"] == attendu
    assert vus == textes
    assert generer_jeu_manuscrit(0, 20) == generer_jeu_manuscrit(0, 20)
