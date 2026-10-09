import json

import pytest

from irl_guard.decision.base import QUESTIONS
from irl_guard.donnees.chartes import contient_reservee
from irl_guard.donnees.clauses import Clause, appliquer_charte
from irl_guard.donnees.gabarits import partitionner
from irl_guard.donnees.lignes import construire_gold, ecrire_jeux, generer_jeux, lire_jeu
from irl_guard.state import estimer_tokens

PETIT = {"train": 120, "val": 20, "test_vues": 40, "test_non_vues": 40}


@pytest.fixture(scope="module")
def jeux():
    return generer_jeux(0, PETIT)


def test_tailles_et_ids_uniques(jeux):
    assert {nom: len(l) for nom, l in jeux.items()} == PETIT
    ids = [r["id"] for l in jeux.values() for r in l]
    assert len(ids) == len(set(ids))


def test_deterministe():
    assert generer_jeux(3, {"train": 10, "val": 1, "test_vues": 1, "test_non_vues": 1}) == \
        generer_jeux(3, {"train": 10, "val": 1, "test_vues": 1, "test_non_vues": 1})


def test_format_des_lignes(jeux):
    for r in jeux["train"][:50]:
        assert set(r) == {"id", "state", "questions", "gold", "clauses"}
        assert json.loads(r["questions"]) == QUESTIONS
        state = json.loads(r["state"])
        assert list(state) == ["categorie", "mode_domicile", "charte", "cible", "contexte"]
        gold = json.loads(r["gold"])
        for nom in ("action", "risque", "gravite"):
            assert sum(gold[nom]["probabilities"].values()) == pytest.approx(1.0)
            assert gold[nom]["label"] in gold[nom]["probabilities"]


def test_label_coherent_avec_les_clauses(jeux):
    for r in jeux["train"]:
        state, gold = json.loads(r["state"]), json.loads(r["gold"])
        clauses = [Clause(**c) for c in json.loads(r["clauses"])]
        attendu = appliquer_charte(clauses, gold["risque"]["label"], state["categorie"], state["mode_domicile"])
        assert gold["action"]["label"] == attendu


def test_combinaisons_reservees_seulement_en_test_non_vues(jeux):
    def reservee(r):
        return contient_reservee([Clause(**c) for c in json.loads(r["clauses"])])
    for nom in ("train", "val", "test_vues"):
        assert not any(reservee(r) for r in jeux[nom])
    assert any(reservee(r) for r in jeux["test_non_vues"])


def test_aucun_composant_reserve_dans_le_train(jeux):
    _, test = partitionner(0)
    reserves = [s for t in test.sujets.values() for s in t]
    reserves += [v.split("{")[0] for t in test.verbes.values() for v in t if not v.startswith("{")]
    reserves += [p.split("{")[0] for t in test.prefixes.values() for p in t if p]
    for r in jeux["train"]:
        charte = json.loads(r["state"])["charte"]
        for fragment in reserves:
            assert fragment.lower() not in charte.lower(), (fragment, charte)


def test_gold():
    gold = construire_gold("pause", "localisation")
    assert gold["action"]["probabilities"]["pause"] == pytest.approx(0.9)
    assert gold["action"]["probabilities"]["diffuser"] == pytest.approx(0.05)
    assert gold["gravite"]["label"] == "3" and gold["gravite"]["score"] == 3.0
    assert construire_gold("diffuser", "aucun")["gravite"]["label"] == "0"
    assert construire_gold("flouter", "arme")["gravite"]["label"] == "2"


def test_states_dans_le_budget_de_tokens(jeux):
    for nom in ("train", "test_non_vues"):
        for r in jeux[nom]:
            assert estimer_tokens(json.loads(r["state"])) <= 700


def test_ecriture_lecture(jeux, tmp_path):
    chemins = ecrire_jeux(jeux, tmp_path / "decision")
    assert sorted(p.name for p in chemins) == sorted(f"{n}.jsonl" for n in PETIT)
    assert lire_jeu(tmp_path / "decision" / "val.jsonl") == jeux["val"]


def test_cli_generer(tmp_path):
    from typer.testing import CliRunner

    from irl_guard import cli
    r = CliRunner().invoke(cli.app, ["donnees", "generer", "--sortie", str(tmp_path / "d"), "--train", "5",
                                     "--val", "1", "--test-vues", "1", "--test-non-vues", "2"])
    assert r.exit_code == 0, r.output
    assert len(lire_jeu(tmp_path / "d" / "test_non_vues.jsonl")) == 2
