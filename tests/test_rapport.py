import json

from irl_guard.config import SessionConfig
from irl_guard.donnees.lignes import ecrire_jeux, generer_jeux
from irl_guard.evaluation.metriques import Resultat
from irl_guard.evaluation.rapport import ecrire_rapport, obtenir_predictions

R = Resultat(10, 0.9, 0.8, 0.7, 0.05, 0.12, 1.5, [(0.55, 0.6, 4), (0.95, 0.9, 6)])
PETIT = {"train": 60, "val": 0, "test_vues": 10, "test_non_vues": 10}


def test_rapport_tableau_et_png(tmp_path):
    chemin = ecrire_rapport({"regles": {"test_vues": R, "test_non_vues": R},
                             "qwen-ft": {"test_vues": None, "test_non_vues": None}}, tmp_path)
    texte = chemin.read_text(encoding="utf-8")
    assert "| regles |" in texte and "90.0 %" in texte and "5.0 %" in texte
    assert "qwen-ft" in texte and "absent" in texte
    assert (tmp_path / "fiabilite_test_vues.png").exists()
    assert (tmp_path / "fiabilite_test_non_vues.png").exists()


def test_approche_absente_signalee(tmp_path):
    jeux = generer_jeux(0, PETIT)
    preds = obtenir_predictions("qwen-ft", SessionConfig(charte="x"), tmp_path, tmp_path / "predictions",
                                {j: jeux[j] for j in ("test_vues", "test_non_vues")})
    assert preds == {"test_vues": None, "test_non_vues": None}


def test_predictions_depuis_fichier(tmp_path):
    jeux = generer_jeux(0, PETIT)
    dossier = tmp_path / "predictions" / "laya-ft"
    dossier.mkdir(parents=True)
    with (dossier / "predictions_test_vues.jsonl").open("w", encoding="utf-8") as f:
        for l in jeux["test_vues"]:
            f.write(json.dumps({"id": l["id"], "answers": {"x": 1}}) + "\n")
    preds = obtenir_predictions("laya-ft", SessionConfig(charte="x"), tmp_path, tmp_path / "predictions",
                                {j: jeux[j] for j in ("test_vues", "test_non_vues")})
    assert len(preds["test_vues"]) == 10 and preds["test_non_vues"] is None


def test_laya_injoignable_donne_absent(tmp_path):
    jeux = generer_jeux(0, PETIT)
    cfg = SessionConfig(charte="x", laya={"url": "http://127.0.0.1:9", "timeout_s": 0.5})
    preds = obtenir_predictions("laya", cfg, tmp_path, tmp_path, {"test_vues": jeux["test_vues"]})
    assert preds == {"test_vues": None}


def test_cli_evaluer_regles_et_xgboost(tmp_path):
    from typer.testing import CliRunner

    from irl_guard import cli
    ecrire_jeux(generer_jeux(0, PETIT), tmp_path / "decision")
    config = tmp_path / "s.yaml"
    config.write_text("charte: x\n", encoding="utf-8")
    r = CliRunner().invoke(cli.app, ["evaluer", "--donnees", str(tmp_path / "decision"),
                                     "--predictions", str(tmp_path / "predictions"),
                                     "--approches", "regles,xgboost,qwen-ft", "--config", str(config),
                                     "--sortie", str(tmp_path / "resultats")])
    assert r.exit_code == 0, r.output
    texte = (tmp_path / "resultats" / "rapport.md").read_text(encoding="utf-8")
    assert "| regles |" in texte and "| xgboost |" in texte and "absent" in texte
