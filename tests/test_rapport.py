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


def test_baseline_toujours_diffuser(tmp_path):
    jeux = generer_jeux(0, PETIT)
    preds = obtenir_predictions("toujours-diffuser", SessionConfig(charte="x"), tmp_path, tmp_path,
                                {"test_vues": jeux["test_vues"]})
    assert all(p["action"]["probabilities"]["diffuser"] == 1.0 for p in preds["test_vues"].values())


def test_laya_avec_trop_de_replis_donne_absent(tmp_path, monkeypatch):
    from irl_guard.decision.base import decision_repli
    from irl_guard.evaluation import rapport

    class LayaQuiDecroche:
        def __init__(self, cfg):
            pass

        def verifier(self):
            pass

        def decide(self, state):
            return decision_repli()

    monkeypatch.setattr(rapport, "LayaDecider", LayaQuiDecroche)
    jeux = generer_jeux(0, PETIT)
    assert obtenir_predictions("laya", SessionConfig(charte="x"), tmp_path, tmp_path,
                               {"test_vues": jeux["test_vues"]}) == {"test_vues": None}


def test_fichier_de_predictions_incomplet_donne_absent(tmp_path):
    from typer.testing import CliRunner

    from irl_guard import cli
    ecrire_jeux(generer_jeux(0, PETIT), tmp_path / "decision")
    dossier = tmp_path / "predictions" / "qwen-ft"
    dossier.mkdir(parents=True)
    (dossier / "predictions_test_vues.jsonl").write_text(json.dumps({"id": "inconnu", "answers": {}}) + "\n",
                                                         encoding="utf-8")
    config = tmp_path / "s.yaml"
    config.write_text("charte: x\n", encoding="utf-8")
    r = CliRunner().invoke(cli.app, ["evaluer", "--donnees", str(tmp_path / "decision"),
                                     "--predictions", str(tmp_path / "predictions"),
                                     "--approches", "regles,qwen-ft", "--config", str(config),
                                     "--sortie", str(tmp_path / "resultats")])
    assert r.exit_code == 0, r.output
    texte = (tmp_path / "resultats" / "rapport.md").read_text(encoding="utf-8")
    assert "| qwen-ft | absent |" in texte


def _rep(action):
    p = {a: (1.0 if a == action else 0.0) for a in ("diffuser", "flouter", "pause")}
    return {"action": {"probabilities": p}, "risque": {"probabilities": {"aucun": 1.0}}, "gravite": {"score": 0}}


def test_combine_fuite_jamais_pire_que_chaque_source():
    from irl_guard.donnees.lignes import construire_gold
    from irl_guard.evaluation.metriques import evaluer_predictions
    from irl_guard.evaluation.predicteurs import combiner_predictions

    cfg = SessionConfig(charte="x")
    attendus = ["flouter", "pause", "diffuser", "flouter", "pause", "flouter"]
    regles = ["diffuser", "pause", "diffuser", "flouter", "diffuser", "diffuser"]
    modele = ["flouter", "diffuser", "diffuser", "diffuser", "diffuser", "diffuser"]
    lignes = [{"id": str(i), "gold": json.dumps(construire_gold(a, "localisation"))} for i, a in enumerate(attendus)]
    p_regles = {str(i): _rep(a) for i, a in enumerate(regles)}
    p_modele = {str(i): _rep(a) for i, a in enumerate(modele)}
    combine = combiner_predictions(p_regles, p_modele, cfg.seuils)
    fuite = lambda p: evaluer_predictions(lignes, p, cfg.seuils, cfg.evaluation.couts).taux_fuite
    assert fuite(combine) <= min(fuite(p_regles), fuite(p_modele))
    assert combine["0"] is p_modele["0"] and combine["1"] is p_regles["1"]


def test_combine_depuis_fichiers(tmp_path):
    jeux = generer_jeux(0, PETIT)
    dossier = tmp_path / "predictions" / "laya-ft"
    dossier.mkdir(parents=True)
    with (dossier / "predictions_test_vues.jsonl").open("w", encoding="utf-8") as f:
        for l in jeux["test_vues"]:
            f.write(json.dumps({"id": l["id"], "answers": _rep("diffuser")}) + "\n")
    preds = obtenir_predictions("combine-laya", SessionConfig(charte="x"), tmp_path, tmp_path / "predictions",
                                {j: jeux[j] for j in ("test_vues", "test_non_vues")})
    assert len(preds["test_vues"]) == 10 and preds["test_non_vues"] is None


def test_combine_avec_predictions_modele_incompletes(tmp_path):
    from typer.testing import CliRunner

    from irl_guard import cli
    jeux = generer_jeux(0, PETIT)
    ecrire_jeux(jeux, tmp_path / "decision")
    dossier = tmp_path / "predictions" / "qwen-ft"
    dossier.mkdir(parents=True)
    with (dossier / "predictions_test_vues.jsonl").open("w", encoding="utf-8") as f:
        for l in jeux["test_vues"][:3]:  # seulement 3 lignes sur 10
            f.write(json.dumps({"id": l["id"], "answers": _rep("pause")}) + "\n")
    config = tmp_path / "s.yaml"
    config.write_text("charte: x\n", encoding="utf-8")
    r = CliRunner().invoke(cli.app, ["evaluer", "--donnees", str(tmp_path / "decision"),
                                     "--predictions", str(tmp_path / "predictions"),
                                     "--approches", "regles,combine-qwen", "--config", str(config),
                                     "--sortie", str(tmp_path / "resultats")])
    assert r.exit_code == 0, r.output
    assert "| combine-qwen | absent |" in (tmp_path / "resultats" / "rapport.md").read_text(encoding="utf-8")
