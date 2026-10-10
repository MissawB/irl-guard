import cv2
import yaml
from typer.testing import CliRunner

from irl_guard import cli
from irl_guard.types import Box, Detection

runner = CliRunner()


class TexteFixe:
    def detect(self, frame_index, image):
        return [Detection("texte", 0.95, Box(20, 100, 280, 145), "Rue Nationale")]


def _dessiner(image, i):
    cv2.putText(image, "Rue Nationale", (25, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)


def _config(tmp_path, **valeurs):
    chemin = tmp_path / "session.yaml"
    chemin.write_text(yaml.safe_dump({"charte": "Jamais de nom de rue.", "fps_analyse": 5.0, **valeurs}),
                      encoding="utf-8")
    return chemin


def test_process_puis_replay(make_video, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "construire_perception", lambda cfg: TexteFixe())
    entree = make_video(n=20, fps=10.0, size=(640, 240), draw=_dessiner)
    config = _config(tmp_path)

    r = runner.invoke(cli.app, ["process", str(entree), "--config", str(config),
                                "--out", str(tmp_path / "out.mp4"), "--decider", "rules", "--debug-overlay"])
    assert r.exit_code == 0, r.output
    assert "temps réel" in r.output
    assert (tmp_path / "out.mp4").exists() and (tmp_path / "out_debug.mp4").exists()
    assert (tmp_path / "out.jsonl").exists()

    r = runner.invoke(cli.app, ["replay", str(tmp_path / "out.jsonl"), str(entree), "--config", str(config),
                                "--out", str(tmp_path / "rejeu.mp4"), "--decider", "rules"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "rejeu.mp4").exists()


def test_replay_journal_incompatible(make_video, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "construire_perception", lambda cfg: TexteFixe())
    entree = make_video(n=20, fps=10.0, size=(640, 240), draw=_dessiner)
    runner.invoke(cli.app, ["process", str(entree), "--config", str(_config(tmp_path)),
                            "--out", str(tmp_path / "out.mp4"), "--decider", "rules"])
    autre = _config(tmp_path, fps_analyse=2.0)
    r = runner.invoke(cli.app, ["replay", str(tmp_path / "out.jsonl"), str(entree), "--config", str(autre),
                                "--out", str(tmp_path / "rejeu.mp4"), "--decider", "rules"])
    assert r.exit_code == 4
    assert "fps_analyse" in r.output


def test_replay_refuse_d_ecraser_le_journal(make_video, tmp_path):
    entree = make_video(n=5)
    journal = tmp_path / "out.jsonl"
    journal.write_text("", encoding="utf-8")
    r = runner.invoke(cli.app, ["replay", str(journal), str(entree), "--config", str(_config(tmp_path)),
                                "--out", str(tmp_path / "out.mp4"), "--decider", "rules"])
    assert r.exit_code == 1


def test_laya_injoignable(make_video, tmp_path):
    entree = make_video(n=5)
    config = _config(tmp_path, laya={"url": "http://127.0.0.1:9", "timeout_s": 0.5})
    r = runner.invoke(cli.app, ["process", str(entree), "--config", str(config), "--out", str(tmp_path / "o.mp4"),
                                "--decider", "laya"])
    assert r.exit_code == 2
    assert "--decider rules" in r.output


class FauxAgentLaya:
    def predict(self, state, questions):
        return {"answers": {
            "action": {"probabilities": {"diffuser": 0.9, "flouter": 0.05, "pause": 0.05}},
            "risque": {"probabilities": {"localisation": 0.2, "donnee_perso": 0.2, "arme": 0.2,
                                         "propos": 0.2, "aucun": 0.2}},
            "gravite": {"score": 0, "probabilities": {"0": 1.0, "1": 0, "2": 0, "3": 0}},
        }}


def test_combine_est_le_defaut_et_garde_la_regle(make_video, tmp_path, monkeypatch):
    import json

    from irl_guard.decision import qwen_local
    monkeypatch.setattr(qwen_local, "_charger_qwen",
                        lambda repo, revision=None: lambda requete: FauxAgentLaya().predict(None, None))
    monkeypatch.setattr(cli, "construire_perception", lambda cfg: TexteFixe())
    entree = make_video(n=10, fps=10.0, size=(640, 240), draw=_dessiner)
    r = runner.invoke(cli.app, ["process", str(entree), "--config", str(_config(tmp_path)),
                                "--out", str(tmp_path / "out.mp4")])
    assert r.exit_code == 0, r.output
    lignes = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    appel = json.loads(lignes[0])["analyse"]["appels"][0]
    # Le modèle factice diffuse, la règle floute « Rue Nationale » : la règle l'emporte.
    assert appel["decision"]["action"] == "flouter"
    assert appel["decision"]["source"] == "combine:rules"


def test_laya_local_seul(make_video, tmp_path, monkeypatch):
    from irl_guard.decision import laya_local
    monkeypatch.setattr(laya_local, "_charger_laya", lambda modele, revision=None: FauxAgentLaya())
    monkeypatch.setattr(cli, "construire_perception", lambda cfg: TexteFixe())
    entree = make_video(n=5, fps=10.0, size=(640, 240), draw=_dessiner)
    r = runner.invoke(cli.app, ["process", str(entree), "--config", str(_config(tmp_path)),
                                "--out", str(tmp_path / "out.mp4"), "--decider", "laya-local"])
    assert r.exit_code == 0, r.output


def test_config_decideur_par_defaut():
    import re

    from irl_guard.config import SessionConfig
    cfg = SessionConfig(charte="x")
    assert cfg.decideur.modele == "qwen-local"
    assert cfg.decideur.qwen_local == "MissawB/irl-guard-decision-qwen-v2"
    assert re.fullmatch(r"[0-9a-f]{40}", cfg.decideur.qwen_revision)
    assert cfg.decideur.laya_local == "MissawB/irl-guard-decision-laya-v2"


def test_combine_avec_laya_local_par_la_config(make_video, tmp_path, monkeypatch):
    from irl_guard.decision import laya_local
    monkeypatch.setattr(laya_local, "_charger_laya", lambda modele, revision=None: FauxAgentLaya())
    monkeypatch.setattr(cli, "construire_perception", lambda cfg: TexteFixe())
    entree = make_video(n=5, fps=10.0, size=(640, 240), draw=_dessiner)
    r = runner.invoke(cli.app, ["process", str(entree), "--config",
                                str(_config(tmp_path, decideur={"modele": "laya-local"})),
                                "--out", str(tmp_path / "out.mp4")])
    assert r.exit_code == 0, r.output
