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
    r = runner.invoke(cli.app, ["process", str(entree), "--config", str(config), "--out", str(tmp_path / "o.mp4")])
    assert r.exit_code == 2
    assert "--decider rules" in r.output
