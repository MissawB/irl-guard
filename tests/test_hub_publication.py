import subprocess
import sys
from pathlib import Path

from irl_guard.donnees.lignes import ecrire_jeux, generer_jeux
from irl_guard.hub import publication

RAPPORT = Path("docs/resultats/rapport.md")


class FauxApi:
    def __init__(self):
        self.appels = []

    def __getattr__(self, nom):
        def appel(*args, **kwargs):
            self.appels.append((nom, args, kwargs))

            class Collection:
                slug = "MissawB/irl-guard-123"
            return Collection()
        return appel

    def noms(self):
        return [a[0] for a in self.appels]


def _construire(tmp_path):
    ecrire_jeux(generer_jeux(0, {"train": 10, "val": 2, "test_vues": 2, "test_non_vues": 2}), tmp_path / "d")
    return publication.construire(tmp_path / "build", RAPPORT, tmp_path / "d", Path("space"), Path("src/irl_guard"))


def test_copie_filtree_du_paquet(tmp_path):
    publication.copier_paquet_space(Path("src/irl_guard"), tmp_path / "irl_guard")
    fichiers = {p.relative_to(tmp_path / "irl_guard").as_posix() for p in (tmp_path / "irl_guard").rglob("*.py")}
    assert "config.py" in fichiers and "decision/combine.py" in fichiers
    assert not any(f.startswith(("perception", "evaluation", "donnees", "hub")) for f in fichiers)
    assert "cli.py" not in fichiers and "pipeline.py" not in fichiers
    r = subprocess.run([sys.executable, "-c", "import irl_guard.decision.combine, irl_guard.decision.rules"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_construire(tmp_path):
    construits = _construire(tmp_path)
    assert set(construits) == {r for r, _ in publication.DEPOTS_PUBLICS} | {publication.SPACE}
    for repo, chemin in construits.items():
        if repo != publication.SPACE:
            assert chemin.name == "README.md" and chemin.read_text(encoding="utf-8").startswith("---")
    space = construits[publication.SPACE]
    assert (space / "app.py").exists() and (space / "irl_guard" / "decision" / "combine.py").exists()


def test_sans_public_rien_ne_devient_public(tmp_path):
    api = FauxApi()
    publication.publier(api, _construire(tmp_path), public=False)
    assert "update_repo_settings" not in api.noms() and "create_collection" not in api.noms()
    creation_space = [a for a in api.appels if a[0] == "create_repo"][0]
    assert creation_space[2]["private"] is True
    # Un compte gratuit ne peut pas créer de Space Gradio sur CPU (402) : il est créé directement sur ZeroGPU.
    assert creation_space[2]["space_hardware"] == "zero-a10g"
    assert ("request_space_hardware", (publication.SPACE, "zero-a10g"), {}) in api.appels


def test_public_ne_touche_que_les_depots_listes(tmp_path):
    api = FauxApi()
    publication.publier(api, _construire(tmp_path), public=True)
    rendus_publics = [a[1][0] for a in api.appels if a[0] == "update_repo_settings"]
    assert set(rendus_publics) == {r for r, _ in publication.DEPOTS_PUBLICS} | {publication.SPACE}
    assert not any("checkpoints" in r for r in rendus_publics)
    elements = [a[2]["item_id"] for a in api.appels if a[0] == "add_collection_item"]
    assert set(elements) == {r for r, _ in publication.DEPOTS_PUBLICS} | {publication.SPACE}


def test_cli_dry_run_sans_reseau(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from irl_guard import cli

    def interdit(*a, **k):
        raise AssertionError("aucun appel réseau en --dry-run")

    monkeypatch.setattr(publication, "publier", interdit)
    ecrire_jeux(generer_jeux(0, {"train": 10, "val": 2, "test_vues": 2, "test_non_vues": 2}), tmp_path / "d")
    r = CliRunner().invoke(cli.app, ["hub", "publier", "--dry-run", "--donnees", str(tmp_path / "d"),
                                     "--build", str(tmp_path / "build")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "build" / "irl-guard-decision-laya" / "README.md").exists()


def test_dry_run_dans_une_console_windows_cp1252(tmp_path):
    import os
    ecrire_jeux(generer_jeux(0, {"train": 10, "val": 2, "test_vues": 2, "test_non_vues": 2}), tmp_path / "d")
    r = subprocess.run([sys.executable, "-c", "from irl_guard.cli import app; app()", "hub", "publier", "--dry-run",
                        "--donnees", str(tmp_path / "d"), "--build", str(tmp_path / "build")],
                       capture_output=True, env={**os.environ, "PYTHONIOENCODING": "cp1252"})
    assert r.returncode == 0, r.stderr.decode("cp1252", "replace")[-500:]
