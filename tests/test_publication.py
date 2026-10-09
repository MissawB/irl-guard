from irl_guard.donnees import publication
from irl_guard.donnees.lignes import ecrire_jeux, generer_jeux


def test_datasetdict(tmp_path):
    ecrire_jeux(generer_jeux(0, {"train": 8, "val": 2, "test_vues": 2, "test_non_vues": 2}), tmp_path / "d")
    dd = publication.construire_datasetdict(tmp_path / "d", cache_dir=tmp_path / "cache")
    assert set(dd) == {"train", "val", "test_vues", "test_non_vues"}
    assert dd["train"].num_rows == 8
    assert set(dd["train"].column_names) == {"id", "state", "questions", "gold", "clauses"}


def test_publier_en_prive(tmp_path, monkeypatch):
    appels = {}

    class Faux:
        def push_to_hub(self, repo, private):
            appels.update(repo=repo, private=private)

    monkeypatch.setattr(publication, "construire_datasetdict", lambda dossier, cache_dir=None: Faux())
    publication.publier(tmp_path, "MissawB/irl-guard-decisions")
    assert appels == {"repo": "MissawB/irl-guard-decisions", "private": True}
