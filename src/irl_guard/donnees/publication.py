"""Publication des jeux JSONL en dataset privé sur le Hub Hugging Face."""

from __future__ import annotations

from pathlib import Path

from irl_guard.donnees.lignes import JEUX


def construire_datasetdict(dossier: str | Path, cache_dir: str | Path | None = None):
    from datasets import load_dataset
    fichiers = {nom: str(Path(dossier) / f"{nom}.jsonl") for nom in JEUX}
    return load_dataset("json", data_files=fichiers, cache_dir=str(cache_dir) if cache_dir else None)


def publier(dossier: str | Path, repo: str) -> None:
    construire_datasetdict(dossier).push_to_hub(repo, private=True)
