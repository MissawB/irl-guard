"""Construction locale et publication de la vitrine Hugging Face (fiches, Space, visibilité, Collection)."""

from __future__ import annotations

import shutil
from pathlib import Path

from irl_guard.hub.cartes import (
    carte_dataset, carte_laya, carte_qwen, carte_qwen_lora, lire_tableaux, stats_dataset,
)

DEPOTS_PUBLICS = [
    ("MissawB/irl-guard-decisions", "dataset"),
    ("MissawB/irl-guard-decision-qwen", "model"),
    ("MissawB/irl-guard-decision-qwen-lora", "model"),
    ("MissawB/irl-guard-decision-laya", "model"),
]
SPACE = "MissawB/irl-guard-demo"
COLLECTION = "IRL Guard"
DESCRIPTION_COLLECTION = ("Bloqueur de contenu pour lives IRL : dataset de décisions guidées par une charte, "
                          "decision models fine-tunés et démo.")
# Le Space n'a besoin que de la décision et de la config : pas de perception vidéo ni d'évaluation.
INCLUS_SPACE = ("__init__.py", "config.py", "decision")


def copier_paquet_space(source: Path, destination: Path) -> None:
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    for nom in INCLUS_SPACE:
        chemin = source / nom
        if chemin.is_dir():
            shutil.copytree(chemin, destination / nom, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(chemin, destination / nom)


def construire(dossier_build: Path, rapport: Path, donnees: Path, source_space: Path,
               source_paquet: Path) -> dict[str, Path]:
    tableaux = lire_tableaux(Path(rapport).read_text(encoding="utf-8"))
    cartes = {
        "MissawB/irl-guard-decisions": carte_dataset(stats_dataset(donnees)),
        "MissawB/irl-guard-decision-qwen": carte_qwen(tableaux),
        "MissawB/irl-guard-decision-qwen-lora": carte_qwen_lora(),
        "MissawB/irl-guard-decision-laya": carte_laya(tableaux),
    }
    construits: dict[str, Path] = {}
    for repo, carte in cartes.items():
        chemin = Path(dossier_build) / repo.split("/")[1] / "README.md"
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text(carte, encoding="utf-8")
        construits[repo] = chemin
    space = Path(dossier_build) / "space"
    shutil.rmtree(space, ignore_errors=True)
    shutil.copytree(source_space, space, ignore=shutil.ignore_patterns("__pycache__"))
    copier_paquet_space(Path(source_paquet), space / "irl_guard")
    construits[SPACE] = space
    return construits


def publier(api, construits: dict[str, Path], public: bool) -> list[str]:
    journal = []
    for repo, repo_type in DEPOTS_PUBLICS:
        api.upload_file(path_or_fileobj=str(construits[repo]), path_in_repo="README.md", repo_id=repo,
                        repo_type=repo_type, commit_message="Fiche IRL Guard")
        journal.append(f"fiche envoyée : {repo}")
    api.create_repo(SPACE, repo_type="space", space_sdk="gradio", private=True, exist_ok=True,
                    space_hardware="zero-a10g")  # sur CPU gratuit, un compte non PRO reçoit une 402
    api.upload_folder(folder_path=str(construits[SPACE]), repo_id=SPACE, repo_type="space",
                      commit_message="Space IRL Guard")
    journal.append(f"Space déployé (ZeroGPU) : {SPACE}")
    if not public:
        journal.append("visibilité inchangée (relancer avec --public pour publier)")
        return journal
    elements = DEPOTS_PUBLICS + [(SPACE, "space")]
    for repo, repo_type in elements:
        api.update_repo_settings(repo, private=False, repo_type=repo_type)
        journal.append(f"public : {repo}")
    collection = api.create_collection(title=COLLECTION, description=DESCRIPTION_COLLECTION, private=False,
                                       exists_ok=True)
    for repo, repo_type in elements:
        api.add_collection_item(collection.slug, item_id=repo, item_type=repo_type, exists_ok=True)
    journal.append(f"Collection : https://huggingface.co/collections/{collection.slug}")
    return journal
