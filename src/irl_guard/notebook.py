"""Génère le notebook Colab de fine-tuning (`python -m irl_guard.notebook`).

Le notebook est construit ici pour rester testé et versionné proprement : aucune logique métier
dedans, seulement l'enchaînement Unsloth décrit dans la spec §3.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

CELLULES = [
    ("md", """# IRL Guard — fine-tuning du decision model

Entraîne `MODELE` sur le dataset privé `MissawB/irl-guard-decisions`, calibre, publie le modèle
et exporte les prédictions des deux jeux de test.

**Avant de lancer :** Exécution → Modifier le type d'exécution → GPU T4, et ajouter le secret
`HF_TOKEN` (jeton Hugging Face en écriture) dans le panneau 🔑 de Colab."""),
    ("code", """MODELE = "qwen"  # "qwen" ou "laya"
VERSION = "v2"   # version du dataset et des modèles : un nouveau nom évite de reprendre un ancien entraînement
MAX_STEPS = -1   # -1 = 2 époques complètes ; 60 pour un essai rapide
REPO_DONNEES = f"MissawB/irl-guard-decisions-{VERSION}"
REPO_MODELE = f"MissawB/irl-guard-decision-{MODELE}-{VERSION}"
BASES = {"qwen": "unsloth/Qwen3.5-2B", "laya": "convaiinnovations/laya-multilingual"}"""),
    ("code", """!pip install -q --upgrade unsloth datasets"""),
    ("code", """from google.colab import userdata
from huggingface_hub import login

login(userdata.get("HF_TOKEN"))"""),
    ("md", """## 0. Vérification : le modèle se charge-t-il avec `FastDecisionModel` ?
Si cette cellule échoue pour Laya, la branche Laya est abandonnée : on le note et on passe à Qwen."""),
    ("code", """from unsloth import FastDecisionModel

try:
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=BASES[MODELE], max_seq_length=1024, load_in_4bit=(MODELE == "qwen"))
    print("Chargement OK :", BASES[MODELE])
except Exception as e:
    raise SystemExit(f"{BASES[MODELE]} ne se charge pas avec FastDecisionModel : {e}")"""),
    ("code", """model = FastDecisionModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    use_gradient_checkpointing="unsloth", random_state=3407)"""),
    ("md", """## 1. Données"""),
    ("code", """from datasets import load_dataset

ds = load_dataset(REPO_DONNEES)
train_items, rapport = FastDecisionModel.build_dataset(ds["train"], tokenizer, model)
print(f"train : {rapport['skipped']} lignes ignorées sur {rapport['total']}")
eval_items, rapport_val = FastDecisionModel.build_dataset(ds["val"], tokenizer, model)
print(f"val : {rapport_val['skipped']} lignes ignorées sur {rapport_val['total']}")
assert rapport["skipped"] == 0, "Des lignes sont ignorées : vérifier le format avant d'entraîner." """),
    ("md", """## 2. Entraînement (reprise automatique depuis le dernier checkpoint publié)"""),
    ("code", """import os

from huggingface_hub import repo_exists, snapshot_download
from transformers import TrainingArguments
from unsloth import DecisionTrainer

REPO_CHECKPOINTS = REPO_MODELE + "-checkpoints"
args = TrainingArguments(
    output_dir="sorties",
    per_device_train_batch_size=4,   # 2B sur T4 : batch 4 x accumulation 8 = 32, comme en v1
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    max_steps=MAX_STEPS,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",
    push_to_hub=True,
    hub_model_id=REPO_CHECKPOINTS,
    hub_private_repo=True,
    hub_strategy="checkpoint",
)
trainer = DecisionTrainer(model=model, processing_class=tokenizer,
                          train_dataset=train_items, eval_dataset=eval_items, args=args)

reprise = None
if repo_exists(REPO_CHECKPOINTS):
    snapshot_download(REPO_CHECKPOINTS, local_dir="sorties", allow_patterns="last-checkpoint/*")
    if os.path.isdir("sorties/last-checkpoint"):
        reprise = "sorties/last-checkpoint"
        print("Reprise depuis", reprise)
trainer.train(resume_from_checkpoint=reprise)"""),
    ("md", """## 3. Calibration sur le jeu de validation"""),
    ("code", """metriques = FastDecisionModel.calibrate(model, tokenizer, eval_items)
print(metriques)  # précision, ECE et loss sur val"""),
    ("md", """## 4. Publication (dépôts privés)"""),
    ("code", """import json
import os
import shutil

from huggingface_hub import HfApi

api = HfApi()
DOSSIER_LORA, DOSSIER_MERGED = f"lora-{MODELE}", f"merged-{MODELE}"
for dossier in (DOSSIER_LORA, DOSSIER_MERGED):
    shutil.rmtree(dossier, ignore_errors=True)  # jamais de fichiers d'un entraînement précédent
sauvegardes = [(DOSSIER_MERGED, REPO_MODELE)]
if hasattr(model, "save_pretrained"):  # Laya n'a pas d'adaptateurs LoRA séparés, seulement le modèle fusionné
    model.save_pretrained(DOSSIER_LORA)
    sauvegardes.append((DOSSIER_LORA, REPO_MODELE + "-lora"))
model.save_pretrained_merged(DOSSIER_MERGED)

fichiers = set(os.listdir(DOSSIER_MERGED))
if MODELE == "laya":  # Laya fusionné : rl_agent_config.json, encoder/, tokenizer/ (pas de config.json)
    assert "rl_agent_config.json" in fichiers and not any(f.startswith("model.safetensors-") for f in fichiers)
    with open(os.path.join(DOSSIER_MERGED, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "laya", "architectures": ["LayaTypedDecisions"]}, f, indent=1)
else:
    with open(os.path.join(DOSSIER_MERGED, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    assert config["model_type"].startswith("qwen"), f"config.json inattendu : {config['model_type']}"

for dossier, repo in sauvegardes:
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=dossier, repo_id=repo)
    print("Publié :", repo, sorted(api.list_repo_files(repo)))"""),
    ("md", """## 5. Prédictions sur les jeux de test
La première réponse brute est affichée : si son format diffère de ce que `vers_api` attend,
s'arrêter et rapporter cette sortie."""),
    ("code", """import json

FastDecisionModel.for_inference(model)
premiere = ds["test_vues"][0]
brute = FastDecisionModel.predict(model, tokenizer, premiere["state"], json.loads(premiere["questions"]))
print(brute)"""),
    ("code", """def vers_api(reponse, questions):
    \"\"\"Convertit une réponse de predict au format de la Decision API (probabilities, score).\"\"\"
    sortie = {}
    for nom, q in questions.items():
        probas = {str(k): float(v) for k, v in reponse[nom]["probabilities"].items()}
        if q["type"] == "score":
            index = {str(i): i for i in range(len(q["criteria"]))}
            index.update({niveau: i for i, niveau in enumerate(q["criteria"])})
            sortie[nom] = {"score": sum(index[k] * p for k, p in probas.items()), "probabilities": probas}
        else:
            sortie[nom] = {"probabilities": probas}
    return sortie


for jeu in ("test_vues", "test_non_vues"):
    nom_fichier = f"predictions_{jeu}.jsonl"
    with open(nom_fichier, "w", encoding="utf-8") as f:
        for ligne in ds[jeu]:
            questions = json.loads(ligne["questions"])
            reponse = FastDecisionModel.predict(model, tokenizer, ligne["state"], questions)
            f.write(json.dumps({"id": ligne["id"], "answers": vers_api(reponse, questions)}, ensure_ascii=False) + "\\n")
    api.upload_file(path_or_fileobj=nom_fichier, path_in_repo=nom_fichier, repo_id=REPO_MODELE)
    print("Prédictions publiées :", nom_fichier)"""),
]


def construire_notebook() -> nbformat.NotebookNode:
    cellules = [new_markdown_cell(src) if kind == "md" else new_code_cell(src) for kind, src in CELLULES]
    nb = new_notebook(cells=cellules)
    nb.metadata.update({"accelerator": "GPU", "colab": {"provenance": []},
                        "kernelspec": {"name": "python3", "display_name": "Python 3"}})
    return nb


def ecrire_notebook(chemin: str | Path) -> Path:
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(construire_notebook(), chemin)
    return chemin


if __name__ == "__main__":
    print(ecrire_notebook("notebooks/finetune_decision.ipynb"))
