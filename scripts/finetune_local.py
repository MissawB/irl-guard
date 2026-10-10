"""Fine-tuning du decision model sur ce PC (mêmes étapes que notebooks/finetune_decision.ipynb, sans Colab).

À lancer avec un Python qui contient Unsloth, par exemple celui d'Unsloth Studio :
    ~/.unsloth/studio/unsloth_studio/Scripts/python.exe scripts/finetune_local.py --modele laya
Le jeton Hugging Face est celui de `hf auth login` (aucun jeton dans ce fichier).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

BASES = {"qwen": "unsloth/Qwen3.5-2B", "laya": "convaiinnovations/laya-multilingual"}


def vers_api(reponse, questions):
    """Convertit une réponse de predict au format de la Decision API (probabilities, score)."""
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


def verifier_identite(merged: str, modele: str) -> None:
    """Le dossier publié contient bien le modèle annoncé (incident v1 : Qwen envoyé dans le dépôt Laya)."""
    fichiers = set(os.listdir(merged))
    if modele == "laya":
        assert "rl_agent_config.json" in fichiers, f"pas un modèle Laya : {sorted(fichiers)}"
        assert not any(f.startswith("model.safetensors-") for f in fichiers), "fichiers Qwen dans le dossier Laya"
        with open(os.path.join(merged, "config.json"), "w", encoding="utf-8") as f:  # suivi Hub, comme Laya d'origine
            json.dump({"model_type": "laya", "architectures": ["LayaTypedDecisions"]}, f, indent=1)
    else:
        with open(os.path.join(merged, "config.json"), encoding="utf-8") as f:
            model_type = json.load(f)["model_type"]
        assert model_type.startswith("qwen"), f"config.json inattendu : {model_type}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modele", choices=sorted(BASES), required=True)
    parser.add_argument("--version", default="v2")
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 = 2 époques ; 10 pour un essai")
    parser.add_argument("--sortie", default="outputs/finetune")
    parser.add_argument("--publier", action="store_true", help="envoie le modèle et les prédictions sur le Hub")
    args = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import TrainingArguments
    from unsloth import DecisionTrainer, FastDecisionModel

    repo_donnees = f"MissawB/irl-guard-decisions-{args.version}"
    repo_modele = f"MissawB/irl-guard-decision-{args.modele}-{args.version}"
    dossier = os.path.join(args.sortie, f"{args.modele}-{args.version}")

    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=BASES[args.modele], max_seq_length=1024, load_in_4bit=(args.modele == "qwen"))
    model = FastDecisionModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, use_gradient_checkpointing="unsloth", random_state=3407)

    ds = load_dataset(repo_donnees)
    train_items, rapport = FastDecisionModel.build_dataset(ds["train"], tokenizer, model)
    eval_items, _ = FastDecisionModel.build_dataset(ds["val"], tokenizer, model)
    print(f"train : {rapport['skipped']} lignes ignorées sur {rapport['total']}")
    assert rapport["skipped"] == 0, "Des lignes sont ignorées : vérifier le format avant d'entraîner."

    trainer = DecisionTrainer(
        model=model, processing_class=tokenizer, train_dataset=train_items, eval_dataset=eval_items,
        args=TrainingArguments(
            output_dir=os.path.join(dossier, "checkpoints"), per_device_train_batch_size=4,
            gradient_accumulation_steps=8, num_train_epochs=2, max_steps=args.max_steps, learning_rate=2e-4,
            lr_scheduler_type="cosine", warmup_steps=10, weight_decay=0.01, eval_strategy="epoch",
            save_strategy="epoch", logging_steps=10, report_to="none", dataloader_num_workers=0))
    derniers = os.path.join(dossier, "checkpoints")
    reprise = any(n.startswith("checkpoint-") for n in os.listdir(derniers)) if os.path.isdir(derniers) else False
    trainer.train(resume_from_checkpoint=True if reprise else None)
    print("calibration :", FastDecisionModel.calibrate(model, tokenizer, eval_items))

    lora, merged = os.path.join(dossier, "lora"), os.path.join(dossier, "merged")
    for d in (lora, merged):
        shutil.rmtree(d, ignore_errors=True)
    sauvegardes = [(merged, repo_modele)]
    if hasattr(model, "save_pretrained"):  # Laya n'a pas d'adaptateurs LoRA séparés, seulement le modèle fusionné
        model.save_pretrained(lora)
        sauvegardes.append((lora, repo_modele + "-lora"))
    model.save_pretrained_merged(merged)
    verifier_identite(merged, args.modele)

    FastDecisionModel.for_inference(model)
    fichiers = []
    for jeu in ("test_vues", "test_non_vues"):
        chemin = os.path.join(dossier, f"predictions_{jeu}.jsonl")
        with open(chemin, "w", encoding="utf-8") as f:
            for ligne in ds[jeu]:
                questions = json.loads(ligne["questions"])
                reponse = FastDecisionModel.predict(model, tokenizer, ligne["state"], questions)
                f.write(json.dumps({"id": ligne["id"], "answers": vers_api(reponse, questions)}, ensure_ascii=False)
                        + "\n")
        fichiers.append(chemin)
    print("prédictions :", fichiers)

    if args.publier:
        api = HfApi()
        for d, repo in sauvegardes:
            api.create_repo(repo, private=True, exist_ok=True)
            api.upload_folder(folder_path=d, repo_id=repo)
        for chemin in fichiers:
            api.upload_file(path_or_fileobj=chemin, path_in_repo=os.path.basename(chemin), repo_id=repo_modele)
        print("publié (privé) :", repo_modele)


if __name__ == "__main__":
    main()
