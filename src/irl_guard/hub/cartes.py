"""Fiches Hugging Face (dataset et modèles), générées depuis le rapport pour rester cohérentes avec lui."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

from irl_guard.donnees.lignes import JEUX, lire_jeu

DATASET = "MissawB/irl-guard-decisions"
COLONNES = ("Approche", "Action", "F1 macro", "Fuite", "ECE", "Coût moyen")
NOMS = {"toujours-diffuser": "Toujours diffuser (plancher)", "regles": "Règles", "xgboost": "XGBoost",
        "laya": "Laya brut", "laya-ft": "Laya fine-tuné", "qwen-ft": "Qwen3.5-0.8B fine-tuné",
        "combine-laya": "Règles + Laya fine-tuné", "combine-qwen": "Règles + Qwen fine-tuné",
        "laya-ft-v2": "Laya fine-tuné (v2)", "qwen-ft-v2": "Qwen3.5-2B fine-tuné (v2)",
        "combine-laya-v2": "Règles + Laya v2", "combine-qwen-v2": "Règles + Qwen v2"}
LIMITES = """## Limites

- Les chartes « non vues » sont des formulations et des combinaisons de clauses absentes de l'entraînement, mais
  écrites dans le même style de gabarits : une charte rédigée librement par un streamer sera plus difficile.
- Seul, le modèle laisse passer environ 14 % des cas à protéger sur les chartes non vues : dans IRL Guard, il est
  utilisé en **mode combiné** avec des règles déterministes (la décision la plus protectrice l'emporte).
- Le modèle décide à partir d'un `state` JSON (texte lu par OCR, objet détecté, charte, contexte) : il ne voit pas
  l'image.
"""


def _frontmatter(meta: dict) -> str:
    return "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + "---\n\n"


def lire_tableaux(rapport_md: str) -> dict[str, list[dict[str, str]]]:
    tableaux: dict[str, list[dict[str, str]]] = {}
    section, entetes = None, None
    for ligne in rapport_md.splitlines():
        if ligne.startswith("## "):
            section, entetes = ligne[3:].strip(), None
            tableaux[section] = []
        elif ligne.startswith("|") and section is not None:
            cellules = [c.strip() for c in ligne.strip().strip("|").split("|")]
            if entetes is None:
                entetes = cellules
            elif not set("".join(cellules)) <= set("-: "):
                tableaux[section].append(dict(zip(entetes, cellules)))
    return {s: lignes for s, lignes in tableaux.items() if lignes}


def stats_dataset(dossier: str | Path) -> dict:
    tailles, actions = {}, {}
    for jeu in JEUX:
        lignes = lire_jeu(Path(dossier) / f"{jeu}.jsonl")
        tailles[jeu] = len(lignes)
        compte = Counter(json.loads(l["gold"])["action"]["label"] for l in lignes)
        actions[jeu] = {a: compte[a] / len(lignes) for a in ("diffuser", "flouter", "pause")} if lignes else {}
    return {"tailles": tailles, "actions": actions}


def _tableau_resultats(tableaux: dict[str, list[dict[str, str]]]) -> str:
    """Première colonne = identifiant brut de l'approche, nom lisible dans la légende."""
    blocs = []
    for section, lignes in tableaux.items():
        presentes = [l for l in lignes if l.get("n") != "absent"]
        legende = ", ".join(f"`{l['Approche']}` : {NOMS.get(l['Approche'], l['Approche'])}" for l in presentes)
        blocs += [f"### {section}", "", legende, "", "| " + " | ".join(COLONNES) + " |", "|" + "---|" * len(COLONNES)]
        blocs += ["| " + " | ".join(l.get(c, "") for c in COLONNES) + " |" for l in presentes]
        blocs.append("")
    return "\n".join(blocs)


def carte_dataset(stats: dict) -> str:
    meta = {"license": "cc-by-4.0", "language": ["fr"], "task_categories": ["text-classification"],
            "size_categories": ["1K<n<10K"], "pretty_name": "IRL Guard — décisions guidées par une charte",
            "tags": ["structured-decisions", "content-moderation", "live-streaming", "synthetic"]}
    tailles = "\n".join(f"| `{jeu}` | {n} |" for jeu, n in stats["tailles"].items())
    repartition = ", ".join(f"{a} {100 * p:.0f} %" for a, p in stats["actions"].get("train", {}).items())
    return _frontmatter(meta) + f"""# IRL Guard — décisions guidées par une charte

Jeu synthétique pour entraîner un *decision model* à appliquer la charte d'une chaîne de live IRL : pour un élément
vu à l'écran (texte lu par OCR ou objet détecté), faut-il le **diffuser**, le **flouter** ou mettre le live en
**pause** ? Chaque ligne suit le format de [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
(`state`, `questions`, `gold`) et sert à fine-tuner avec `FastDecisionModel` d'Unsloth.

> ⚠️ **Avertissement** : le jeu contient des insultes (pour apprendre à les masquer) et des **données personnelles
> fictives** (noms, e-mails, téléphones) générées par Faker. Aucune donnée réelle.

## Génération

- Une **charte** assemble 2 à 5 **clauses** structurées (risque, sanction, condition : toujours, en mode domicile,
  sauf dans une catégorie de stream), rendues en français par des gabarits.
- Le **label** se calcule de façon déterministe à partir des clauses : la clause applicable la plus protectrice
  l'emporte ; un risque non mentionné est diffusé.
- La **cible** est tirée en fonction de la charte et un bruit d'OCR est simulé (confusions, casse, troncature).

## Jeux

| Jeu | Lignes |
|---|---|
{tailles}

Répartition des actions attendues (train) : {repartition}.

`test_non_vues` n'utilise que des formulations contenant au moins un morceau de gabarit réservé, et des
combinaisons de clauses absentes de l'entraînement : il mesure si un modèle applique une charte **jamais vue**.

## Projet

Généré par `irl-guard donnees generer` (projet IRL Guard, bloqueur de contenu pour lives IRL). Modèles entraînés :
[`MissawB/irl-guard-decision-qwen`](https://huggingface.co/MissawB/irl-guard-decision-qwen) et
[`MissawB/irl-guard-decision-laya`](https://huggingface.co/MissawB/irl-guard-decision-laya).
"""


def _carte_modele(base: str, titre: str, usage: str, tableaux: dict, library: str) -> str:
    meta = {"license": "apache-2.0", "base_model": base, "datasets": [DATASET], "language": ["fr"],
            "library_name": library, "pipeline_tag": "text-classification",
            "tags": ["structured-decisions", "content-moderation", "live-streaming", "irl-guard"]}
    return _frontmatter(meta) + f"""# {titre}

*Decision model* d'IRL Guard : à partir d'un `state` JSON (charte de la chaîne en langage naturel, catégorie du
stream, mode domicile, élément vu à l'écran), il répond à trois questions typées — **action**
(diffuser / flouter / pause), **risque** et **gravité** — avec des probabilités calibrées. Fine-tuné à partir de
`{base}` sur [`{DATASET}`](https://huggingface.co/datasets/{DATASET}).

## Utilisation

{usage}

## Résultats

Précision de l'action, F1 macro, taux de fuite (cas à protéger que l'approche diffuse), calibration (ECE) et coût
moyen par décision, sur 1 000 cas par jeu.

{_tableau_resultats(tableaux)}
{LIMITES}"""


def carte_qwen(tableaux: dict) -> str:
    usage = """Servi par [Unsloth Studio](https://unsloth.ai) (Decision API, `POST /v1/systemone`) :

```bash
UNSLOTH_SYSTEMONE_MODEL=<chemin du modèle téléchargé> unsloth studio
```

ou chargé avec `transformers` grâce au fichier `joint_schema_model.py` du dépôt :

```python
from joint_schema_model import load_release_model, systemone
model, processor = load_release_model("MissawB/irl-guard-decision-qwen", device="cuda")
reponse = systemone(model, processor, {"model": "irl-guard", "state": state, "questions": questions})
```

Dans IRL Guard : `decideur.modele: studio` puis `irl-guard process clip.mp4 --decider combine`."""
    return _carte_modele("unsloth/Qwen3.5-0.8B", "IRL Guard — decision model (Qwen3.5-0.8B fine-tuné)",
                         usage, tableaux, "unsloth")


def carte_laya(tableaux: dict) -> str:
    usage = """Sur CPU, sans serveur, avec le paquet [`laya`](https://pypi.org/project/laya/) :

```python
import laya
agent = laya.load("MissawB/irl-guard-decision-laya")
reponse = agent.predict(state, questions)
```

C'est le modèle par défaut d'IRL Guard (`--decider combine`, `decideur.modele: laya-local`)."""
    return _carte_modele("convaiinnovations/laya-multilingual", "IRL Guard — decision model (Laya fine-tuné)",
                         usage, tableaux, "laya")


def carte_qwen_lora() -> str:
    meta = {"license": "apache-2.0", "base_model": "unsloth/Qwen3.5-0.8B", "datasets": [DATASET],
            "library_name": "peft", "tags": ["lora", "irl-guard"]}
    return _frontmatter(meta) + """# IRL Guard — adaptateurs LoRA (Qwen3.5-0.8B)

Adaptateurs LoRA et tête de décision du modèle
[`MissawB/irl-guard-decision-qwen`](https://huggingface.co/MissawB/irl-guard-decision-qwen). Pour l'inférence,
utilisez plutôt le modèle fusionné ; résultats et limites sur sa fiche.
"""
