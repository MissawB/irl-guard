# IRL Guard

Bloqueur de contenu pour lives IRL. À partir d'une vidéo enregistrée, IRL Guard masque les plaques d'immatriculation, les visages et les textes sensibles (noms de rue, numéros, données personnelles, insultes), et bascule sur un écran de pause quand le contenu est grave.

La perception repose sur des détecteurs pré-entraînés (YOLO, NudeNet, RapidOCR). La décision dépend du contexte : un decision model ([Laya](https://unsloth.ai/docs/models/decision-laya)) lit la charte de la chaîne, écrite en langage naturel, et décide pour chaque élément détecté. À détection identique, « Rue Nationale » est floutée en balade, mais déclenche une pause en mode domicile.

## Installation

```bash
uv sync
```

Laya est servi par l'app Unsloth :

1. Sous Windows, installer l'app avec `irm https://unsloth.ai/install.ps1 | iex`.
2. Dans l'app, ouvrir Settings → API → **Decision API**, activer *Serve requests* et choisir le modèle multilingue.
3. Activer *Keyless API access* pour localhost, ou renseigner `laya.cle` dans la config.

## Utilisation

```bash
uv run irl-guard process clip.mp4 --config configs/exemple.yaml --out sorties/clip_protege.mp4 --debug-overlay
uv run irl-guard process clip.mp4 --config configs/exemple.yaml --out sorties/clip_regles.mp4 --decider rules
uv run irl-guard replay sorties/clip_protege.jsonl clip.mp4 --config autre_reglage.yaml --out sorties/rejeu.mp4
```

Fichiers produits :

- la vidéo protégée ;
- `<out>.jsonl`, le journal (détections, `state` envoyés, décisions) ;
- avec `--debug-overlay`, `<out>_debug.mp4`, qui affiche les boîtes, les pistes, le texte lu et les probabilités.

⚠️ La vidéo de debug affiche le texte OCR en clair : elle sert à l'analyse, jamais à la diffusion.

Les vidéos de téléphone sont souvent à fréquence d'images variable. Le masquage reste correct image par image, mais la coupure audio des pauses peut être décalée. Un avertissement le signale ; convertis alors d'abord la vidéo :

```bash
uv run python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"   # chemin de ffmpeg
<ffmpeg> -i clip.mp4 -fps_mode cfr -r 30 clip_cfr.mp4
```

## Fonctionnement

1. **Échantillonnage.** 3 images par seconde par défaut. Une image presque identique à la précédente réutilise son analyse.
2. **Perception.** YOLO COCO, détecteurs de visages et de plaques, NudeNet et RapidOCR.
3. **Suivi.** ByteTrack par classe, avec le texte OCR accumulé par piste.
4. **Décision.** Les visages et les plaques sont toujours masqués. Les textes et les objets ambigus sont jugés par le décideur, une fois par piste et par texte.
5. **Hystérésis.** La pause commence immédiatement et ne se termine qu'après plusieurs analyses calmes.
6. **Rendu.** Les masques sont interpolés entre deux analyses, et une piste perdue garde son masque quelques analyses. L'audio est coupé pendant les pauses.

## Tests

```bash
uv run pytest                 # tests rapides, sans GPU ni Laya
uv run pytest -m "gpu"        # vrais détecteurs (téléchargement des poids)
uv run pytest -m "slow"       # test anti-fuite avec OCR réel
```

## Limites de la V1

- Traitement hors ligne uniquement : le temps réel (SRT/RTMP) est l'étape suivante.
- Pas de détection d'armes à feu (absentes de COCO), pas d'analyse audio.
- L'OCR peine sur le texte stylisé ou flou.
- Les décisions de Laya ne sont pas encore fine-tunées sur ce domaine.

## Fine-tuning du decision model

Laya sans fine-tuning ne suit pas une charte écrite (voir `docs/resultats/rapport.md`). Le fine-tuning lui apprend à le faire sur des chartes synthétiques.

1. **Données** : chaque charte est composée de clauses structurées, rendues en français par des gabarits. Les labels se calculent de façon déterministe à partir des clauses. Le jeu « chartes non vues » utilise des formulations et des combinaisons de clauses absentes de l'entraînement.
   ```bash
   uv run irl-guard donnees generer
   uv run hf auth login            # jeton en écriture, une seule fois
   uv run irl-guard donnees publier
   ```
2. **Entraînement sur Colab** : ouvrir `notebooks/finetune_decision.ipynb` dans Colab, choisir un GPU T4, ajouter le secret `HF_TOKEN`, régler `MODELE = "qwen"` (puis `"laya"`) et tout exécuter. Si la session est coupée, relancer le notebook : il reprend au dernier checkpoint.
3. **Récupérer les prédictions** :
   ```bash
   uv run hf download MissawB/irl-guard-decision-qwen predictions_test_vues.jsonl predictions_test_non_vues.jsonl --local-dir data/predictions/qwen-ft
   uv run hf download MissawB/irl-guard-decision-laya predictions_test_vues.jsonl predictions_test_non_vues.jsonl --local-dir data/predictions/laya-ft
   ```
4. **Comparer** :
   ```bash
   uv run irl-guard evaluer
   ```
5. **Utiliser le modèle dans IRL Guard** : le servir avec Unsloth Studio (réglage Decision API, ou `UNSLOTH_SYSTEMONE_MODEL=<chemin du modèle fusionné> unsloth studio`), puis renseigner `laya.modele` dans la config.

## Mode combiné et démo en ligne

Par défaut, `irl-guard process` décide en **mode combiné** : les règles et Laya fine-tuné (chargé en local, sur
CPU) donnent chacun leur décision, et la plus protectrice l'emporte. Pour utiliser Qwen fine-tuné via Unsloth
Studio, régler `decideur.modele: studio` dans la config. Sur les chartes jamais vues, le mode combiné fait passer
la fuite de 13,6 % (Qwen seul) à 3,7 %, au prix d'environ 15 points de précision (les règles ignorent la charte).

- Démo : https://huggingface.co/spaces/MissawB/irl-guard-demo
- Collection « IRL Guard » (dataset et modèles) : https://huggingface.co/collections/MissawB/irl-guard-6ac9ea0b623e8e00170b6ffd
- Republier les fiches après une nouvelle évaluation : `uv run irl-guard hub publier`
