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
