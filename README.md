# IRL Guard

**Bloqueur de contenu pour lives IRL, guidé par la charte du streamer.**

Un streamer qui se filme dans la rue diffuse en permanence des informations qu'il ne contrôle pas : un nom de rue
près de chez lui, un numéro de téléphone sur une vitrine, une insulte sur une pancarte, un couteau sur une table.
IRL Guard analyse la vidéo, détecte ces éléments et décide pour chacun de le **diffuser**, le **flouter** ou de
mettre le live en **pause**, en appliquant une **charte écrite en langage naturel** par le streamer.

À détection identique, la décision change selon le contexte :
- « Rue Nationale » est diffusée en balade, mais déclenche une pause quand le streamer est près de chez lui ;
- un couteau est normal dans un live cuisine, pas en pleine rue ;
- « les gros mots ne me dérangent pas » autorise ce qu'une autre charte ferait flouter.

🔗 **Démo en ligne** : [huggingface.co/spaces/MissawB/irl-guard-demo](https://huggingface.co/spaces/MissawB/irl-guard-demo)
· **Modèles et datasets** : [Collection IRL Guard](https://huggingface.co/collections/MissawB/irl-guard-6ac9ea0b623e8e00170b6ffd)

---

## Résultats

La question de recherche : **un petit modèle fine-tuné peut-il appliquer une charte qu'il n'a jamais vue, là où un
classifieur tabulaire s'effondre ?** Le juge final est un jeu de **20 chartes écrites à la main**, dans un style libre
(familier, abréviations, formulations indirectes). Aucune de leurs expressions n'apparaît dans les données
d'entraînement, et un test le vérifie.

| Approche (chartes écrites à la main, 1 000 cas) | Précision | Fuite* | Coût moyen** |
|---|---|---|---|
| Toujours diffuser (plancher) | 36,9 % | 100 % | 8,31 |
| Règles déterministes | 53,1 % | 20,4 % | 2,52 |
| XGBoost (charte en TF-IDF) | 46,0 % | 59,6 % | 4,83 |
| Laya, sans fine-tuning | 40,5 % | 44,7 % | 4,58 |
| Laya fine-tuné (v2) | 55,0 % | 47,4 % | 3,90 |
| Qwen3.5-2B fine-tuné (v2) | **79,1 %** | 29,2 % | 2,21 |
| **Règles + Qwen v2 (mode par défaut)** | 71,9 % | **5,5 %** | **0,77** |

\* Part des cas à protéger (flouter ou pause attendu) que l'approche laisse diffuser.
\** Matrice de coûts : une fuite coûte 10 à 20, une censure inutile 1 à 3.

Sur les chartes synthétiques jamais vues, Qwen v2 atteint 97,3 % de précision avec 2,2 % de fuite, et XGBoost
retombe au niveau du plancher. Le rapport complet, avec les diagrammes de calibration, est dans
[`docs/resultats/rapport.md`](docs/resultats/rapport.md).

**Ce que le projet a appris en route :**
- **v1** (Qwen3.5-0.8B, 5 formulations par sanction) : 84,6 % sur les chartes synthétiques, mais seulement 59,9 %
  sur les chartes écrites à la main. Le modèle avait appris les *mots* des gabarits (« floute », « masque »), pas la
  notion : « pixelise-les » ou « brouille-les » étaient diffusés.
- **v2** (Qwen3.5-2B, environ 20 formulations par sanction, liste noire des expressions de test) : +19 points sur
  les chartes écrites à la main.
- **Seul, un modèle fuit encore trop pour du direct** : le mode combiné, où la décision la plus protectrice entre les
  règles et le modèle l'emporte, divise la fuite par 5. En contrepartie, les règles écrasent parfois une autorisation
  écrite dans la charte.

---

## Fonctionnement

```text
vidéo → échantillonnage → perception → suivi → state JSON → décision → hystérésis → rendu → vidéo protégée
          (3 i/s)          YOLO, OCR,   ByteTrack   charte +      règles +     pause        flou, écran
                           NudeNet      par classe  élément vu    Qwen v2      lissée       de pause, audio
```

1. **Perception** : YOLO (objets COCO), détecteurs YOLO de visages et de plaques, NudeNet, RapidOCR.
2. **Suivi** : ByteTrack par classe. Le texte OCR s'accumule par piste, et une piste perdue garde son masque
   quelques analyses.
3. **Décision**, une fois par piste et par texte :
   - visages, plaques et nudité sont toujours masqués ;
   - textes et objets ambigus passent au **decision model**, qui reçoit un `state` JSON (charte, catégorie du stream,
     mode domicile, élément vu) et répond à trois questions typées (action, risque, gravité) avec des
     probabilités ;
   - en mode combiné, la décision la plus protectrice entre les règles et le modèle l'emporte.
4. **Hystérésis** : la pause commence immédiatement et ne se termine qu'après plusieurs analyses calmes.
5. **Rendu** : les masques sont interpolés entre deux analyses grâce au tampon, et l'audio est coupé pendant les
   pauses.

**Échec du côté sûr** : toute erreur d'un modèle (délai dépassé, réponse invalide, plantage) produit un flou, jamais
une diffusion. Un test de bout en bout vérifie qu'aucune image de sortie ne laisse l'OCR relire un texte sensible.

### Le decision model

- **Données** : chartes synthétiques composées de 2 à 5 clauses structurées (risque, sanction, condition : toujours,
  en mode domicile, sauf dans une catégorie), rendues en français par des gabarits. Les labels se calculent de façon
  déterministe à partir des clauses, et un bruit d'OCR est simulé. Les labels sont équilibrés à environ 45 %
  diffuser, 25 % flouter et 30 % pause.
- **Entraînement** : LoRA avec [Unsloth](https://unsloth.ai) (`FastDecisionModel`), sur Colab ou en local sur une
  RTX 4060 (Laya v2 en 20 minutes).
- **Inférence** : Qwen se charge sans Unsloth, avec le `joint_schema_model.py` publié dans son dépôt, en environ
  0,15 s par décision sur GPU. Laya tourne sur CPU avec le paquet `laya`.

---

## Installation

Python 3.12 et [uv](https://docs.astral.sh/uv/). PyTorch est installé avec CUDA 12.8.

```bash
uv sync
```

Au premier lancement, IRL Guard télécharge depuis le Hub les détecteurs et le decision model, ce dernier à une
version figée (`decideur.qwen_revision`). Ensuite, tout fonctionne hors ligne avec `HF_HUB_OFFLINE=1`.

## Utilisation

```bash
# Protéger une vidéo (mode combiné règles + Qwen v2 par défaut)
uv run irl-guard process clip.mp4 --config configs/exemple.yaml --out sorties/clip.mp4 --debug-overlay

# Règles seules, sans modèle
uv run irl-guard process clip.mp4 --config configs/exemple.yaml --out sorties/clip.mp4 --decider rules

# Rejouer une analyse avec d'autres réglages, sans refaire la perception
uv run irl-guard replay sorties/clip.jsonl clip.mp4 --config autre.yaml --out sorties/rejeu.mp4
```

La charte, la catégorie du stream, le mode domicile, les seuils et le modèle de décision se règlent dans un YAML
([`configs/exemple.yaml`](configs/exemple.yaml)). Sans GPU, utiliser `decideur.modele: laya-local` (Laya v2 sur CPU).

Fichiers produits :
- la vidéo protégée ;
- un journal JSONL (détections, `state` envoyés, décisions) ;
- avec `--debug-overlay`, une vidéo de debug (boîtes, pistes, probabilités). ⚠️ Elle affiche le texte lu en clair :
  ne jamais la diffuser.

### Recherche et reproduction

```bash
uv run irl-guard donnees generer                      # 5 jeux : train, val, test_vues, test_non_vues, test_manuscrites
uv run irl-guard predire --approche qwen-ft-v2 --jeu test_manuscrites   # prédictions locales d'un modèle fine-tuné
uv run irl-guard evaluer                              # rapport comparatif + diagrammes de calibration
uv run irl-guard hub publier --dry-run                # fiches Hugging Face et Space, générées en local
```

Entraînement : [`notebooks/finetune_decision.ipynb`](notebooks/finetune_decision.ipynb) sur Colab, ou en local avec
le Python d'Unsloth Studio :

```bash
~/.unsloth/studio/unsloth_studio/Scripts/python.exe scripts/finetune_local.py --modele qwen
```

## Tests

```bash
uv run pytest            # 237 tests, sans GPU ni réseau
uv run pytest -m gpu     # vrais détecteurs et modèles (téléchargements)
uv run pytest -m slow    # test anti-fuite de bout en bout avec OCR réel
```

---

## Structure

```text
src/irl_guard/
  sources, sampling, tracking, state, analysis, temporal, planning, render, media, pipeline   pipeline vidéo
  perception/      YOLO, visages, plaques, NudeNet, RapidOCR
  decision/        règles, Laya (API ou local), Qwen local, mode combiné
  donnees/         clauses, gabarits, chartes synthétiques et 20 chartes écrites à la main
  evaluation/      métriques (précision, fuite, ECE, coût), XGBoost, rapport
  hub/             fiches Hugging Face et publication
space/             démo Gradio (ZeroGPU)
scripts/           fine-tuning local
docs/superpowers/  specs et plans de chaque étape
docs/resultats/    rapport et diagrammes de calibration
```

Le document de vision initial est
[`Bloqueur de contenu pour lives IRL — YOLO × Decision Models.md`](Bloqueur%20de%20contenu%20pour%20lives%20IRL%20%E2%80%94%20YOLO%20%C3%97%20Decision%20Models.md).

## Limites et suite

- **Hors ligne uniquement** : le temps réel (webcam ou SRT vers OBS) reste à faire. Les briques sont prêtes :
  tampon, interpolation des masques, hystérésis.
- **Pas encore testé sur de vraies vidéos IRL** à grande échelle. Les mesures portent sur les décisions, à partir des
  `state` JSON.
- **Les 20 chartes manuscrites ont été écrites et annotées par une seule personne.** Des chartes rédigées par d'autres
  streamers seraient le vrai test de généralisation.
- **Perception** : pas de détection d'armes à feu (absentes de COCO), et l'OCR peine sur le texte stylisé ou flou.
  L'audio n'est pas analysé.
- **En mode combiné, les règles ignorent les autorisations de la charte**, d'où environ 7 points de précision perdus
  pour une fuite divisée par 5.

## Licence

[Apache-2.0](LICENSE), comme les modèles et le Space publiés sur Hugging Face.
