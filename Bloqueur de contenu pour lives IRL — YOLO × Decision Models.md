# Bloqueur de contenu pour lives IRL — YOLO × Decision Models

Oct 9, 2026 · @Wassim Bahmani

## Contexte et objectif

Le projet construit un bloqueur qui protège un live IRL : il floute une zone ou bascule sur un écran de pause avant qu'un contenu problématique ne soit diffusé. Il combine YOLO pour la perception, un OCR pour le texte et un decision model Unsloth pour décider selon le contexte.

En IRL, le premier risque est la localisation du streamer : une plaque de rue, un numéro de maison, une enseigne, une plaque d'immatriculation ou un badge suffisent à le retrouver. Viennent ensuite les données personnelles visibles, la nudité, les armes réelles, les propos offensants et les visages de passants.

Le projet a deux volets : une démo temps réel qui fonctionne de bout en bout, puis une question de recherche sur la capacité du decision model à appliquer des chartes jamais vues.

## Principe : pourquoi un decision model

Le decision model n'apporte de valeur que si la décision dépend d'un texte qui varie. Sinon, un XGBoost sur les détections fait aussi bien pour beaucoup moins cher, et un jury le verra.

Un decision model Unsloth est un LLM fine-tuné en LoRA, surmonté d'une petite tête de classification inspirée de Clef. Il ne génère pas de texte : il lit un `state` (chaîne ou JSON) et des questions typées, puis renvoie des probabilités calibrées.

| Type | Usage | Retour |
| --- | --- | --- |
| `noul` | Oui / non | Probabilité du oui, entre 0 et 1 |
| `choice` | Une option parmi N, décrites en langage naturel | Option la plus probable + probabilité de chaque option |
| `score` | Échelle jusqu'à 10 niveaux | Position sur l'échelle + probabilité de chaque niveau |

Dans un live IRL, à détection identique, la décision change selon le contexte :

- « Rue Nationale » révèle une position, « SOLDES -50 % » non.
- Un couteau est normal en live cuisine, pas en pleine rue.
- Une plaque de rue est anodine dans un quartier touristique, critique en mode domicile.

La charte de la chaîne s'écrit en langage naturel. Le modèle doit donc appliquer des règles exprimées en texte, y compris des règles absentes de son entraînement.

## Architecture système

Le flux ne part vers la plateforme qu'après un tampon de quelques secondes : la branche d'analyse a ce délai pour décider du flou ou de la pause.

&#91;embedded content: architecture · flux principal et branche d’analyse\]

Le téléphone envoie le flux en SRT vers un serveur relais équipé d'un GPU, qui renvoie la sortie en RTMP vers la plateforme. Les streamers IRL utilisent déjà ce schéma : NOALBS bascule sur une scène « BRB » quand le débit chute, ce projet le fait sur un critère de contenu. En local, la même chaîne peut sortir vers une caméra virtuelle (pyvirtualcam) reprise par OBS.

## Pipeline de traitement

Chaque image analysée passe par six étapes, exécutées sur le serveur relais pendant le délai du tampon.

1. **Échantillonnage.** On analyse 2 à 5 images par seconde. Un hash perceptuel évite de relancer la chaîne quand l'image n'a pas changé.
2. **Perception.** YOLO détecte les objets à risque et ByteTrack les suit d'une image à l'autre. NudeNet couvre la nudité. Un OCR (PaddleOCR ou EasyOCR) lit le texte visible.
3. **Construction du state.** Détections, texte OCR accumulé par piste, catégorie du stream, charte et mode domicile forment un JSON compact.
4. **Décision.** Des règles déterministes floutent visages et plaques. Le decision model n'est appelé que lorsqu'un nouveau texte ou un nouvel objet apparaît.
5. **Couche temporelle.** Entrée en pause quand la probabilité dépasse un seuil haut, sortie après plusieurs analyses consécutives sous un seuil bas. Les deux seuils découlent de la matrice de coûts et évitent le clignotement.
6. **Sortie.** On floute la boîte YOLO ou OCR quand le problème est localisé, on bascule sur l'écran de pause sinon. L'audio est retardé exactement du même délai que la vidéo.

## Répartition règles / decision model

Ce qu'une règle fait mieux reste une règle ; le decision model ne traite que ce qui dépend du sens ou du contexte.

| Élément | Traité par | Action par défaut |
| --- | --- | --- |
| Visage de passant | Règle | Flou systématique, sauf liste blanche validée par le streamer |
| Plaque d'immatriculation | Règle | Flou systématique |
| Nom de rue, numéro, enseigne | Decision model | Flou, ou pause en mode domicile |
| Donnée personnelle (document, écran de téléphone) | Decision model | Flou ou pause |
| Arme, couteau | Decision model | Selon la catégorie du stream et la charte |
| Texte offensant (pancarte, tag, vêtement) | Decision model | Flou ou pause |
| Nudité | NudeNet, puis decision model selon la charte | Pause |

Exemple de requête envoyée au decision model :

```json
{
  "state": {
    "categorie": "IRL",
    "mode_domicile": true,
    "charte": "Tout public. Ne jamais montrer de nom de rue ni de numéro près de mon domicile. Armes réelles interdites.",
    "detections": [
      {"classe": "panneau", "conf": 0.88, "piste": 12},
      {"classe": "couteau", "conf": 0.64, "piste": 15}
    ],
    "texte_ocr": [{"piste": 12, "texte": "Rue Nationale"}]
  },
  "questions": {
    "action": {
      "type": "choice",
      "instructions": "Que faire de cette image ?",
      "criteria": {
        "diffuser": "rien de problématique",
        "flouter": "problème localisé et masquable",
        "pause": "problème grave ou impossible à masquer"
      }
    },
    "risque": {
      "type": "choice",
      "instructions": "Quel est le risque principal ?",
      "criteria": {
        "localisation": "nom de rue, numéro, enseigne, repère identifiable",
        "donnee_perso": "document, écran, nom, adresse, e-mail",
        "nudite": "nudité ou contenu sexuel",
        "arme": "arme réelle ou objet menaçant",
        "propos": "texte injurieux ou haineux",
        "aucun": "aucun risque"
      }
    },
    "localisable": {
      "type": "noul",
      "instructions": "L'image permet-elle de localiser précisément le streamer ?"
    },
    "gravite": {
      "type": "score",
      "instructions": "Quelle est la gravité ?",
      "criteria": ["aucune", "faible", "élevée", "critique"]
    }
  }
}
```

Les seuils d'action portent sur `probabilities`, jamais sur `confidence`, comme le recommande la documentation Unsloth.

## Decision model avec Unsloth

Le modèle de départ est Qwen3.5-0.8B fine-tuné en LoRA : environ 4 Go de VRAM et une quarantaine de minutes d'entraînement selon les mesures d'Unsloth. Laya, utilisé sans fine-tuning sur CPU, sert de premier prototype.

1. Charger le modèle en 4 bits avec `FastDecisionModel.from_pretrained`, puis ajouter LoRA (rang 16) avec `get_peft_model`.
2. Transformer les lignes state / questions / réponses attendues avec `build_dataset`, puis isoler un holdout avec `split_holdout`.
3. Entraîner avec `DecisionTrainer` : 2 époques, learning rate 2e-4, 3 à 4 époques si les données sont peu nombreuses. `max_steps = 60` donne un premier résultat en une dizaine de minutes.
4. Calibrer sur le holdout avec `calibrate`, qui renvoie précision, ECE et loss.
5. Sauvegarder avec `save_pretrained_merged`, puis appeler `predict` depuis le pipeline ou servir le modèle via la Decision API locale.

Contraintes à respecter :

- **Longueur d'entrée.** Laya lit 1024 tokens ; le modèle fine-tuné lit la longueur fixée par `max_seq_length` (2048 dans l'exemple Unsloth). Arrondir les coordonnées et agréger les détections.
- **Options.** Au-delà d'une vingtaine d'options décrites dans un `choice`, les descriptions sont tronquées. Garder 3 à 6 options par question.
- **Limites par requête.** 64 questions, 255 options par `choice`, 10 niveaux par `score`.
- **Latence.** Laya répond en moins d'une seconde sur CPU une fois chargé ; un modèle fine-tuné servi par la Decision API demande un GPU. Le délai du tampon doit couvrir cette latence avec marge.

## Difficultés propres à l'IRL

Le danger principal est la fuite intermittente : une plaque floutée sur neuf images et nette sur la dixième a fuié.

- **Caméra en mouvement, flou, nuit.** YOLO et l'OCR ratent des images de façon intermittente. On floute par piste et non par détection : ByteTrack maintient le masque quand la détection décroche, et les boîtes sont légèrement élargies.
- **Texte vu par morceaux.** Une enseigne apparaît partiellement sur plusieurs images. On accumule l'OCR par piste avant d'interroger le decision model.
- **Coût de calcul.** Le decision model n'est appelé qu'à l'apparition d'un nouveau texte ou d'un nouvel objet, pas à chaque image.
- **Mode domicile.** Le streamer l'active lui-même, ou un calcul GPS l'active à l'approche de son domicile. Il entre dans le `state` comme un simple champ de contexte.

## Données et labels

La perception s'appuie sur des jeux publics et des détecteurs existants ; les labels du decision model sont construits par un référentiel de règles écrit en texte.

| Besoin | Source |
| --- | --- |
| Armes de poing, couteaux, plaques d'immatriculation, visages | Open Images V7, téléchargé filtré par classes avec FiftyOne |
| Visages (détection dédiée) | Détecteur pré-entraîné sur WIDER FACE |
| Nudité | NudeNet, détecteur pré-entraîné |
| Texte offensant | Jigsaw Toxic Comments |
| Fausses données personnelles | Faker (noms, adresses, e-mails, numéros) |

**Decision model.** On génère des `state` synthétiques qui combinent détections, textes de rue réels ou générés, chartes variées et mode domicile activé ou non. Le référentiel de règles fournit les réponses attendues.

- Les chartes et formulations du jeu de test sont absentes de l'entraînement.
- Le modèle s'entraîne sur les détections YOLO, pas sur la vérité terrain, pour apprendre à composer avec les erreurs de perception. Comparer les deux versions constitue une expérience en soi.

**Bout en bout.** Une dizaine de sessions filmées, avec des événements mis en scène par des amis consentants : pièce d'identité factice, plaque de rue, notification sur un téléphone. Chaque événement est horodaté.

- Pas de réplique d'arme dans la rue, qui peut déclencher une vraie intervention : scènes en intérieur ou image affichée sur un écran.
- Les passants sont floutés dans toutes les vidéos conservées.

## Évaluation

Question de recherche : le decision model applique-t-il correctement des chartes jamais vues, avec une calibration qui tient, là où un classifieur tabulaire s'effondre ?

| Approche | Ce qu'elle mesure |
| --- | --- |
| Règles seules (seuils fixes sur YOLO, listes de mots) | Plancher de référence |
| XGBoost sur features tabulaires, charte encodée | Apport de la lecture du texte libre |
| Laya sans fine-tuning | Apport du fine-tuning |
| Qwen3.5-0.8B fine-tuné et calibré | Modèle proposé |
| VLM open-source de taille comparable, image et charte en entrée | Pipeline modulaire face au bout en bout |

Hypothèse pour la dernière ligne : le pipeline modulaire est plus rapide, mieux calibré et plus auditable, car on sait si une erreur vient de la perception ou de la décision. Le VLM capte en revanche ce que YOLO ne sait pas détecter.

**Métriques du modèle**, sur chartes vues et non vues :

- précision par question ;
- ECE (erreur de calibration) ;
- coût moyen par décision selon la matrice de coûts.

**Métriques opérationnelles**, sur les sessions mises en scène :

- images problématiques réellement diffusées, objectif zéro ;
- temps de pause injustifiée par heure de stream ;
- latence de décision comparée au délai du tampon ;
- nombre de bascules par heure.

## Limites, éthique et cadre légal

Un faux positif revient à censurer du contenu légitime : le streamer doit pouvoir forcer la reprise à tout moment.

- **OCR.** Il peine sur le texte stylisé ou qui défile vite.
- **Audio.** Il n'est pas couvert. L'extension naturelle est une transcription Whisper envoyée au même decision model.
- **Violences et altercations.** Elles relèvent de la reconnaissance d'actions, hors du périmètre de YOLO.
- **Confidentialité.** Tout tourne sur le relais du streamer : le flux ne passe par aucun service tiers d'analyse.
- **RGPD.** Si la liste blanche repose sur la reconnaissance faciale, il s'agit de données biométriques : consentement explicite des personnes concernées et traitement local.
- **Jeux d'évaluation.** Consentement des participants aux sessions filmées, passants floutés dans tout ce qui est conservé.

## Feuille de route

Les phases 1 à 3 livrent une démo qui fonctionne sur un vrai flux ; les phases 4 et 5 apportent la contribution de recherche.

&#91;embedded content: feuille de route · 5 phases, 1 jalon\]

## Références

**Sources consultées**

- [Train your own Decision Model with Unsloth](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)
- [Run and Serve Decision Models Locally with Unsloth: Laya + Jev API](https://unsloth.ai/docs/models/decision-laya)

**Outils et jeux de données mentionnés**

- [Ultralytics YOLO](https://docs.ultralytics.com)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) · [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [NudeNet](https://github.com/notAI-tech/NudeNet)
- [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) · [FiftyOne](https://docs.voxel51.com)
- [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
- [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [Faker](https://github.com/joke2k/faker)
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam)
- [NOALBS](https://github.com/NOALBS/nginx-obs-automatic-low-bitrate-switching)
- [Whisper](https://github.com/openai/whisper)
