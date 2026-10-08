# IRL Guard V1 — Spec de conception

Date : 2026-10-09 · Auteur : Wassim Bahmani · Statut : à relire

Document parent : `Bloqueur de contenu pour lives IRL — YOLO × Decision Models.md` (vision complète du projet). Cette spec couvre uniquement la **V1** : traitement hors ligne d'une vidéo enregistrée.

## 1. Cadrage

**Ce qui a été décidé avec l'utilisateur**

- Projet personnel, destiné au portfolio, sans échéance.
- Matériel : PC portable RTX 4060 Laptop (8 Go VRAM), Ryzen 5 7535HS, 16 Go RAM, Windows 11, Python 3.12 ; instances gratuites Colab et Hugging Face pour l'entraînement (hors V1).
- Premier livrable : **vidéo enregistrée → vidéo protégée** (flou, écran de pause, overlay de debug).
- Le **decision model fait partie de la V1**, via Laya non fine-tuné, derrière une interface `Decider` remplaçable.
- Perception : **assemblage de détecteurs pré-entraînés** (approche A), OCR **RapidOCR**.

**Hypothèses (non confirmées explicitement)**

- Laya est servi par l'app Unsloth (Desktop/Studio) installée séparément, sur `http://localhost:8888`.
- ffmpeg est disponible dans le PATH (ou installé par la procédure de setup).

**Objectif V1** : démontrer de bout en bout qu'un pipeline YOLO + OCR + decision model protège une vidéo IRL, et qu'à détection identique la décision change selon la charte et le mode domicile.

## 2. Architecture

Paquet Python `irl_guard/`. Chaque module a une responsabilité unique et une interface simple.

| Module | Rôle | Entrée → sortie |
|---|---|---|
| `sources/` | Lecture des images | fichier vidéo → `Frame(index, t, image)` |
| `sampling` | Sélection des images analysées (2–5 i/s), hash perceptuel pour sauter les images inchangées | `Frame` → analyser / réutiliser |
| `perception/` | Un wrapper par détecteur : `objects` (YOLO COCO), `faces`, `plates`, `nudity` (NudeNet), `text` (RapidOCR) | image → `list[Detection(classe, conf, boite, texte?)]` |
| `tracking` | ByteTrack + accumulation OCR par piste | détections → `list[Track(id, classe, boite, texte_cumule, nouveau, modifie)]` |
| `state` | Construction du `state` JSON compact pour une piste | piste + scène + config → `dict` |
| `decision/` | Interface `Decider` ; implémentations `RulesDecider` et `LayaDecider` (client HTTP) | `state` → `Decision(action, risque, gravite, probabilites)` |
| `temporal` | Hystérésis de la pause | score par analyse → `diffusion` / `pause` |
| `render` | Flou/pixellisation par piste, écran de pause, overlay de debug | image + plan de rendu → image |
| `pipeline` | Orchestration, écriture vidéo, remux audio ffmpeg, journal | vidéo → vidéo protégée + journal JSONL |
| `config` | Chargement et validation du YAML de session | fichier → objet `SessionConfig` |
| `cli` | Point d'entrée | ligne de commande |

**CLI**

```
irl-guard process input.mp4 --config session.yaml --out out.mp4 [--debug-overlay] [--decider laya|rules]
irl-guard replay journal.jsonl input.mp4 --config session.yaml --out out.mp4   # rejoue sans refaire la perception
```

**Configuration de session (YAML)** : catégorie du stream, charte (texte libre), mode domicile, fréquence d'échantillonnage, seuils (décision, hystérésis, NudeNet), mode de masquage (flou ou pixellisation, défaut : pixellisation), marge d'élargissement des boîtes, coupure audio en pause, URL et modèle Laya, choix des poids de chaque détecteur.

**Journal JSONL** : une ligne par image analysée (index, détections, pistes, `state` envoyés, réponses du décideur, action par piste, action de l'image, état temporel). Les détections sont aussi mises en cache sur disque pour `replay`.

## 3. Perception

| Détecteur | Implémentation | Classes utilisées |
|---|---|---|
| Objets | Ultralytics YOLO pré-entraîné COCO (taille n ou s) | `person`, `knife`, `scissors`, `car`, `motorcycle`, `bus`, `truck`, `cell phone`, `laptop`, `stop sign` |
| Visages | Détecteur pré-entraîné (YOLO-face ou SCRFD) | `visage` |
| Plaques | Détecteur YOLO pré-entraîné de plaques (Hub) | `plaque` |
| Nudité | NudeNet | classes exposées de NudeNet |
| Texte | RapidOCR (modèles PaddleOCR en ONNX, onnxruntime-gpu) | `texte` (boîte + chaîne reconnue) |

Le choix exact des poids (visages, plaques) se fait pendant l'implémentation, sur un critère simple : licence compatible, téléchargeable depuis le Hub, fonctionne sur les images de référence. Chaque wrapper implémente la même interface `Detector.detect(image) -> list[Detection]`.

Les armes à feu ne sont pas couvertes en V1 (absentes de COCO).

## 4. Flux de données

```
Frame → sampling ──(inchangée)──→ réutilise la dernière analyse
          │
          ▼
   perception → tracking (ByteTrack + OCR cumulé par piste)
          │
          ├─ pistes « règle » (visage, plaque) → flouter, sans décideur
          ├─ nudité NudeNet ≥ seuil           → pause, sans décideur
          └─ pistes « contexte » (texte, couteau, ciseaux, téléphone, ordinateur)
             nouvelles ou dont le texte a changé
                    │
                    ▼
              state → Decider → cache par (piste_id, hash du texte)
                    │
                    ▼
   action de l'image = max(pause > flouter > diffuser)
          → temporal → render → vidéo de sortie
```

**Une décision par piste.** Le décideur juge une seule piste à la fois. Le résultat est mis en cache tant que la piste existe et que son texte cumulé ne change pas.

**Format du `state`**

```json
{
  "categorie": "IRL",
  "mode_domicile": true,
  "charte": "Tout public. Jamais de nom de rue ni de numéro près de chez moi.",
  "cible": {"type": "texte", "texte": "Rue Nationale", "conf": 0.91, "taille": "grande"},
  "contexte": ["personne x2", "voiture x1", "texte: SOLDES -50 %"]
}
```

- `cible.taille` : `petite` / `moyenne` / `grande` selon la part de l'image occupée par la boîte (seuils dans la config).
- `contexte` : résumé du reste de la scène, sans coordonnées.
- Budget : 1024 tokens (Laya multilingue). Si le budget est dépassé, `contexte` est tronqué en premier ; `cible` et `charte` ne sont jamais tronquées. Le comptage utilise une estimation conservatrice (caractères / 3) en l'absence du tokenizer exact.

**Questions envoyées**

```json
{
  "action": {"type": "choice", "instructions": "Que faire de cet élément dans le live ?",
             "criteria": {"diffuser": "rien de problématique",
                          "flouter": "problème localisé et masquable",
                          "pause": "problème grave ou impossible à masquer"}},
  "risque": {"type": "choice", "instructions": "Quel est le risque principal ?",
             "criteria": {"localisation": "nom de rue, numéro, enseigne, repère identifiable",
                          "donnee_perso": "document, écran, nom, adresse, e-mail",
                          "arme": "arme réelle ou objet menaçant",
                          "propos": "texte injurieux ou haineux",
                          "aucun": "aucun risque"}},
  "gravite": {"type": "score", "instructions": "Quelle est la gravité ?",
              "criteria": ["aucune", "faible", "élevée", "critique"]}
}
```

**Règle d'action par piste** (sur `probabilities`, jamais sur `confidence`) :

- `pause` si P(pause) ≥ `seuil_pause` (défaut 0,5) ;
- sinon `flouter` si P(flouter) + P(pause) ≥ `seuil_flou` (défaut 0,4, biais vers la protection) ;
- sinon `diffuser`.

**Score de l'image pour la couche temporelle** : max des P(pause) des pistes contexte de l'image ; vaut 1,0 si NudeNet a déclenché.

## 5. Décideurs

**Interface**

```python
class Decider(Protocol):
    def decide(self, state: dict) -> Decision: ...
```

`Decision` contient `action`, `risque`, `gravite`, `probabilites` (par question) et `source` (`laya`, `rules`, `fallback`).

**`LayaDecider`** : `POST {url}/v1/systemone` avec `model` (défaut `laya-multilingual`), `state`, `questions`, en-tête `Authorization: Bearer <clé>` si une clé est configurée. Timeout configurable (défaut 5 s). Le même client servira plus tard le modèle fine-tuné (même API).

**`RulesDecider`** : plancher de référence et repli. Listes de mots (types de voie, motifs de numéro, mots injurieux), règles par catégorie de stream pour les couteaux, mode domicile → `pause` pour toute piste `localisation` de taille moyenne ou grande.

## 6. Couche temporelle et rendu

**Images non analysées.** Pour chaque piste, la boîte est interpolée linéairement entre l'analyse précédente et la suivante (connues grâce au tampon), puis élargie de `marge` (défaut 15 %). Une piste perdue garde son masque pendant `maintien` analyses (défaut : durée de survie ByteTrack). En temps réel, cela reste valable tant que le tampon dépasse l'intervalle d'échantillonnage.

**Hystérésis**

- Entrée en pause dès qu'un score ≥ `seuil_haut` (défaut 0,7).
- Sortie après `K` analyses consécutives < `seuil_bas` (défauts : 0,3 et K = 3).

**Rendu**

- Masquage par piste : pixellisation (défaut) ou flou gaussien fort.
- Écran de pause : image configurable avec le texte « Pause de sécurité ».
- Audio : remux de la piste d'origine via ffmpeg, coupée pendant les pauses (option `couper_audio_en_pause`, défaut vrai).
- Overlay de debug (option) : boîtes colorées selon l'action, id de piste, texte OCR, probabilités, état de l'hystérésis.

## 7. Gestion des erreurs : échec du côté sûr

| Situation | Comportement |
|---|---|
| Laya injoignable au démarrage | Arrêt avec message explicite ; l'utilisateur relance avec `--decider rules` s'il le souhaite |
| Timeout ou erreur Laya en cours de traitement | Piste **floutée** (`source = fallback`), événement journalisé, nouvel essai à l'analyse suivante |
| Réponse hors format | Même traitement que le timeout |
| Échec de chargement d'un détecteur | Arrêt au démarrage en nommant le modèle en cause |
| `state` trop long | Troncature de `contexte` (voir §4) |
| Vidéo sans piste audio | Sortie sans audio, avertissement |

## 8. Tests

- **Unitaires** (pytest, sans GPU ni Laya) : échantillonnage et hash, accumulation OCR, construction du `state` et troncature, règle d'action par piste, agrégation par image, hystérésis, interpolation et élargissement des boîtes, maintien après perte, `RulesDecider`, chargement de la config.
- **`LayaDecider`** contre un faux serveur HTTP local : réponse valide, timeout, réponse invalide (vérifie le flou par défaut).
- **Pipeline** avec des détecteurs factices injectés.
- **Intégration GPU** (`@pytest.mark.gpu`, exclus par défaut) : chaque vrai détecteur sur 3–4 images de référence (plaque, panneau de rue, visage).
- **Test anti-fuite bout en bout** : clip synthétique avec « Rue Nationale » incrusté et en mouvement ; en sortie, l'OCR ne doit relire le texte sur **aucune** image.

## 9. Critères de réussite V1

1. `irl-guard process clip.mp4` produit une vidéo protégée avec audio, et une variante avec overlay de debug.
2. Test anti-fuite : zéro image où le texte reste lisible.
3. Sur 2–3 clips IRL réels, la décision change selon la charte et le mode domicile à détection identique.
4. Débit de traitement hors ligne mesuré et affiché ; cible indicative ≥ 1× temps réel sur la RTX 4060.
5. README de portfolio avec GIF avant/après.

## 10. Hors périmètre V1

- Temps réel : webcam, SRT, RTMP, pyvirtualcam, reprise forcée par le streamer.
- Fine-tuning YOLO sur Open Images (dont armes à feu).
- Données synthétiques et fine-tuning du decision model.
- Évaluation comparative (règles, XGBoost, Laya, modèle fine-tuné, VLM) et sessions mises en scène.
- Audio / Whisper, liste blanche de visages, mode domicile par GPS, matrice de coûts formelle.

## 11. Environnement

- Python 3.12, venv géré par `uv`, `pyproject.toml`.
- Dépendances : PyTorch (CUDA), ultralytics, rapidocr + onnxruntime-gpu, nudenet, opencv-python, imagehash, httpx, pydantic, pyyaml, typer ; pytest en dev.
- ffmpeg dans le PATH.
- App Unsloth installée séparément, Decision API activée, modèle multilingue.
- Code portable Linux (pas de chemins Windows en dur) pour réutilisation sur Colab.
