# Fine-tuning du decision model — Spec de conception

Date : 2026-10-09 · Auteur : Wassim Bahmani · Statut : à relire

Documents liés : `Bloqueur de contenu pour lives IRL — YOLO × Decision Models.md` (vision, sections « Données et labels » et « Évaluation ») et `docs/superpowers/specs/2026-10-09-irl-guard-v1-design.md` (V1 livrée).

## 1. Cadrage

**Question de recherche** (document de vision) : le decision model applique-t-il correctement des chartes jamais vues, avec une calibration qui tient, là où un classifieur tabulaire s'effondre ?

**Point de départ mesuré le 2026-10-09** : Laya multilingue sans fine-tuning, servi par Unsloth Studio 2026.10.3, se trompe sur 5 des 9 scénarios de référence. Il diffuse un e-mail, une insulte et un couteau en pleine rue, et ignore presque le mode domicile. Sa latence médiane est de 0,63 s par décision.

**Décidé avec l'utilisateur**

- Labels issus de **chartes composées de clauses** structurées, rendues en texte par des gabarits (option a). Pas de labellisation par un grand LLM.
- Approches comparées : **règles, XGBoost, Laya brut, Laya fine-tuné, Qwen3.5-0.8B fine-tuné**. Le VLM est reporté.
- Entraînement sur **Colab gratuit** (T4), stockage sur le **Hub Hugging Face** (compte `MissawB`, dépôts privés).

**Critères de réussite**

1. Un dataset reproductible (graine fixe), publié en privé sur le Hub, avec deux jeux de test : chartes vues et chartes non vues.
2. Au moins un modèle fine-tuné (Qwen en priorité) entraîné sur Colab et publié sur le Hub. Ses prédictions sur les deux jeux de test sont exportées.
3. Un rapport `docs/resultats/rapport.md` qui compare les cinq approches (ou quatre si Laya ne se fine-tune pas) : précision, taux de fuite, ECE, coût moyen, diagrammes de fiabilité.
4. Le modèle fine-tuné est utilisable par IRL Guard sans changer le code, en modifiant seulement `laya.modele` dans la config.

## 2. Générateur de données (`irl_guard/donnees/`)

### 2.1 Clauses et chartes

Une clause est un triplet structuré :

| Champ | Valeurs |
|---|---|
| `risque` | `localisation`, `donnee_perso`, `arme`, `propos` |
| `sanction` | `diffuser`, `flouter`, `pause` |
| `condition` | `toujours`, `domicile` (s'applique seulement en mode domicile), `sauf_categorie:<nom>` (ex. `sauf_categorie:cuisine`) |

Une charte compte entre 2 et 5 clauses. Elle contient au plus une clause par couple (risque, condition), et au plus une clause `toujours` par risque.

**Règle d'application** (fonction pure `appliquer_charte(clauses, risque, categorie, mode_domicile) -> sanction`) :

1. On retient les clauses du risque de la cible dont la condition est remplie :
   - `toujours` : toujours remplie ;
   - `domicile` : remplie si `mode_domicile` est vrai ;
   - `sauf_categorie:X` : remplie si la catégorie (en minuscules) est différente de X.
2. Si plusieurs clauses sont retenues, la plus protectrice l'emporte (`pause` > `flouter` > `diffuser`).
3. Si aucune ne l'est, ou si le risque vaut `aucun` : `diffuser`.

### 2.2 Formulations

Chaque combinaison (risque, sanction, condition) possède **au moins 5 gabarits français**. Exemples pour (localisation, flouter, domicile) :

- « Quand je suis en mode domicile, floute les noms de rue et les numéros. »
- « Près de chez moi, rien qui permette de me localiser ne doit être lisible. »

Une charte est le texte de ses clauses, assemblées dans un ordre aléatoire et précédées éventuellement d'une phrase neutre (« Live tout public. », « Ambiance détendue. »).

**Partition des gabarits** : pour chaque combinaison, environ 30 % des gabarits (au moins 1) sont réservés au test « non vues ». Le tirage est fixé par la graine.

**Combinaisons réservées** : quelques ensembles de clauses (par exemple « arme / diffuser / toujours » associé à « propos / pause / toujours ») n'apparaissent que dans le test « non vues ».

### 2.3 Génération des `state`

Le format est exactement celui de `irl_guard.state.construire_state` : `categorie`, `mode_domicile`, `charte`, `cible`, `contexte`.

| Type de cible | Source | Risque de la cible |
|---|---|---|
| Nom de rue, numéro, code postal | Listes de types de voie et de noms, nombres | `localisation` |
| E-mail, téléphone, nom et prénom | Faker `fr_FR` | `donnee_perso` |
| Insulte | Liste fixe | `propos` |
| Couteau, ciseaux | `cible.type = objet` | `arme` |
| Enseigne, slogan, texte neutre | Listes fixes (« SOLDES -50 % », « Boulangerie Paul »…) | `aucun` |

Champs aléatoires : catégorie (`IRL`, `Cuisine`, `Gaming`, `Musique`, `Sport`), mode domicile (vrai une fois sur trois), taille (`petite`, `moyenne`, `grande`), confiance (0,5 à 0,99), contexte (0 à 4 éléments).

**Bruit d'OCR** appliqué au texte de la cible, une fois sur trois :
- confusions de caractères (`O`/`0`, `l`/`1`, `e`/`é`) ;
- casse aléatoire ;
- troncature de 1 à 3 caractères ;
- espace manquante.

Le risque de la cible reste celui d'avant le bruit.

### 2.4 Labels (`gold`)

- `action` = `appliquer_charte(...)`.
- `risque` = le risque de la cible (indépendant de la charte).
- `gravite` = 0 si l'action est `diffuser`, 2 si `flouter`, 3 si `pause`.
- Les distributions sont lissées : 0,9 pour la bonne réponse, 0,1 répartis également entre les autres options.

### 2.5 Jeux de données et format

| Jeu | Taille par défaut | Gabarits | Combinaisons |
|---|---|---|---|
| `train` | 4 000 | « train » | non réservées |
| `val` | 500 | « train » | non réservées |
| `test_vues` | 1 000 | « train » | non réservées |
| `test_non_vues` | 1 000 | « test » uniquement | réservées et non réservées |

Les tailles se règlent en ligne de commande.

Format de chaque ligne, identique au dataset de référence d'Unsloth `LocalLLaMA/typed-decisions` :
- `id` (str) ;
- `state` (JSON sérialisé) ;
- `questions` (JSON sérialisé, **exactement** `irl_guard.decision.base.QUESTIONS`) ;
- `gold` (JSON sérialisé ; par question : `type`, `label`, `probabilities`, et `score` pour `gravite`) ;
- `clauses` (JSON des clauses structurées, pour l'analyse, ignoré à l'entraînement).

Les fichiers sont écrits en JSONL local sous `data/decision/`, puis publiés par `irl-guard donnees publier` dans le dataset privé `MissawB/irl-guard-decisions`.

## 3. Entraînement sur Colab (`notebooks/finetune_decision.ipynb`)

Un paramètre en tête : `MODELE = "qwen"` (`unsloth/Qwen3.5-0.8B`) ou `"laya"` (`convaiinnovations/laya-multilingual`).

0. **Vérification** : charger le modèle avec `FastDecisionModel.from_pretrained`. Si Laya refuse de se charger, on le note dans la cellule de sortie et la branche Laya est abandonnée.
1. Installer `unsloth` et se connecter au Hub. Le token est lu dans les secrets Colab (`HF_TOKEN`), jamais écrit dans le notebook.
2. Charger `train` et `val` depuis le Hub, puis appliquer `FastDecisionModel.build_dataset`. On n'utilise pas `split_holdout`.
3. Préparer le modèle :
   - Qwen : `load_in_4bit=True` ;
   - Laya : sans quantification ;
   - `max_seq_length=1024` ;
   - LoRA `r=16`, `lora_alpha=16`, `lora_dropout=0`.
4. Entraîner avec `DecisionTrainer` : 2 époques, `learning_rate=2e-4`, scheduler cosinus, `per_device_train_batch_size=8`, `gradient_accumulation_steps=4`. En cas de manque de mémoire, diviser le batch par 2 et doubler l'accumulation. Pousser un checkpoint sur le Hub à chaque époque ; une session coupée reprend depuis le dernier.
5. Calibrer avec `FastDecisionModel.calibrate` sur `val`, puis afficher précision, ECE et loss.
6. Exporter :
   - adaptateurs et modèle fusionné vers `MissawB/irl-guard-decision-<modele>` (privé) ;
   - prédictions sur `test_vues` et `test_non_vues`, en JSONL `{id, answers}` (`answers` au format de la Decision API), à télécharger dans `data/predictions/<modele>/`.

Le notebook ne contient aucune logique métier : génération, labels et métriques vivent dans `irl_guard`.

## 4. Évaluation (`irl_guard/evaluation/`)

### 4.1 Prédicteurs

Chaque prédicteur renvoie, pour une ligne de test, des réponses au format de la Decision API.

| Prédicteur | Fonctionnement |
|---|---|
| Règles | `RulesDecider`, avec les probabilités one-hot qu'il fournit déjà |
| XGBoost | Entraîné en local sur `train`, un classifieur par question. Features : type de cible, classe d'objet, indicateurs regex sur le texte de la cible (voie, numéro, code postal, e-mail, téléphone, insulte), catégorie, mode domicile, taille, et **TF-IDF de la charte** (mots et bigrammes, 500 dimensions au plus). Les clauses structurées ne sont jamais fournies. |
| Laya brut | `LayaDecider` contre la Decision API locale |
| Fine-tunés | Lecture des JSONL exportés par Colab |

### 4.2 Métriques

Chacune est calculée par approche et par jeu de test.

- **Précision** pour `action`, `risque` et `gravite` (pour `gravite`, on compare le niveau prédit, arrondi, au niveau attendu).
- **Taux de fuite** : part des lignes où l'action attendue est `flouter` ou `pause` et où l'action prédite est `diffuser`.
- **ECE** sur `action`, avec 10 classes de confiance, plus un diagramme de fiabilité en PNG.
- **Coût moyen par décision**, avec une matrice réglable dans la config :

| Prédit ↓ / Attendu → | diffuser | flouter | pause |
|---|---|---|---|
| diffuser | 0 | 10 | 20 |
| flouter | 1 | 0 | 5 |
| pause | 3 | 1 | 0 |

L'action prédite se calcule par `action_depuis_probas` avec les seuils de la config, comme dans le pipeline.

### 4.3 Rapport

`irl-guard evaluer --approches regles,xgboost,laya,laya-ft,qwen-ft` produit `docs/resultats/rapport.md`, avec un tableau par jeu de test et les diagrammes de fiabilité PNG associés. Une approche dont les prédictions manquent est signalée dans le rapport et ne bloque pas les autres.

## 5. Intégration

- Pour servir le modèle fusionné : `UNSLOTH_SYSTEMONE_MODEL=<chemin> unsloth studio`, ou le réglage Decision API de Studio.
- IRL Guard l'utilise via `laya.modele` dans la config, sans changer le code.
- Les questions d'inférence et d'entraînement ont une source unique : `irl_guard.decision.base.QUESTIONS`.

## 6. Tests (pytest, local, sans GPU)

- **Générateur.**
  - Même graine, mêmes lignes.
  - `appliquer_charte` est testé pour chaque condition et pour le conflit entre clauses.
  - Aucun gabarit du test « non vues » n'apparaît dans `train`.
  - Les combinaisons réservées sont absentes de `train`.
  - Format des lignes : clés, JSON valide, `questions == QUESTIONS`, probabilités dont la somme fait 1.
  - Le bruit d'OCR ne change pas le risque.
- **Évaluation.**
  - Précision, taux de fuite, ECE et coût sont vérifiés sur des exemples calculés à la main.
  - Les features XGBoost sont testées (indicateurs regex, dimension TF-IDF).
  - Le rapport est généré avec des prédicteurs factices.
  - Une approche manquante est signalée sans planter.
- **Notebook** : pas de test automatique. La cellule 0 et la calibration sur `val` servent de garde-fous.

## 7. Hors périmètre

- VLM.
- Entraînement sur de vraies détections YOLO : seul le bruit d'OCR est simulé.
- Mode combiné règles + Laya.
- Mode domicile par GPS.
- Fine-tuning sur le GPU local.
- Labellisation par un grand LLM.

## 8. Dépendances ajoutées

`faker`, `xgboost`, `scikit-learn` (TF-IDF), `matplotlib` (diagrammes), `datasets` (publication sur le Hub). Le notebook installe `unsloth` sur Colab ; rien de plus en local.
