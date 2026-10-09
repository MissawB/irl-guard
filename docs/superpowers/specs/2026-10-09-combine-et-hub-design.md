# Mode combiné et vitrine Hugging Face — Spec de conception

Date : 2026-10-09 · Auteur : Wassim Bahmani · Statut : à relire

Documents liés : `docs/superpowers/specs/2026-10-09-irl-guard-v1-design.md`, `docs/superpowers/specs/2026-10-09-fine-tuning-decision-design.md` et `docs/resultats/rapport.md`.

## 1. Cadrage

**Point de départ (2026-10-09)** : deux modèles fine-tunés sont publiés en privé, Qwen3.5-0.8B (`MissawB/irl-guard-decision-qwen`, avec sa variante `-lora`) et Laya multilingue (`MissawB/irl-guard-decision-laya`).

| Modèle | Précision, chartes non vues | Fuite, chartes non vues |
|---|---|---|
| Qwen | 84,6 % | 13,6 % |
| Laya | 79,9 % | 14,7 % |

Ce taux de fuite est trop élevé pour un modèle utilisé seul.

**Incident corrigé avant cette spec** : le dépôt Laya contenait aussi les fichiers du Qwen fusionné. Le notebook partageait les dossiers `merged/` et `lora/` entre deux entraînements faits dans la même session Colab. Le dépôt a été nettoyé (commit `dcac3cf` sur le Hub, `config.json` de Laya restauré) puis vérifié : `laya.load("MissawB/irl-guard-decision-laya")` prédit correctement sur CPU.

**Décidé avec l'utilisateur**

- IRL Guard utilise un **mode combiné** : règles et modèle, la décision la plus protectrice l'emporte. Le modèle se choisit dans la config : Laya fine-tuné en local (par défaut) ou Qwen fine-tuné via Unsloth Studio.
- Une **vitrine publique** sur Hugging Face : fiches complètes, Collection et Space de démo sur **ZeroGPU** (gratuit pour un compte gratuit vérifié de plus de 30 jours, 2 Spaces au maximum).

**Critères de réussite**

1. `irl-guard process` fonctionne par défaut en mode combiné avec Laya fine-tuné en local, sans Studio.
2. Le rapport inclut `combine-laya` et `combine-qwen`. Pour chacun, le taux de fuite est inférieur ou égal à celui du modèle seul, et à celui des règles seules.
3. Le dataset, les modèles Qwen (fusionné et LoRA) et Laya sont publics, avec des fiches complètes. Une Collection « IRL Guard » les regroupe avec le Space.
4. Le Space `MissawB/irl-guard-demo` répond à une charte et une cible saisies, avec au moins les règles, Laya fine-tuné et le mode combiné. Qwen s'y ajoute s'il se charge sur ZeroGPU.

## 2. Mode combiné dans IRL Guard

### 2.1 `LayaLocalDecider` (`irl_guard/decision/laya_local.py`)

- Charge le modèle avec le paquet `laya` : `laya.load(modele)`. Le chargement est paresseux, au premier appel, puis l'agent est gardé en mémoire.
- `decide(state)` appelle `agent.predict(state, QUESTIONS)`. Sa réponse a le format de la Decision API (`{"answers": {...}}`) et passe par `parser_reponse` (`decision/laya.py`), avec les seuils de la config. La source vaut `"laya-local"`.
- Toute exception pendant `predict` ou le parsing produit `decision_repli()` et un avertissement dans le journal.
- `verifier()` charge le modèle et fait une prédiction de test. En cas d'échec, il lève `LayaIndisponible` avec le nom du modèle.

### 2.2 `CombinedDecider` (`irl_guard/decision/combine.py`)

- Prend un décideur `regles` et un décideur `modele`.
- `decide(state)` appelle les deux. L'action retenue est la plus sévère (`pause` > `flouter` > `diffuser`). Si les deux sont à égalité, c'est le modèle qui l'emporte, car il apporte des probabilités.
- La `Decision` renvoyée est celle du décideur gagnant, avec `source = "combine:<source du gagnant>"`. Le journal garde les deux décisions grâce à `appels` (`Analyzer`).
- Si le modèle est en repli (`source == "fallback"`), la règle s'applique, avec un plancher à `flouter` : un repli ne doit jamais aboutir à une diffusion.

### 2.3 Configuration et ligne de commande

Nouvelle section YAML :

```yaml
decideur:
  modele: laya-local                                 # laya-local | studio
  laya_local: MissawB/irl-guard-decision-laya
```

En mode `studio`, on utilise `laya.url` et `laya.modele`, déjà présents.

`--decider` accepte `rules`, `laya` (Studio seul, comme aujourd'hui), `laya-local` (seul) et `combine`. **Le défaut devient `combine`.** En mode combiné, le modèle se choisit avec `decideur.modele`.

### 2.4 Évaluation

Deux nouvelles approches dans `irl-guard evaluer` : `combine-laya` et `combine-qwen`. Elles combinent ligne par ligne les prédictions des règles et celles du fichier exporté (`laya-ft` ou `qwen-ft`), selon la même règle que `CombinedDecider`. Elles ne demandent aucun appel de modèle. Si les prédictions du modèle sont absentes, l'approche est marquée absente.

### 2.5 Correction du notebook

- Les dossiers de sauvegarde deviennent `lora-<MODELE>` et `merged-<MODELE>`, et sont supprimés (`shutil.rmtree`) avant chaque sauvegarde.
- L'envoi publie seulement ces dossiers.
- Une cellule finale liste les fichiers publiés et vérifie que `config.json` correspond au bon modèle : `model_type` vaut `laya` pour Laya, ou commence par `qwen` pour Qwen.

## 3. Vitrine Hugging Face

### 3.1 Fiches (`irl_guard/hub/cartes.py`)

Elles sont générées depuis `docs/resultats/rapport.md`, relu par un parseur de tableaux Markdown, et depuis des constantes du projet.

| Fiche | Métadonnées YAML | Sections |
|---|---|---|
| Dataset `irl-guard-decisions` | `license: cc-by-4.0`, `language: [fr]`, `task_categories: [text-classification]`, `size_categories`, `pretty_name` | Description ; format des lignes ; génération (clauses, gabarits, partition) ; jeux et répartition des labels ; **avertissement de contenu** (insultes, données personnelles fictives Faker) ; limites ; citation du projet |
| Modèle `irl-guard-decision-qwen` | `license: apache-2.0`, `base_model: unsloth/Qwen3.5-0.8B`, `datasets: [MissawB/irl-guard-decisions]`, `language: [fr]`, `library_name: unsloth`, `tags` | Résumé ; usage (Unsloth Studio + `laya.modele`, IRL Guard `--decider combine`) ; entraînement ; **tableau de résultats** (les 6 approches et les 2 jeux, plus les approches combinées si présentes) ; limites (chartes synthétiques, fuite sans le mode combiné) |
| Modèle `irl-guard-decision-qwen-lora` | `base_model: unsloth/Qwen3.5-0.8B`, `library_name: peft`, `license: apache-2.0` | Courte : adaptateurs LoRA, renvoi au modèle fusionné |
| Modèle `irl-guard-decision-laya` | `license: apache-2.0`, `base_model: convaiinnovations/laya-multilingual`, `datasets`, `language: [fr]`, `tags` | Comme Qwen, avec en plus l'usage `laya.load("MissawB/irl-guard-decision-laya")` sur CPU |

### 3.2 Space `MissawB/irl-guard-demo` (`space/`)

- Gradio, `hardware: zero-a10g`, Python 3.12.
- **Saisie** :
  - la charte (zone de texte, 4 exemples prêts à l'emploi) ;
  - la catégorie et le mode domicile ;
  - le type de cible (texte ou objet), le texte ou la classe d'objet, la taille.
- **Sortie** : un tableau avec une ligne par approche (règles, Laya fine-tuné, Qwen fine-tuné, combiné Laya, combiné Qwen) : action, probabilités, risque et gravité. Un rappel montre les scores du rapport sur les chartes non vues.
- **Modèles** :
  - Laya se charge au démarrage, sur CPU.
  - Qwen se charge dans une fonction décorée par `@spaces.GPU`. Le Space essaie d'abord sans Unsloth, avec `transformers` et le `joint_schema_model.py` du dépôt, puis avec Unsloth s'il est installable.
  - En cas d'échec, Qwen est marqué « indisponible en ligne » et le reste fonctionne.
- **Code** : `space/app.py` et `space/requirements.txt`. À la publication, une copie de `src/irl_guard` est ajoutée, **sans** `perception/`, `pipeline.py`, `render.py`, `media.py`, `sources.py` ni `cli.py` : seuls la décision, la config et `state` sont nécessaires.

### 3.3 Publication (`irl_guard/hub/publication.py`, commande `irl-guard hub publier`)

Dans l'ordre :

1. **Générer les fiches** et les écrire dans `build/hub/<dépôt>/README.md`. C'est la seule étape exécutée avec `--dry-run`.
2. **Envoyer les fiches** sur chaque dépôt (`upload_file` de `README.md`).
3. **Créer ou mettre à jour le Space** : envoi du dossier `space/` et de la copie filtrée d'`irl_guard`, puis demande du matériel ZeroGPU.
4. **Changer la visibilité** : rendre publics le dataset, `-qwen`, `-qwen-lora`, `-laya` et le Space. Les `-checkpoints` restent privés. **Cette étape demande la confirmation `--public`.** Sans ce drapeau, rien ne change de visibilité.
5. **Créer ou mettre à jour la Collection** « IRL Guard » (publique) avec le dataset, les 3 modèles et le Space.

## 4. Tests (pytest, sans réseau ni GPU par défaut)

- **`CombinedDecider`** :
  - les 9 combinaisons d'actions ;
  - en cas d'égalité, le modèle gagne ;
  - un repli du modèle donne au moins `flouter` ;
  - la source vaut `combine:...`.
- **`LayaLocalDecider`** :
  - faux agent injecté, réponse au bon format ;
  - exception dans `predict`, donc repli ;
  - `verifier` sur un modèle introuvable, donc `LayaIndisponible`.

  Un test `gpu` charge le vrai modèle depuis le Hub (réseau).
- **CLI** : `--decider combine` est le défaut, et `laya-local` fonctionne avec un faux agent.
- **Évaluation** : `combine-laya` à partir de fichiers factices. On vérifie que le taux de fuite combiné est inférieur ou égal à `min(règles, modèle)` sur les mêmes lignes, et qu'une approche dont les prédictions manquent est marquée absente.
- **Notebook** : les dossiers de sauvegarde dépendent de `MODELE`, et `shutil.rmtree` est présent avant les sauvegardes.
- **Fiches** : construites depuis un rapport factice. On vérifie :
  - un frontmatter YAML valide, avec les clés attendues ;
  - le tableau de résultats et l'avertissement de contenu ;
  - aucune chaîne `hf_` ressemblant à un token.
- **Space** : `space/app.py` s'importe avec des décideurs factices, et `decider_tout(...)` renvoie une ligne par approche disponible. La copie filtrée d'`irl_guard` ne contient pas `perception/`.
- **Publication** avec un faux `HfApi` :
  - `--dry-run` n'appelle aucune méthode réseau ;
  - sans `--public`, aucune visibilité ne change ;
  - avec `--public`, seuls les dépôts listés deviennent publics, et `-checkpoints` n'est jamais touché ;
  - la Collection contient les 5 éléments.

## 5. Hors périmètre

- Démo vidéo complète dans le Space.
- Qwen3.8-27B (reporté, faute de quota Kaggle).
- Chartes écrites à la main.
- Publication sur GitHub (le dépôt n'a pas de remote).
- Réentraînement des modèles.

## 6. Dépendances ajoutées

`laya` (inférence Laya sur CPU). Le Space y ajoute `gradio` et `spaces` dans son propre `requirements.txt`.
