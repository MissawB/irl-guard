# Fine-tuning du decision model — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Générer un dataset synthétique de décisions guidées par des chartes, fine-tuner Qwen3.5-0.8B et Laya sur Colab, puis comparer règles, XGBoost, Laya brut et modèles fine-tunés dans un rapport.

**Architecture :**
- `irl_guard.donnees` produit les lignes : des clauses structurées sont rendues en texte par des gabarits, des `state` sont tirés au sort et les labels sont calculés de façon déterministe.
- Le format des lignes est celui de `LocalLLaMA/typed-decisions`, et le dataset est publié sur le Hub.
- Un notebook Colab est généré par `irl_guard.notebook` : il entraîne avec Unsloth et exporte les prédictions.
- `irl_guard.evaluation` calcule les métriques et écrit le rapport.

**Tech Stack :** Python 3.12, uv, Faker, scikit-learn, XGBoost, matplotlib, datasets, nbformat ; Unsloth (`FastDecisionModel`, `DecisionTrainer`) sur Colab uniquement.

**Spec :** `docs/superpowers/specs/2026-10-09-fine-tuning-decision-design.md` (lire avec ce plan).

## Global Constraints

- **Format des questions.** Elles sont **exactement** `irl_guard.decision.base.QUESTIONS`, à l'entraînement comme à l'inférence. Ne jamais les recopier.
- **Format d'une ligne.** Clés `id`, `state`, `questions`, `gold`, `clauses`. Les quatre dernières sont du JSON sérialisé avec `ensure_ascii=False`.
- **Format du `state`.** Celui de `construire_state` : `categorie`, `mode_domicile`, `charte`, `cible` (`type`, `classe`?, `texte`?, `conf`, `taille`), `contexte`.
- **Labels.** `gravite` vaut 0 pour diffuser, 2 pour flouter et 3 pour pause. Les distributions sont lissées : 0,9 pour la bonne réponse, et 0,1 réparti également entre les autres options.
- **Tailles par défaut.** train 4 000, val 500, test_vues 1 000, test_non_vues 1 000. Graine par défaut 0.
- **Hub.** Dataset `MissawB/irl-guard-decisions` et modèles `MissawB/irl-guard-decision-<qwen|laya>`, tous **privés**. Aucun token dans le code ni dans le notebook : sur Colab, il est lu dans le secret `HF_TOKEN`.
- **Fichiers locaux.** Données générées dans `data/decision/`, prédictions dans `data/predictions/<approche>/predictions_<jeu>.jsonl`. `data/` est ignoré par git.
- **Tests.** pytest, sans GPU ni réseau par défaut.
- **Commits.** Chaque commit se termine par `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Précision apportée à la spec, §2.2.** Une formulation se compose de trois parties : un préfixe de condition, un verbe de sanction et un sujet de risque. Environ 30 % de chaque liste de parties (au moins une) est réservé au test. Une formulation « non vue » contient au moins une partie réservée. Elle n'a donc jamais été vue à l'entraînement, et chaque combinaison (risque, sanction, condition) garde plus de 5 formulations dans chaque partition.

## Review Focus

1. **Une partie de gabarit réservée au test apparaît dans l'entraînement.** On attend zéro fuite. Tests : `test_partition_disjointe` et `test_train_n_utilise_que_des_composants_train` (Task 2).
2. **Un `state` généré dépasse le budget de tokens** (charte de 5 clauses, contexte chargé). On attend `estimer_tokens(state) <= 700`, la valeur par défaut de `budget_tokens_state`. Test : `test_states_dans_le_budget_de_tokens` (Task 5).
3. **Un texte court ou sans caractère confusable passe dans le bruit d'OCR** (« 12 », « 75011 »). On attend aucun plantage et un texte non vide. Test : `test_bruit_sur_textes_courts` (Task 4).
4. **XGBoost voit une classe absente du train** (par exemple aucune `pause`). On attend que les probabilités couvrent toutes les options, sans plantage. Test : `test_xgboost_classe_absente_du_train` (Task 8).
5. **Laya est injoignable pendant l'évaluation, ou le fichier de prédictions d'un modèle fine-tuné manque.** On attend que le rapport marque l'approche comme absente et compare les autres. Tests : `test_approche_absente_signalee` (Task 9) et `test_laya_injoignable_donne_absent` (Task 9).

---

## Structure des fichiers

```
src/irl_guard/
  config.py                    (modif.) + EvaluationConfig (matrice de coûts)
  donnees/__init__.py
  donnees/clauses.py           Clause, constantes, condition_remplie, appliquer_charte, charte_valide
  donnees/gabarits.py          SUJETS, VERBES, PREFIXES, Composants, partitionner, choisir_composants, composer
  donnees/chartes.py           RESERVEES, PREAMBULES, contient_reservee, tirer_clauses, rendre_charte
  donnees/cibles.py            CibleTiree, tirer_cible, bruiter
  donnees/lignes.py            construire_gold, tirer_contexte, generer_ligne, generer_jeux, ecrire_jeux, lire_jeu
  donnees/publication.py       construire_datasetdict, publier
  notebook.py                  construire_notebook, ecrire_notebook (+ __main__)
  evaluation/__init__.py
  evaluation/metriques.py      Resultat, action_predite, niveau_gravite, ece_et_fiabilite, evaluer_predictions
  evaluation/predicteurs.py    reponses_depuis_decision, predire_avec_decider, charger_predictions
  evaluation/xgb.py            features_structurees, PredicteurXGBoost
  evaluation/rapport.py        APPROCHES, JEUX_TEST, obtenir_predictions, ecrire_rapport
  cli.py                       (modif.) + groupe `donnees` (generer, publier) + commande `evaluer`
notebooks/finetune_decision.ipynb   généré par `python -m irl_guard.notebook`
tests/test_clauses.py  test_gabarits.py  test_chartes.py  test_cibles.py  test_lignes.py
tests/test_publication.py  test_notebook.py  test_metriques.py  test_xgb.py  test_rapport.py
```

---

### Task 1 : Clauses, règle d'application et dépendances

**Files:**
- Modify: `pyproject.toml` (dépendances), `.gitignore` (+ `data/`)
- Create: `src/irl_guard/donnees/__init__.py`, `src/irl_guard/donnees/clauses.py`
- Test: `tests/test_clauses.py`

**Interfaces:**
- Consumes : rien
- Produces :
  - constantes `RISQUES_CHARTE = ("localisation", "donnee_perso", "arme", "propos")`, `SANCTIONS = ("diffuser", "flouter", "pause")`, `TYPES_CONDITION = ("toujours", "domicile", "sauf_categorie")`, `CATEGORIES_CLAUSE = ("cuisine", "gaming", "musique", "sport")`, `SEVERITE`
  - `Clause(risque, sanction, condition)` (frozen), avec `.type_condition`, `.categorie` (str, `""` hors `sauf_categorie`), `.signature -> (risque, sanction, type_condition)` et `.to_dict()`
  - `condition_remplie(clause, categorie, mode_domicile) -> bool`
  - `appliquer_charte(clauses, risque, categorie, mode_domicile) -> str`
  - `charte_valide(clauses) -> bool`

- [ ] **Step 1 : Ajouter les dépendances**

Dans `pyproject.toml`, ajouter à la liste `dependencies` :

```toml
    "faker>=30",
    "scikit-learn>=1.5",
    "xgboost>=2.1",
    "matplotlib>=3.9",
    "datasets>=3.0",
    "nbformat>=5.10",
```

Ajouter la ligne `data/` à la fin de `.gitignore`.

Run : `uv sync`
Attendu : se termine sans erreur.

- [ ] **Step 2 : Écrire les tests**

`tests/test_clauses.py` :

```python
import pytest

from irl_guard.donnees.clauses import Clause, appliquer_charte, charte_valide, condition_remplie

ROUGE = Clause("localisation", "flouter", "toujours")


def test_proprietes_de_clause():
    c = Clause("arme", "flouter", "sauf_categorie:cuisine")
    assert c.type_condition == "sauf_categorie"
    assert c.categorie == "cuisine"
    assert c.signature == ("arme", "flouter", "sauf_categorie")
    assert c.to_dict() == {"risque": "arme", "sanction": "flouter", "condition": "sauf_categorie:cuisine"}
    assert ROUGE.categorie == ""


@pytest.mark.parametrize("condition, categorie, domicile, attendu", [
    ("toujours", "IRL", False, True),
    ("domicile", "IRL", False, False),
    ("domicile", "IRL", True, True),
    ("sauf_categorie:cuisine", "Cuisine", False, False),
    ("sauf_categorie:cuisine", "IRL", False, True),
])
def test_condition_remplie(condition, categorie, domicile, attendu):
    assert condition_remplie(Clause("arme", "flouter", condition), categorie, domicile) is attendu


def test_condition_inconnue():
    with pytest.raises(ValueError):
        condition_remplie(Clause("arme", "flouter", "parfois"), "IRL", False)


def test_couteau_interdit_sauf_cuisine():
    clauses = [Clause("arme", "flouter", "sauf_categorie:cuisine")]
    assert appliquer_charte(clauses, "arme", "IRL", False) == "flouter"
    assert appliquer_charte(clauses, "arme", "Cuisine", False) == "diffuser"


def test_la_plus_protectrice_l_emporte():
    clauses = [Clause("localisation", "flouter", "toujours"), Clause("localisation", "pause", "domicile")]
    assert appliquer_charte(clauses, "localisation", "IRL", False) == "flouter"
    assert appliquer_charte(clauses, "localisation", "IRL", True) == "pause"


def test_risque_non_mentionne_ou_aucun_est_diffuse():
    assert appliquer_charte([ROUGE], "propos", "IRL", True) == "diffuser"
    assert appliquer_charte([ROUGE], "aucun", "IRL", True) == "diffuser"


def test_autorisation_explicite():
    assert appliquer_charte([Clause("propos", "diffuser", "toujours")], "propos", "IRL", False) == "diffuser"


def test_charte_valide():
    assert charte_valide([ROUGE, Clause("arme", "pause", "toujours")])
    assert not charte_valide([ROUGE])  # moins de 2 clauses
    assert not charte_valide([ROUGE, Clause("localisation", "pause", "toujours")])  # même (risque, condition)
    assert not charte_valide([Clause(r, "flouter", t) for r in ("localisation", "arme", "propos")
                              for t in ("toujours", "domicile")])  # plus de 5 clauses
```

- [ ] **Step 3 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_clauses.py -q`
Attendu : échec par `ModuleNotFoundError: No module named 'irl_guard.donnees'`

- [ ] **Step 4 : Implémenter**

`src/irl_guard/donnees/__init__.py` :

```python
"""Données synthétiques pour fine-tuner le decision model : chartes, states et labels."""
```

`src/irl_guard/donnees/clauses.py` :

```python
"""Clauses de charte structurées et règle d'application déterministe (spec §2.1)."""

from __future__ import annotations

from dataclasses import dataclass

RISQUES_CHARTE = ("localisation", "donnee_perso", "arme", "propos")
SANCTIONS = ("diffuser", "flouter", "pause")
TYPES_CONDITION = ("toujours", "domicile", "sauf_categorie")
CATEGORIES_CLAUSE = ("cuisine", "gaming", "musique", "sport")
SEVERITE = {"diffuser": 0, "flouter": 1, "pause": 2}


@dataclass(frozen=True)
class Clause:
    risque: str
    sanction: str
    condition: str  # "toujours", "domicile" ou "sauf_categorie:<nom>"

    @property
    def type_condition(self) -> str:
        return self.condition.split(":", 1)[0]

    @property
    def categorie(self) -> str:
        return self.condition.split(":", 1)[1] if ":" in self.condition else ""

    @property
    def signature(self) -> tuple[str, str, str]:
        return (self.risque, self.sanction, self.type_condition)

    def to_dict(self) -> dict:
        return {"risque": self.risque, "sanction": self.sanction, "condition": self.condition}


def condition_remplie(clause: Clause, categorie: str, mode_domicile: bool) -> bool:
    t = clause.type_condition
    if t == "toujours":
        return True
    if t == "domicile":
        return mode_domicile
    if t == "sauf_categorie":
        return categorie.lower() != clause.categorie.lower()
    raise ValueError(f"condition inconnue : {clause.condition}")


def appliquer_charte(clauses: list[Clause], risque: str, categorie: str, mode_domicile: bool) -> str:
    """Sanction de la clause applicable la plus protectrice ; diffuser si aucune ne s'applique."""
    if risque == "aucun":
        return "diffuser"
    retenues = [c.sanction for c in clauses
                if c.risque == risque and condition_remplie(c, categorie, mode_domicile)]
    return max(retenues, key=SEVERITE.__getitem__, default="diffuser")


def charte_valide(clauses: list[Clause]) -> bool:
    paires = [(c.risque, c.type_condition) for c in clauses]
    return 2 <= len(clauses) <= 5 and len(paires) == len(set(paires))
```

- [ ] **Step 5 : Lancer les tests**

Run : `uv run pytest tests/test_clauses.py -q`
Attendu : 12 tests PASS

- [ ] **Step 6 : Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/irl_guard/donnees tests/test_clauses.py
git commit -m "feat: clauses de charte et règle d'application déterministe" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2 : Gabarits de formulation et partition train/test

**Files:**
- Create: `src/irl_guard/donnees/gabarits.py`
- Test: `tests/test_gabarits.py`

**Interfaces:**
- Consumes : `Clause`, `RISQUES_CHARTE`, `SANCTIONS`, `TYPES_CONDITION` (Task 1)
- Produces :
  - `SUJETS: dict[risque, tuple[str, ...]]`, `VERBES: dict[sanction, tuple[str, ...]]`, `PREFIXES: dict[type_condition, tuple[str, ...]]`, `PART_TEST = 0.3`
  - `Composants(sujets, verbes, prefixes)` (frozen ; les trois sont des dicts de tuples)
  - `partitionner(graine: int) -> tuple[Composants, Composants]`, qui renvoie (train, test)
  - `choisir_composants(clause, rng, train, test, mode) -> tuple[str, str, str]`, qui renvoie (sujet, verbe, prefixe) ; `mode` vaut `"train"` ou `"test"`
  - `composer(sujet, verbe, prefixe, categorie) -> str`

- [ ] **Step 1 : Écrire les tests**

`tests/test_gabarits.py` :

```python
import random

from irl_guard.donnees.clauses import RISQUES_CHARTE, SANCTIONS, TYPES_CONDITION, Clause
from irl_guard.donnees.gabarits import (
    PREFIXES, SUJETS, VERBES, choisir_composants, composer, partitionner,
)


def test_composer():
    assert composer("les noms de rue", "Floute {sujet}.", "", "") == "Floute les noms de rue."
    assert composer("les noms de rue", "Floute {sujet}.", "En mode domicile, ", "") == \
        "En mode domicile, floute les noms de rue."
    assert composer("les couteaux", "{Sujet} : c'est autorisé.", "Sauf en live {cat}, ", "cuisine") == \
        "Sauf en live cuisine, les couteaux : c'est autorisé."


def test_partition_deterministe():
    assert partitionner(0) == partitionner(0)
    assert partitionner(0) != partitionner(1)


def test_partition_disjointe():
    train, test = partitionner(0)
    for source, cle in ((SUJETS, "sujets"), (VERBES, "verbes"), (PREFIXES, "prefixes")):
        for nom, elements in source.items():
            tr, te = getattr(train, cle)[nom], getattr(test, cle)[nom]
            assert set(tr) | set(te) == set(elements)
            assert not set(tr) & set(te)
            assert len(te) >= 1 and len(tr) >= 1


def test_au_moins_5_formulations_par_combinaison_et_par_partition():
    train, test = partitionner(0)
    for r in RISQUES_CHARTE:
        for s in SANCTIONS:
            for t in TYPES_CONDITION:
                total = len(SUJETS[r]) * len(VERBES[s]) * len(PREFIXES[t])
                n_train = len(train.sujets[r]) * len(train.verbes[s]) * len(train.prefixes[t])
                assert n_train >= 5
                assert total - n_train >= 5  # formulations contenant au moins une partie réservée


def test_train_n_utilise_que_des_composants_train():
    train, test = partitionner(0)
    rng = random.Random(1)
    clause = Clause("localisation", "flouter", "domicile")
    for _ in range(200):
        sujet, verbe, prefixe = choisir_composants(clause, rng, train, test, "train")
        assert sujet in train.sujets["localisation"]
        assert verbe in train.verbes["flouter"]
        assert prefixe in train.prefixes["domicile"]


def test_test_utilise_au_moins_un_composant_reserve():
    train, test = partitionner(0)
    rng = random.Random(2)
    clause = Clause("propos", "pause", "toujours")
    for _ in range(200):
        sujet, verbe, prefixe = choisir_composants(clause, rng, train, test, "test")
        assert (sujet in test.sujets["propos"] or verbe in test.verbes["pause"]
                or prefixe in test.prefixes["toujours"])
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_gabarits.py -q`
Attendu : échec par `ModuleNotFoundError`

- [ ] **Step 3 : Implémenter**

`src/irl_guard/donnees/gabarits.py` :

```python
"""Formulations françaises des clauses : préfixe de condition + verbe de sanction + sujet de risque.

Environ 30 % de chaque liste de composants est réservé au test « chartes non vues » : une
formulation de test contient au moins un composant réservé, donc jamais vue à l'entraînement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from irl_guard.donnees.clauses import Clause

PART_TEST = 0.3

SUJETS = {
    "localisation": ("les noms de rue", "les numéros de maison", "les adresses postales",
                     "les panneaux de rue", "les codes postaux", "les plaques de rue"),
    "donnee_perso": ("les adresses e-mail", "les numéros de téléphone", "les noms des gens",
                     "les infos personnelles", "les coordonnées perso", "les données privées"),
    "arme": ("les couteaux", "les armes à feu", "les lames", "les objets tranchants",
             "les armes blanches", "les ciseaux"),
    "propos": ("les insultes", "les gros mots", "les propos haineux", "les injures",
               "les messages insultants", "les grossièretés"),
}

VERBES = {
    "diffuser": ("Tu peux montrer {sujet}.", "Je n'ai aucun problème avec {sujet}.",
                 "{Sujet} : c'est autorisé.", "Pas besoin de cacher {sujet}.", "Laisse passer {sujet}."),
    "flouter": ("Floute {sujet}.", "Cache {sujet} à l'écran.", "Masque {sujet}.",
                "Je ne veux pas voir {sujet} en clair.", "{Sujet} : flou obligatoire."),
    "pause": ("Coupe le live si on voit {sujet}.", "{Sujet} : mets le stream en pause.",
              "Passe en écran de pause dès que {sujet} apparaissent à l'écran.",
              "Interromps la diffusion si {sujet} apparaissent.", "Dès qu'on voit {sujet}, pause immédiate."),
}

PREFIXES = {
    "toujours": ("", "Dans tous les cas, ", "Quel que soit le contexte, ", "En toutes circonstances, "),
    "domicile": ("En mode domicile, ", "Quand je suis près de chez moi, ", "Près de mon domicile, ",
                 "Lorsque le mode domicile est activé, ", "Autour de chez moi, "),
    "sauf_categorie": ("Sauf en live {cat}, ", "Hors stream {cat}, ", "Excepté pendant les lives {cat}, ",
                       "En dehors des streams {cat}, "),
}


@dataclass(frozen=True)
class Composants:
    sujets: dict[str, tuple[str, ...]]
    verbes: dict[str, tuple[str, ...]]
    prefixes: dict[str, tuple[str, ...]]


def _decouper(elements: tuple[str, ...], rng: random.Random) -> tuple[tuple[str, ...], tuple[str, ...]]:
    melanges = list(elements)
    rng.shuffle(melanges)
    n_test = max(1, round(len(melanges) * PART_TEST))
    return tuple(melanges[n_test:]), tuple(melanges[:n_test])


def partitionner(graine: int) -> tuple[Composants, Composants]:
    rng = random.Random(graine)
    parts: dict[str, tuple[dict, dict]] = {}
    for cle, source in (("sujets", SUJETS), ("verbes", VERBES), ("prefixes", PREFIXES)):
        train, test = {}, {}
        for nom in sorted(source):
            train[nom], test[nom] = _decouper(source[nom], rng)
        parts[cle] = (train, test)
    return (Composants(*(parts[c][0] for c in ("sujets", "verbes", "prefixes"))),
            Composants(*(parts[c][1] for c in ("sujets", "verbes", "prefixes"))))


def choisir_composants(clause: Clause, rng: random.Random, train: Composants, test: Composants,
                       mode: str) -> tuple[str, str, str]:
    sources = [(train.sujets[clause.risque], test.sujets[clause.risque]),
               (train.verbes[clause.sanction], test.verbes[clause.sanction]),
               (train.prefixes[clause.type_condition], test.prefixes[clause.type_condition])]
    if mode == "train":
        return tuple(rng.choice(tr) for tr, _ in sources)
    if mode != "test":
        raise ValueError(f"mode inconnu : {mode}")
    depuis_test = [rng.random() < 0.5 for _ in sources]
    if not any(depuis_test):
        depuis_test[rng.randrange(len(sources))] = True
    return tuple(rng.choice(te if t else tr) for (tr, te), t in zip(sources, depuis_test))


def composer(sujet: str, verbe: str, prefixe: str, categorie: str) -> str:
    phrase = verbe.format(sujet=sujet, Sujet=sujet[0].upper() + sujet[1:])
    prefixe = prefixe.format(cat=categorie)
    if prefixe:
        phrase = prefixe + phrase[0].lower() + phrase[1:]
    return phrase
```

- [ ] **Step 4 : Lancer les tests**

Run : `uv run pytest tests/test_gabarits.py -q`
Attendu : 6 tests PASS

- [ ] **Step 5 : Commit**

```bash
git add src/irl_guard/donnees/gabarits.py tests/test_gabarits.py
git commit -m "feat: gabarits de formulation des clauses avec partition train/test" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3 : Composition et rendu des chartes, combinaisons réservées

**Files:**
- Create: `src/irl_guard/donnees/chartes.py`
- Test: `tests/test_chartes.py`

**Interfaces:**
- Consumes : `Clause`, `RISQUES_CHARTE`, `SANCTIONS`, `TYPES_CONDITION`, `CATEGORIES_CLAUSE`, `charte_valide` (Task 1) ; `Composants`, `choisir_composants`, `composer` (Task 2)
- Produces :
  - `RESERVEES: tuple[tuple[signature, signature], ...]`, `PREAMBULES`
  - `contient_reservee(clauses) -> bool`
  - `tirer_clauses(rng, reservee: bool) -> list[Clause]` (charte valide ; contient une paire réservée si et seulement si `reservee`)
  - `rendre_charte(clauses, rng, train, test, mode) -> str`

- [ ] **Step 1 : Écrire les tests**

`tests/test_chartes.py` :

```python
import random

from irl_guard.donnees.chartes import RESERVEES, contient_reservee, rendre_charte, tirer_clauses
from irl_guard.donnees.clauses import Clause, charte_valide
from irl_guard.donnees.gabarits import partitionner


def test_contient_reservee():
    (a, b) = RESERVEES[0]
    clauses = [Clause(a[0], a[1], a[2]), Clause(b[0], b[1], b[2])]
    assert contient_reservee(clauses)
    assert not contient_reservee(clauses[:1] + [Clause("arme", "flouter", "domicile")])


def test_tirage_non_reserve():
    rng = random.Random(0)
    for _ in range(300):
        clauses = tirer_clauses(rng, reservee=False)
        assert charte_valide(clauses) and not contient_reservee(clauses)


def test_tirage_reserve():
    rng = random.Random(0)
    for _ in range(300):
        clauses = tirer_clauses(rng, reservee=True)
        assert charte_valide(clauses) and contient_reservee(clauses)


def test_sauf_categorie_porte_une_categorie():
    rng = random.Random(3)
    for _ in range(200):
        for c in tirer_clauses(rng, reservee=False):
            if c.type_condition == "sauf_categorie":
                assert c.categorie in ("cuisine", "gaming", "musique", "sport")


def test_rendu_contient_chaque_clause():
    train, test = partitionner(0)
    rng = random.Random(4)
    clauses = [Clause("arme", "flouter", "sauf_categorie:cuisine"), Clause("propos", "pause", "toujours")]
    texte = rendre_charte(clauses, rng, train, test, "train")
    assert "cuisine" in texte
    assert texte.count(".") >= 2
    assert texte == texte.strip() and "  " not in texte
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_chartes.py -q`
Attendu : échec par `ModuleNotFoundError`

- [ ] **Step 3 : Implémenter**

`src/irl_guard/donnees/chartes.py` :

```python
"""Tirage des chartes (ensembles de clauses) et rendu en texte."""

from __future__ import annotations

import random

from irl_guard.donnees.clauses import (
    CATEGORIES_CLAUSE, RISQUES_CHARTE, SANCTIONS, TYPES_CONDITION, Clause, charte_valide,
)
from irl_guard.donnees.gabarits import Composants, choisir_composants, composer

# Paires de signatures (risque, sanction, type_condition) réservées au test « chartes non vues ».
RESERVEES = (
    (("arme", "diffuser", "toujours"), ("propos", "pause", "toujours")),
    (("localisation", "pause", "domicile"), ("donnee_perso", "diffuser", "toujours")),
    (("arme", "pause", "sauf_categorie"), ("localisation", "diffuser", "toujours")),
)
PREAMBULES = ("", "Live tout public.", "Ambiance détendue.", "Merci de respecter ces règles.", "Règles du stream :")


def contient_reservee(clauses: list[Clause]) -> bool:
    signatures = {c.signature for c in clauses}
    return any(a in signatures and b in signatures for a, b in RESERVEES)


def _clause(risque: str, sanction: str, type_condition: str, rng: random.Random) -> Clause:
    condition = f"sauf_categorie:{rng.choice(CATEGORIES_CLAUSE)}" if type_condition == "sauf_categorie" \
        else type_condition
    return Clause(risque, sanction, condition)


def _clause_aleatoire(rng: random.Random) -> Clause:
    return _clause(rng.choice(RISQUES_CHARTE), rng.choice(SANCTIONS), rng.choice(TYPES_CONDITION), rng)


def tirer_clauses(rng: random.Random, reservee: bool) -> list[Clause]:
    while True:
        n = rng.randint(2, 5)
        clauses: list[Clause] = []
        if reservee:
            a, b = rng.choice(RESERVEES)
            clauses = [_clause(*a, rng), _clause(*b, rng)]
        while len(clauses) < n:
            candidate = _clause_aleatoire(rng)
            if (candidate.risque, candidate.type_condition) not in {(c.risque, c.type_condition) for c in clauses}:
                clauses.append(candidate)
        rng.shuffle(clauses)
        if charte_valide(clauses) and contient_reservee(clauses) == reservee:
            return clauses


def rendre_charte(clauses: list[Clause], rng: random.Random, train: Composants, test: Composants,
                  mode: str) -> str:
    phrases = [rng.choice(PREAMBULES)]
    for clause in clauses:
        sujet, verbe, prefixe = choisir_composants(clause, rng, train, test, mode)
        phrases.append(composer(sujet, verbe, prefixe, clause.categorie))
    return " ".join(p for p in phrases if p)
```

- [ ] **Step 4 : Lancer les tests**

Run : `uv run pytest tests/test_chartes.py -q`
Attendu : 5 tests PASS

- [ ] **Step 5 : Commit**

```bash
git add src/irl_guard/donnees/chartes.py tests/test_chartes.py
git commit -m "feat: tirage et rendu des chartes avec combinaisons réservées au test" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4 : Cibles et bruit d'OCR

**Files:**
- Create: `src/irl_guard/donnees/cibles.py`
- Test: `tests/test_cibles.py`

**Interfaces:**
- Consumes : rien du projet (Faker)
- Produces :
  - `CibleTiree(risque, type, texte, classe)` (frozen ; `type` vaut `"texte"` ou `"objet"`)
  - `tirer_cible(rng, faker) -> CibleTiree`
  - `bruiter(texte, rng) -> str`

- [ ] **Step 1 : Écrire les tests**

`tests/test_cibles.py` :

```python
import random

from faker import Faker

from irl_guard.decision.rules import RulesDecider
from irl_guard.config import SessionConfig
from irl_guard.donnees.cibles import bruiter, tirer_cible


def _faker():
    f = Faker("fr_FR")
    f.seed_instance(0)
    return f


def test_tous_les_risques_sont_tires():
    rng, faker = random.Random(0), _faker()
    cibles = [tirer_cible(rng, faker) for _ in range(500)]
    assert {c.risque for c in cibles} == {"localisation", "donnee_perso", "arme", "propos", "aucun"}


def test_coherence_type_texte_classe():
    rng, faker = random.Random(1), _faker()
    for _ in range(300):
        c = tirer_cible(rng, faker)
        if c.risque == "arme":
            assert c.type == "objet" and c.classe in ("couteau", "ciseaux") and c.texte is None
        else:
            assert c.type == "texte" and c.classe is None and c.texte


def test_cibles_reconnues_par_les_regles():
    # Les textes générés sans bruit doivent être cohérents avec leur risque (contrôle de vraisemblance).
    rules = RulesDecider(SessionConfig(charte="x"))
    rng, faker = random.Random(2), _faker()
    for _ in range(300):
        c = tirer_cible(rng, faker)
        if c.risque in ("localisation", "propos"):
            state = {"categorie": "IRL", "mode_domicile": False, "charte": "x",
                     "cible": {"type": "texte", "texte": c.texte, "conf": 0.9, "taille": "moyenne"}, "contexte": []}
            assert rules.decide(state).risque == c.risque, c.texte


def test_bruit_modifie_le_texte_le_plus_souvent():
    rng = random.Random(3)
    modifies = sum(bruiter("Rue Nationale", rng) != "Rue Nationale" for _ in range(200))
    assert modifies > 150


def test_bruit_sur_textes_courts():
    rng = random.Random(4)
    for texte in ("12", "75011", "a", "Bd"):
        for _ in range(100):
            assert bruiter(texte, rng)


def test_bruit_deterministe():
    assert bruiter("Rue Nationale", random.Random(5)) == bruiter("Rue Nationale", random.Random(5))
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_cibles.py -q`
Attendu : échec par `ModuleNotFoundError`

- [ ] **Step 3 : Implémenter**

`src/irl_guard/donnees/cibles.py` :

```python
"""Tirage de la cible d'un state (texte lu ou objet détecté) et bruit d'OCR simulé."""

from __future__ import annotations

import random
from dataclasses import dataclass

from faker import Faker

TYPES_VOIE = ("Rue", "Avenue", "Boulevard", "Place", "Impasse", "Allée", "Chemin", "Quai")
NOMS_VOIE = ("Nationale", "Victor Hugo", "de la République", "Jean Jaurès", "des Lilas", "du Port",
             "Pasteur", "de la Gare")
INSULTES = ("Sale connard", "Gros batard", "T'es qu'une pute", "Espèce de salaud", "Connasse",
            "Ferme-la connard", "fdp", "ntm")
NEUTRES = ("SOLDES -50 %", "Boulangerie Paul", "Bienvenue", "Ouvert 7j/7", "Café des Amis", "Pharmacie",
           "Promo d'été", "Sortie", "Pizza 24h/24", "Merci de votre visite")
CONFUSIONS = {"O": "0", "o": "0", "l": "1", "I": "1", "e": "é", "a": "à", "S": "5", "B": "8"}


@dataclass(frozen=True)
class CibleTiree:
    risque: str
    type: str
    texte: str | None
    classe: str | None


def _localisation(rng: random.Random) -> str:
    tirage = rng.random()
    if tirage < 0.5:
        return f"{rng.choice(TYPES_VOIE)} {rng.choice(NOMS_VOIE)}"
    if tirage < 0.75:
        return f"{rng.randint(1, 180)}{rng.choice(('', ' bis', ' ter'))}"
    return f"{rng.randint(1, 95):02d}{rng.randint(0, 999):03d}"


def _donnee_perso(rng: random.Random, faker: Faker) -> str:
    return rng.choice((faker.email, faker.phone_number, faker.name))()


def tirer_cible(rng: random.Random, faker: Faker) -> CibleTiree:
    risque = rng.choice(("localisation", "donnee_perso", "arme", "propos", "aucun"))
    if risque == "arme":
        return CibleTiree(risque, "objet", None, rng.choice(("couteau", "ciseaux")))
    if risque == "localisation":
        texte = _localisation(rng)
    elif risque == "donnee_perso":
        texte = _donnee_perso(rng, faker)
    elif risque == "propos":
        texte = rng.choice(INSULTES)
    else:
        texte = rng.choice(NEUTRES)
    return CibleTiree(risque, "texte", texte, None)


def bruiter(texte: str, rng: random.Random) -> str:
    """Applique 1 ou 2 erreurs de lecture OCR ; ne renvoie jamais une chaîne vide."""
    for operation in rng.sample(("confusion", "casse", "troncature", "espace"), k=rng.randint(1, 2)):
        if operation == "confusion":
            positions = [i for i, ch in enumerate(texte) if ch in CONFUSIONS]
            for i in rng.sample(positions, k=min(len(positions), rng.randint(1, 2))):
                texte = texte[:i] + CONFUSIONS[texte[i]] + texte[i + 1:]
        elif operation == "casse":
            texte = texte.upper() if rng.random() < 0.5 else texte.lower()
        elif operation == "troncature" and len(texte) > 4:
            texte = texte[:-rng.randint(1, 3)]
        elif operation == "espace" and " " in texte:
            i = rng.choice([i for i, ch in enumerate(texte) if ch == " "])
            texte = texte[:i] + texte[i + 1:]
    return texte
```

- [ ] **Step 4 : Lancer les tests**

Run : `uv run pytest tests/test_cibles.py -q`
Attendu : 6 tests PASS. Si `test_cibles_reconnues_par_les_regles` échoue sur un texte de localisation, c'est qu'un nom de `TYPES_VOIE` ne figure pas dans la regex `VOIE` de `decision/rules.py` : corriger la liste `TYPES_VOIE` (sans toucher aux règles) et le consigner.

- [ ] **Step 5 : Commit**

```bash
git add src/irl_guard/donnees/cibles.py tests/test_cibles.py
git commit -m "feat: tirage des cibles et bruit d'OCR simulé" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5 : Lignes, jeux de données, fichiers JSONL et publication

**Files:**
- Create: `src/irl_guard/donnees/lignes.py`, `src/irl_guard/donnees/publication.py`
- Modify: `src/irl_guard/cli.py` (groupe `donnees`)
- Test: `tests/test_lignes.py`, `tests/test_publication.py`

**Interfaces:**
- Consumes :
  - `appliquer_charte` et `Clause` (Task 1), `partitionner` et `Composants` (Task 2), `tirer_clauses` et `rendre_charte` (Task 3), `tirer_cible` et `bruiter` (Task 4)
  - `QUESTIONS`, `ACTIONS` et `RISQUES` de `irl_guard.decision.base`, `estimer_tokens` de `irl_guard.state`
- Produces :
  - `JEUX = ("train", "val", "test_vues", "test_non_vues")`, `TAILLES_DEFAUT`, `LISSAGE = 0.9`, `GRAVITE_PAR_ACTION`
  - `construire_gold(action, risque) -> dict`
  - `generer_ligne(id_, rng, faker, train, test, mode, reservee) -> dict`
  - `generer_jeux(graine=0, tailles=None) -> dict[str, list[dict]]`
  - `ecrire_jeux(jeux, dossier) -> list[Path]`, `lire_jeu(chemin) -> list[dict]`
  - `construire_datasetdict(dossier, cache_dir=None)`, `publier(dossier, repo) -> None`
  - CLI `irl-guard donnees generer` et `irl-guard donnees publier`

- [ ] **Step 1 : Écrire les tests**

`tests/test_lignes.py` :

```python
import json

import pytest

from irl_guard.decision.base import QUESTIONS
from irl_guard.donnees.chartes import contient_reservee
from irl_guard.donnees.clauses import Clause, appliquer_charte
from irl_guard.donnees.gabarits import partitionner
from irl_guard.donnees.lignes import construire_gold, ecrire_jeux, generer_jeux, lire_jeu
from irl_guard.state import estimer_tokens

PETIT = {"train": 120, "val": 20, "test_vues": 40, "test_non_vues": 40}


@pytest.fixture(scope="module")
def jeux():
    return generer_jeux(0, PETIT)


def test_tailles_et_ids_uniques(jeux):
    assert {nom: len(l) for nom, l in jeux.items()} == PETIT
    ids = [r["id"] for l in jeux.values() for r in l]
    assert len(ids) == len(set(ids))


def test_deterministe():
    assert generer_jeux(3, {"train": 10, "val": 1, "test_vues": 1, "test_non_vues": 1}) == \
        generer_jeux(3, {"train": 10, "val": 1, "test_vues": 1, "test_non_vues": 1})


def test_format_des_lignes(jeux):
    for r in jeux["train"][:50]:
        assert set(r) == {"id", "state", "questions", "gold", "clauses"}
        assert json.loads(r["questions"]) == QUESTIONS
        state = json.loads(r["state"])
        assert list(state) == ["categorie", "mode_domicile", "charte", "cible", "contexte"]
        gold = json.loads(r["gold"])
        for nom in ("action", "risque", "gravite"):
            assert sum(gold[nom]["probabilities"].values()) == pytest.approx(1.0)
            assert gold[nom]["label"] in gold[nom]["probabilities"]


def test_label_coherent_avec_les_clauses(jeux):
    for r in jeux["train"]:
        state, gold = json.loads(r["state"]), json.loads(r["gold"])
        clauses = [Clause(**c) for c in json.loads(r["clauses"])]
        attendu = appliquer_charte(clauses, gold["risque"]["label"], state["categorie"], state["mode_domicile"])
        assert gold["action"]["label"] == attendu


def test_combinaisons_reservees_seulement_en_test_non_vues(jeux):
    def reservee(r):
        return contient_reservee([Clause(**c) for c in json.loads(r["clauses"])])
    for nom in ("train", "val", "test_vues"):
        assert not any(reservee(r) for r in jeux[nom])
    assert any(reservee(r) for r in jeux["test_non_vues"])


def test_aucun_composant_reserve_dans_le_train(jeux):
    _, test = partitionner(0)
    reserves = [s for t in test.sujets.values() for s in t]
    reserves += [v.split("{")[0] for t in test.verbes.values() for v in t if not v.startswith("{")]
    reserves += [p.split("{")[0] for t in test.prefixes.values() for p in t if p]
    for r in jeux["train"]:
        charte = json.loads(r["state"])["charte"]
        for fragment in reserves:
            assert fragment.lower() not in charte.lower(), (fragment, charte)


def test_gold():
    gold = construire_gold("pause", "localisation")
    assert gold["action"]["probabilities"]["pause"] == pytest.approx(0.9)
    assert gold["action"]["probabilities"]["diffuser"] == pytest.approx(0.05)
    assert gold["gravite"]["label"] == "3" and gold["gravite"]["score"] == 3.0
    assert construire_gold("diffuser", "aucun")["gravite"]["label"] == "0"
    assert construire_gold("flouter", "arme")["gravite"]["label"] == "2"


def test_states_dans_le_budget_de_tokens(jeux):
    for nom in ("train", "test_non_vues"):
        for r in jeux[nom]:
            assert estimer_tokens(json.loads(r["state"])) <= 700


def test_ecriture_lecture(jeux, tmp_path):
    chemins = ecrire_jeux(jeux, tmp_path / "decision")
    assert sorted(p.name for p in chemins) == sorted(f"{n}.jsonl" for n in PETIT)
    assert lire_jeu(tmp_path / "decision" / "val.jsonl") == jeux["val"]
```

Le test `test_aucun_composant_reserve_dans_le_train` vérifie des fragments de texte. Aucun sujet, verbe ou préfixe n'est sous-chaîne d'un autre composant de la même liste (vérifié à la rédaction du plan) ; garder cette propriété si l'on ajoute des gabarits. Les sujets réservés sont cherchés tels quels. Les verbes et les préfixes réservés le sont par leur partie fixe (avant le premier `{`). Les verbes qui commencent par `{Sujet}` sont ignorés : leur partie fixe est vide.

`tests/test_publication.py` :

```python
from irl_guard.donnees import publication
from irl_guard.donnees.lignes import ecrire_jeux, generer_jeux


def test_datasetdict(tmp_path):
    ecrire_jeux(generer_jeux(0, {"train": 8, "val": 2, "test_vues": 2, "test_non_vues": 2}), tmp_path / "d")
    dd = publication.construire_datasetdict(tmp_path / "d", cache_dir=tmp_path / "cache")
    assert set(dd) == {"train", "val", "test_vues", "test_non_vues"}
    assert dd["train"].num_rows == 8
    assert set(dd["train"].column_names) == {"id", "state", "questions", "gold", "clauses"}


def test_publier_en_prive(tmp_path, monkeypatch):
    appels = {}

    class Faux:
        def push_to_hub(self, repo, private):
            appels.update(repo=repo, private=private)

    monkeypatch.setattr(publication, "construire_datasetdict", lambda dossier, cache_dir=None: Faux())
    publication.publier(tmp_path, "MissawB/irl-guard-decisions")
    assert appels == {"repo": "MissawB/irl-guard-decisions", "private": True}
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_lignes.py tests/test_publication.py -q`
Attendu : échec par `ModuleNotFoundError`

- [ ] **Step 3 : Implémenter `lignes.py`**

`src/irl_guard/donnees/lignes.py` :

```python
"""Assemblage des lignes (state, questions, gold) et des quatre jeux de données (spec §2.3 à §2.5)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from faker import Faker

from irl_guard.decision.base import ACTIONS, QUESTIONS, RISQUES
from irl_guard.donnees.chartes import rendre_charte, tirer_clauses
from irl_guard.donnees.cibles import NEUTRES, bruiter, tirer_cible
from irl_guard.donnees.clauses import appliquer_charte
from irl_guard.donnees.gabarits import Composants, partitionner

JEUX = ("train", "val", "test_vues", "test_non_vues")
TAILLES_DEFAUT = {"train": 4000, "val": 500, "test_vues": 1000, "test_non_vues": 1000}
CATEGORIES_STREAM = ("IRL", "Cuisine", "Gaming", "Musique", "Sport")
TAILLES = ("petite", "moyenne", "grande")
LISSAGE = 0.9
GRAVITE_PAR_ACTION = {"diffuser": 0, "flouter": 2, "pause": 3}
NIVEAUX = ("0", "1", "2", "3")


def _distribution(options: tuple[str, ...], bonne: str) -> dict[str, float]:
    reste = (1.0 - LISSAGE) / (len(options) - 1)
    return {o: LISSAGE if o == bonne else reste for o in options}


def construire_gold(action: str, risque: str) -> dict:
    niveau = GRAVITE_PAR_ACTION[action]
    return {
        "action": {"type": "choice", "label": action, "probabilities": _distribution(ACTIONS, action)},
        "risque": {"type": "choice", "label": risque, "probabilities": _distribution(RISQUES, risque)},
        "gravite": {"type": "score", "label": str(niveau), "probabilities": _distribution(NIVEAUX, str(niveau)),
                    "score": float(niveau)},
    }


def tirer_contexte(rng: random.Random) -> list[str]:
    contexte = []
    for _ in range(rng.randint(0, 4)):
        tirage = rng.random()
        if tirage < 0.4:
            contexte.append(f"personne x{rng.randint(1, 5)}")
        elif tirage < 0.7:
            contexte.append(f"voiture x{rng.randint(1, 3)}")
        else:
            contexte.append(f"texte: {rng.choice(NEUTRES)}")
    return contexte


def generer_ligne(id_: str, rng: random.Random, faker: Faker, train: Composants, test: Composants,
                  mode: str, reservee: bool) -> dict:
    clauses = tirer_clauses(rng, reservee)
    charte = rendre_charte(clauses, rng, train, test, mode)
    cible = tirer_cible(rng, faker)
    categorie = rng.choice(CATEGORIES_STREAM)
    mode_domicile = rng.random() < 1 / 3
    texte = cible.texte
    if texte and rng.random() < 1 / 3:
        texte = bruiter(texte, rng)

    description: dict = {"type": cible.type}
    if cible.classe:
        description["classe"] = cible.classe
    if texte:
        description["texte"] = texte
    description["conf"] = round(rng.uniform(0.5, 0.99), 2)
    description["taille"] = rng.choice(TAILLES)
    state = {"categorie": categorie, "mode_domicile": mode_domicile, "charte": charte,
             "cible": description, "contexte": tirer_contexte(rng)}

    action = appliquer_charte(clauses, cible.risque, categorie, mode_domicile)
    return {
        "id": id_,
        "state": json.dumps(state, ensure_ascii=False),
        "questions": json.dumps(QUESTIONS, ensure_ascii=False),
        "gold": json.dumps(construire_gold(action, cible.risque), ensure_ascii=False),
        "clauses": json.dumps([c.to_dict() for c in clauses], ensure_ascii=False),
    }


def generer_jeux(graine: int = 0, tailles: dict[str, int] | None = None) -> dict[str, list[dict]]:
    tailles = {**TAILLES_DEFAUT, **(tailles or {})}
    train, test = partitionner(graine)
    rng = random.Random(graine)
    faker = Faker("fr_FR")
    faker.seed_instance(graine)
    jeux: dict[str, list[dict]] = {}
    for nom in JEUX:
        lignes = []
        for i in range(tailles[nom]):
            if nom == "test_non_vues":
                mode, reservee = "test", i % 2 == 0
            else:
                mode, reservee = "train", False
            lignes.append(generer_ligne(f"{nom}_{i:06d}", rng, faker, train, test, mode, reservee))
        jeux[nom] = lignes
    return jeux


def ecrire_jeux(jeux: dict[str, list[dict]], dossier: str | Path) -> list[Path]:
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    chemins = []
    for nom, lignes in jeux.items():
        chemin = dossier / f"{nom}.jsonl"
        with chemin.open("w", encoding="utf-8") as f:
            for ligne in lignes:
                f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
        chemins.append(chemin)
    return chemins


def lire_jeu(chemin: str | Path) -> list[dict]:
    with Path(chemin).open(encoding="utf-8") as f:
        return [json.loads(ligne) for ligne in f if ligne.strip()]
```

- [ ] **Step 4 : Implémenter `publication.py`**

`src/irl_guard/donnees/publication.py` :

```python
"""Publication des jeux JSONL en dataset privé sur le Hub Hugging Face."""

from __future__ import annotations

from pathlib import Path

from irl_guard.donnees.lignes import JEUX


def construire_datasetdict(dossier: str | Path, cache_dir: str | Path | None = None):
    from datasets import load_dataset
    fichiers = {nom: str(Path(dossier) / f"{nom}.jsonl") for nom in JEUX}
    return load_dataset("json", data_files=fichiers, cache_dir=str(cache_dir) if cache_dir else None)


def publier(dossier: str | Path, repo: str) -> None:
    construire_datasetdict(dossier).push_to_hub(repo, private=True)
```

- [ ] **Step 5 : Ajouter le groupe `donnees` à la CLI**

Dans `src/irl_guard/cli.py`, ajouter après la définition de `app` :

```python
donnees_app = typer.Typer(help="Données synthétiques pour fine-tuner le decision model.")
app.add_typer(donnees_app, name="donnees")


@donnees_app.command("generer")
def donnees_generer(
    sortie: Path = typer.Option(Path("data/decision"), "--sortie"),
    graine: int = typer.Option(0, "--graine"),
    train: int = typer.Option(4000, "--train"),
    val: int = typer.Option(500, "--val"),
    test_vues: int = typer.Option(1000, "--test-vues"),
    test_non_vues: int = typer.Option(1000, "--test-non-vues"),
) -> None:
    """Génère les quatre jeux JSONL (train, val, test_vues, test_non_vues)."""
    from irl_guard.donnees.lignes import ecrire_jeux, generer_jeux
    jeux = generer_jeux(graine, {"train": train, "val": val, "test_vues": test_vues, "test_non_vues": test_non_vues})
    for chemin in ecrire_jeux(jeux, sortie):
        typer.echo(f"{chemin} : {len(jeux[chemin.stem])} lignes")


@donnees_app.command("publier")
def donnees_publier(
    dossier: Path = typer.Option(Path("data/decision"), "--dossier", exists=True, file_okay=False),
    repo: str = typer.Option("MissawB/irl-guard-decisions", "--repo"),
) -> None:
    """Publie les jeux en dataset privé sur le Hub (nécessite `hf auth login`)."""
    from irl_guard.donnees.publication import publier
    publier(dossier, repo)
    typer.echo(f"Publié en privé : https://huggingface.co/datasets/{repo}")
```

Ajouter dans `tests/test_lignes.py` :

```python
def test_cli_generer(tmp_path):
    from typer.testing import CliRunner

    from irl_guard import cli
    r = CliRunner().invoke(cli.app, ["donnees", "generer", "--sortie", str(tmp_path / "d"), "--train", "5",
                                     "--val", "1", "--test-vues", "1", "--test-non-vues", "2"])
    assert r.exit_code == 0, r.output
    assert len(lire_jeu(tmp_path / "d" / "test_non_vues.jsonl")) == 2
```

- [ ] **Step 6 : Lancer les tests**

Run : `uv run pytest tests/test_lignes.py tests/test_publication.py -q`
Attendu : 12 tests PASS

Run : `uv run pytest -q`
Attendu : toute la suite PASS (les commandes `process` et `replay` existantes ne sont pas affectées)

- [ ] **Step 7 : Commit**

```bash
git add src/irl_guard/donnees/lignes.py src/irl_guard/donnees/publication.py src/irl_guard/cli.py tests/test_lignes.py tests/test_publication.py
git commit -m "feat: génération des jeux de décisions, export JSONL et publication sur le Hub" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6 : Notebook Colab de fine-tuning

**Files:**
- Create: `src/irl_guard/notebook.py`, `notebooks/finetune_decision.ipynb` (généré)
- Test: `tests/test_notebook.py`

**Interfaces:**
- Consumes : `QUESTIONS` (lu depuis les lignes du dataset, pas importé dans le notebook)
- Produces : `construire_notebook() -> nbformat.NotebookNode`, `ecrire_notebook(chemin) -> Path` ; exécutable par `uv run python -m irl_guard.notebook`

API Unsloth utilisée d'après la documentation (`unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth`) :
- `FastDecisionModel.from_pretrained`, `get_peft_model`, `build_dataset`, `calibrate`, `for_inference`, `predict` ;
- `DecisionTrainer` ;
- `save_pretrained`, `save_pretrained_merged`.

La signature exacte de `predict` n'est pas documentée. Le notebook affiche une réponse brute avant de convertir quoi que ce soit.

- [ ] **Step 1 : Écrire les tests**

`tests/test_notebook.py` :

```python
import ast
import re

import nbformat

from irl_guard.notebook import construire_notebook, ecrire_notebook


def _code(nb):
    return [c.source for c in nb.cells if c.cell_type == "code"]


def test_notebook_valide_et_parametre_en_tete(tmp_path):
    chemin = ecrire_notebook(tmp_path / "nb.ipynb")
    nb = nbformat.read(chemin, as_version=4)
    nbformat.validate(nb)
    assert 'MODELE = "qwen"' in _code(nb)[0]


def test_cellules_python_syntaxiquement_valides():
    for source in _code(construire_notebook()):
        lignes = [l for l in source.splitlines() if not l.lstrip().startswith(("!", "%"))]
        ast.parse("\n".join(lignes))


def test_aucun_token_en_clair():
    texte = "\n".join(c.source for c in construire_notebook().cells)
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", texte)
    assert 'userdata.get("HF_TOKEN")' in texte


def test_etapes_cles_presentes():
    texte = "\n".join(_code(construire_notebook()))
    for attendu in ("FastDecisionModel.from_pretrained", "get_peft_model", "build_dataset", "DecisionTrainer",
                    "resume_from_checkpoint", "calibrate", "save_pretrained_merged", "private=True",
                    "predictions_", "test_non_vues"):
        assert attendu in texte, attendu


def test_le_fichier_du_depot_est_a_jour():
    depot = nbformat.read("notebooks/finetune_decision.ipynb", as_version=4)
    assert [c.source for c in depot.cells] == [c.source for c in construire_notebook().cells]
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_notebook.py -q`
Attendu : échec par `ModuleNotFoundError: No module named 'irl_guard.notebook'`

- [ ] **Step 3 : Implémenter**

`src/irl_guard/notebook.py` :

````python
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
MAX_STEPS = -1   # -1 = 2 époques complètes ; 60 pour un essai rapide
REPO_DONNEES = "MissawB/irl-guard-decisions"
REPO_MODELE = f"MissawB/irl-guard-decision-{MODELE}"
BASES = {"qwen": "unsloth/Qwen3.5-0.8B", "laya": "convaiinnovations/laya-multilingual"}"""),
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
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
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
    ("code", """from huggingface_hub import HfApi

api = HfApi()
model.save_pretrained("lora")
model.save_pretrained_merged("merged")
for dossier, repo in (("lora", REPO_MODELE + "-lora"), ("merged", REPO_MODELE)):
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=dossier, repo_id=repo)
    print("Publié :", repo)"""),
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
````

- [ ] **Step 4 : Générer le notebook du dépôt**

Run : `uv run python -m irl_guard.notebook`
Attendu : `notebooks\finetune_decision.ipynb`

- [ ] **Step 5 : Lancer les tests**

Run : `uv run pytest tests/test_notebook.py -q`
Attendu : 5 tests PASS

- [ ] **Step 6 : Commit**

```bash
git add src/irl_guard/notebook.py notebooks/finetune_decision.ipynb tests/test_notebook.py
git commit -m "feat: notebook Colab de fine-tuning généré et testé" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7 : Métriques d'évaluation et matrice de coûts

**Files:**
- Modify: `src/irl_guard/config.py` (+ `EvaluationConfig`, champ `evaluation` dans `SessionConfig`)
- Create: `src/irl_guard/evaluation/__init__.py`, `src/irl_guard/evaluation/metriques.py`
- Test: `tests/test_metriques.py`

**Interfaces:**
- Consumes : `action_depuis_probas` (`decision/base.py`), `SeuilsConfig`
- Produces :
  - `EvaluationConfig.couts: dict[str, dict[str, float]]`, indexé par `couts[predit][attendu]`
  - `Resultat(n, precision_action, precision_risque, precision_gravite, taux_fuite, ece, cout_moyen, fiabilite)` (frozen ; `fiabilite` est une liste de `(confiance_moyenne, precision, effectif)`)
  - `action_predite(reponse, seuils) -> str`
  - `niveau_gravite(reponse) -> int`
  - `ece_et_fiabilite(confiances, corrects, n_classes=10) -> tuple[float, list]`
  - `evaluer_predictions(lignes, predictions, seuils, couts) -> Resultat`

Format d'une réponse, commun à tous les prédicteurs : `{"action": {"probabilities": {...}}, "risque": {"probabilities": {...}}, "gravite": {"score": float}}`.

- [ ] **Step 1 : Écrire les tests**

`tests/test_metriques.py` :

```python
import json

import pytest

from irl_guard.config import SessionConfig
from irl_guard.donnees.lignes import construire_gold
from irl_guard.evaluation.metriques import (
    action_predite, ece_et_fiabilite, evaluer_predictions, niveau_gravite,
)

CFG = SessionConfig(charte="x")


def _ligne(id_, action, risque):
    return {"id": id_, "gold": json.dumps(construire_gold(action, risque))}


def _rep(p_action, risque, score):
    return {"action": {"probabilities": dict(zip(("diffuser", "flouter", "pause"), p_action))},
            "risque": {"probabilities": {risque: 1.0}}, "gravite": {"score": score}}


def test_couts_par_defaut():
    c = CFG.evaluation.couts
    assert c["diffuser"]["flouter"] == 10 and c["diffuser"]["pause"] == 20
    assert c["pause"]["diffuser"] == 3 and c["flouter"]["flouter"] == 0


def test_action_predite_et_gravite():
    assert action_predite(_rep((0.5, 0.3, 0.2), "aucun", 0), CFG.seuils) == "flouter"
    assert niveau_gravite(_rep((1, 0, 0), "aucun", 1.6)) == 2
    assert niveau_gravite(_rep((1, 0, 0), "aucun", 7)) == 3


def test_ece_calcule_a_la_main():
    ece, points = ece_et_fiabilite([0.95, 0.95, 0.55, 0.55], [True, False, True, True])
    # classe 0.9-1.0 : confiance 0.95, précision 0.5 ; classe 0.5-0.6 : confiance 0.55, précision 1.0
    assert ece == pytest.approx(0.5 * 0.45 + 0.5 * 0.45)
    assert points == [(pytest.approx(0.55), 1.0, 2), (pytest.approx(0.95), 0.5, 2)]


def test_evaluer_predictions():
    lignes = [_ligne("a", "flouter", "localisation"), _ligne("b", "pause", "propos"),
              _ligne("c", "diffuser", "aucun"), _ligne("d", "flouter", "arme")]
    predictions = {
        "a": _rep((0.1, 0.8, 0.1), "localisation", 2.0),  # juste
        "b": _rep((0.9, 0.05, 0.05), "propos", 0.0),      # fuite grave : coût 20
        "c": _rep((0.0, 0.0, 1.0), "aucun", 3.0),         # pause injustifiée : coût 3
        "d": _rep((0.1, 0.9, 0.0), "donnee_perso", 2.0),  # action juste, risque faux
    }
    r = evaluer_predictions(lignes, predictions, CFG.seuils, CFG.evaluation.couts)
    assert r.n == 4
    assert r.precision_action == 0.5
    assert r.precision_risque == 0.75
    assert r.precision_gravite == 0.5
    assert r.taux_fuite == pytest.approx(1 / 3)  # 1 fuite sur 3 lignes à protéger
    assert r.cout_moyen == pytest.approx((0 + 20 + 3 + 0) / 4)


def test_prediction_manquante():
    with pytest.raises(ValueError, match="b"):
        evaluer_predictions([_ligne("a", "flouter", "arme"), _ligne("b", "pause", "arme")],
                            {"a": _rep((0, 1, 0), "arme", 2)}, CFG.seuils, CFG.evaluation.couts)
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_metriques.py -q`
Attendu : échec par `ModuleNotFoundError: No module named 'irl_guard.evaluation'`

- [ ] **Step 3 : Ajouter la matrice de coûts à la configuration**

Dans `src/irl_guard/config.py`, ajouter avant `class SessionConfig` :

```python
def _couts_par_defaut() -> dict[str, dict[str, float]]:
    # couts[action prédite][action attendue]
    return {
        "diffuser": {"diffuser": 0, "flouter": 10, "pause": 20},
        "flouter": {"diffuser": 1, "flouter": 0, "pause": 5},
        "pause": {"diffuser": 3, "flouter": 1, "pause": 0},
    }


class EvaluationConfig(_Strict):
    couts: dict[str, dict[str, float]] = Field(default_factory=_couts_par_defaut)
```

et dans `SessionConfig`, après `detecteurs` :

```python
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
```

- [ ] **Step 4 : Implémenter les métriques**

`src/irl_guard/evaluation/__init__.py` :

```python
"""Évaluation comparative des décideurs sur les jeux de test (spec §4)."""
```

`src/irl_guard/evaluation/metriques.py` :

```python
"""Précision, taux de fuite, ECE et coût moyen sur un jeu de test."""

from __future__ import annotations

import json
from dataclasses import dataclass

from irl_guard.config import SeuilsConfig
from irl_guard.decision.base import action_depuis_probas


@dataclass(frozen=True)
class Resultat:
    n: int
    precision_action: float
    precision_risque: float
    precision_gravite: float
    taux_fuite: float
    ece: float
    cout_moyen: float
    fiabilite: list[tuple[float, float, int]]


def _argmax(probas: dict[str, float]) -> str:
    return max(probas, key=probas.get)


def action_predite(reponse: dict, seuils: SeuilsConfig) -> str:
    return action_depuis_probas(reponse["action"]["probabilities"], seuils.pause, seuils.flou)


def niveau_gravite(reponse: dict) -> int:
    return min(max(round(reponse["gravite"]["score"]), 0), 3)


def ece_et_fiabilite(confiances: list[float], corrects: list[bool],
                     n_classes: int = 10) -> tuple[float, list[tuple[float, float, int]]]:
    classes: list[list[tuple[float, bool]]] = [[] for _ in range(n_classes)]
    for c, ok in zip(confiances, corrects):
        classes[min(int(c * n_classes), n_classes - 1)].append((c, ok))
    total = len(confiances)
    ece, points = 0.0, []
    for contenu in classes:
        if not contenu:
            continue
        confiance = sum(c for c, _ in contenu) / len(contenu)
        precision = sum(ok for _, ok in contenu) / len(contenu)
        ece += len(contenu) / total * abs(precision - confiance)
        points.append((confiance, precision, len(contenu)))
    return ece, points


def evaluer_predictions(lignes: list[dict], predictions: dict[str, dict], seuils: SeuilsConfig,
                        couts: dict[str, dict[str, float]]) -> Resultat:
    manquantes = [l["id"] for l in lignes if l["id"] not in predictions]
    if manquantes:
        raise ValueError(f"prédictions manquantes pour {len(manquantes)} lignes, dont {manquantes[:3]}")
    ok_action = ok_risque = ok_gravite = a_proteger = fuites = 0
    cout = 0.0
    confiances, corrects = [], []
    for ligne in lignes:
        gold = json.loads(ligne["gold"])
        reponse = predictions[ligne["id"]]
        attendue = gold["action"]["label"]
        predite = action_predite(reponse, seuils)
        ok_action += predite == attendue
        ok_risque += _argmax(reponse["risque"]["probabilities"]) == gold["risque"]["label"]
        ok_gravite += niveau_gravite(reponse) == int(gold["gravite"]["label"])
        if attendue != "diffuser":
            a_proteger += 1
            fuites += predite == "diffuser"
        cout += couts[predite][attendue]
        p_action = reponse["action"]["probabilities"]
        confiances.append(max(p_action.values()))
        corrects.append(_argmax(p_action) == attendue)
    n = len(lignes)
    ece, points = ece_et_fiabilite(confiances, corrects)
    return Resultat(n, ok_action / n, ok_risque / n, ok_gravite / n,
                    fuites / a_proteger if a_proteger else 0.0, ece, cout / n, points)
```

- [ ] **Step 5 : Lancer les tests**

Run : `uv run pytest tests/test_metriques.py tests/test_config.py -q`
Attendu : tout PASS (5 nouveaux + 5 existants)

- [ ] **Step 6 : Commit**

```bash
git add src/irl_guard/config.py src/irl_guard/evaluation tests/test_metriques.py
git commit -m "feat: métriques d'évaluation (précision, fuite, ECE, coût) et matrice de coûts" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8 : Prédicteurs — décideurs, fichiers et XGBoost

**Files:**
- Create: `src/irl_guard/evaluation/predicteurs.py`, `src/irl_guard/evaluation/xgb.py`
- Test: `tests/test_xgb.py` (inclut les tests des prédicteurs)

**Interfaces:**
- Consumes :
  - `Decision`, `ACTIONS`, `RISQUES` (`decision/base.py`)
  - `VOIE`, `NUMERO`, `CODE_POSTAL`, `EMAIL`, `TELEPHONE`, `MOTS_OFFENSANTS_BASE`, `normaliser` (`decision/rules.py`)
  - `generer_jeux` (Task 5)
- Produces :
  - `reponses_depuis_decision(decision) -> dict`
  - `predire_avec_decider(decider, lignes) -> dict[str, dict]`
  - `charger_predictions(chemin) -> dict[str, dict]`
  - `features_structurees(state) -> dict[str, float]`
  - `PredicteurXGBoost(graine=0)` avec `.entrainer(lignes) -> self` et `.predire(lignes) -> dict[str, dict]`

- [ ] **Step 1 : Écrire les tests**

`tests/test_xgb.py` :

```python
import json

import pytest

from irl_guard.config import SessionConfig
from irl_guard.decision.base import Decision
from irl_guard.decision.rules import RulesDecider
from irl_guard.donnees.lignes import construire_gold, generer_jeux
from irl_guard.evaluation.predicteurs import charger_predictions, predire_avec_decider, reponses_depuis_decision
from irl_guard.evaluation.xgb import PredicteurXGBoost, features_structurees


def test_reponses_depuis_decision():
    rep = reponses_depuis_decision(Decision("pause", "propos", 3))
    assert rep["action"]["probabilities"] == {"diffuser": 0.0, "flouter": 0.0, "pause": 1.0}
    assert rep["risque"]["probabilities"]["propos"] == 1.0
    assert rep["gravite"]["score"] == 3.0
    repli = reponses_depuis_decision(Decision("flouter", "inconnu", 2, {}, "fallback"))
    assert sum(repli["risque"]["probabilities"].values()) == pytest.approx(1.0)


def test_predire_avec_decider_regles():
    lignes = generer_jeux(0, {"train": 0, "val": 0, "test_vues": 5, "test_non_vues": 0})["test_vues"]
    preds = predire_avec_decider(RulesDecider(SessionConfig(charte="x")), lignes)
    assert set(preds) == {l["id"] for l in lignes}
    assert all("probabilities" in p["action"] for p in preds.values())


def test_charger_predictions(tmp_path):
    chemin = tmp_path / "p.jsonl"
    chemin.write_text(json.dumps({"id": "x", "answers": {"gravite": {"score": 1}}}) + "\n", encoding="utf-8")
    assert charger_predictions(chemin) == {"x": {"gravite": {"score": 1}}}


def test_features_structurees():
    state = {"categorie": "Cuisine", "mode_domicile": True, "charte": "x",
             "cible": {"type": "texte", "texte": "Rue Nationale", "conf": 0.9, "taille": "grande"}, "contexte": []}
    f = features_structurees(state)
    assert f["voie"] == 1 and f["email"] == 0 and f["domicile"] == 1
    assert f["categorie=cuisine"] == 1 and f["type=texte"] == 1 and f["taille=grande"] == 1


@pytest.fixture(scope="module")
def jeux():
    return generer_jeux(0, {"train": 400, "val": 0, "test_vues": 50, "test_non_vues": 0})


def test_xgboost_entraine_et_predit(jeux):
    modele = PredicteurXGBoost().entrainer(jeux["train"])
    assert modele.tfidf.transform(["test"]).shape[1] <= 500
    preds = modele.predire(jeux["test_vues"])
    assert set(preds) == {l["id"] for l in jeux["test_vues"]}
    for p in preds.values():
        assert set(p["action"]["probabilities"]) == {"diffuser", "flouter", "pause"}
        assert sum(p["action"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
        assert 0 <= p["gravite"]["score"] <= 3


def test_xgboost_classe_absente_du_train(jeux):
    # Train sans aucune ligne « pause » : les prédictions couvrent quand même les trois actions.
    sans_pause = [l for l in jeux["train"] if json.loads(l["gold"])["action"]["label"] != "pause"]
    preds = PredicteurXGBoost().entrainer(sans_pause).predire(jeux["test_vues"][:5])
    for p in preds.values():
        assert p["action"]["probabilities"]["pause"] == 0.0
        assert set(p["risque"]["probabilities"]) == {"localisation", "donnee_perso", "arme", "propos", "aucun"}
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_xgb.py -q`
Attendu : échec par `ModuleNotFoundError`

- [ ] **Step 3 : Implémenter les prédicteurs**

`src/irl_guard/evaluation/predicteurs.py` :

```python
"""Prédicteurs communs : décideur du pipeline (règles, Laya) et fichiers exportés par Colab."""

from __future__ import annotations

import json
from pathlib import Path

from irl_guard.decision.base import ACTIONS, RISQUES, Decider, Decision


def _one_hot(options: tuple[str, ...], valeur: str) -> dict[str, float]:
    if valeur not in options:
        return {o: 1.0 / len(options) for o in options}
    return {o: 1.0 if o == valeur else 0.0 for o in options}


def reponses_depuis_decision(decision: Decision) -> dict:
    return {
        "action": {"probabilities": decision.probabilites.get("action") or _one_hot(ACTIONS, decision.action)},
        "risque": {"probabilities": decision.probabilites.get("risque") or _one_hot(RISQUES, decision.risque)},
        "gravite": {"score": float(decision.gravite)},
    }


def predire_avec_decider(decider: Decider, lignes: list[dict]) -> dict[str, dict]:
    return {l["id"]: reponses_depuis_decision(decider.decide(json.loads(l["state"]))) for l in lignes}


def charger_predictions(chemin: str | Path) -> dict[str, dict]:
    with Path(chemin).open(encoding="utf-8") as f:
        return {d["id"]: d["answers"] for d in (json.loads(l) for l in f if l.strip())}
```

- [ ] **Step 4 : Implémenter XGBoost**

`src/irl_guard/evaluation/xgb.py` :

```python
"""Témoin tabulaire : XGBoost sur features structurées + TF-IDF de la charte (jamais les clauses)."""

from __future__ import annotations

import json
import re

from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from irl_guard.decision.base import ACTIONS, RISQUES
from irl_guard.decision.rules import CODE_POSTAL, EMAIL, MOTS_OFFENSANTS_BASE, NUMERO, TELEPHONE, VOIE, normaliser

INSULTE = re.compile(r"\b(" + "|".join(re.escape(m) for m in MOTS_OFFENSANTS_BASE) + r")\b")
OPTIONS = {"action": ACTIONS, "risque": RISQUES, "gravite": ("0", "1", "2", "3")}


def features_structurees(state: dict) -> dict[str, float]:
    cible = state["cible"]
    texte = normaliser(cible.get("texte") or "")
    return {
        f"type={cible['type']}": 1, f"classe={cible.get('classe') or 'aucune'}": 1,
        f"taille={cible.get('taille', 'petite')}": 1, f"categorie={state['categorie'].lower()}": 1,
        "domicile": int(bool(state["mode_domicile"])),
        "voie": int(bool(VOIE.search(texte))), "numero": int(bool(NUMERO.match(texte))),
        "code_postal": int(bool(CODE_POSTAL.search(texte))), "email": int(bool(EMAIL.search(texte))),
        "telephone": int(bool(TELEPHONE.search(texte))), "insulte": int(bool(INSULTE.search(texte))),
    }


class PredicteurXGBoost:
    def __init__(self, graine: int = 0):
        self.graine = graine
        self.dv = DictVectorizer()
        self.tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.modeles: dict[str, tuple[LabelEncoder, XGBClassifier | None]] = {}

    def _matrice(self, states: list[dict], entrainement: bool):
        structurees = [features_structurees(s) for s in states]
        chartes = [s["charte"] for s in states]
        if entrainement:
            return hstack([self.dv.fit_transform(structurees), self.tfidf.fit_transform(chartes)]).tocsr()
        return hstack([self.dv.transform(structurees), self.tfidf.transform(chartes)]).tocsr()

    def entrainer(self, lignes: list[dict]) -> PredicteurXGBoost:
        states = [json.loads(l["state"]) for l in lignes]
        golds = [json.loads(l["gold"]) for l in lignes]
        x = self._matrice(states, entrainement=True)
        for question in OPTIONS:
            encodeur = LabelEncoder().fit([g[question]["label"] for g in golds])
            y = encodeur.transform([g[question]["label"] for g in golds])
            if len(encodeur.classes_) < 2:
                self.modeles[question] = (encodeur, None)  # une seule classe : prédiction constante
                continue
            modele = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   random_state=self.graine, n_jobs=-1, verbosity=0)
            modele.fit(x, y)
            self.modeles[question] = (encodeur, modele)
        return self

    def predire(self, lignes: list[dict]) -> dict[str, dict]:
        x = self._matrice([json.loads(l["state"]) for l in lignes], entrainement=False)
        probas = {}
        for question, (encodeur, modele) in self.modeles.items():
            probas[question] = modele.predict_proba(x) if modele is not None else [[1.0]] * len(lignes)
        predictions = {}
        for i, ligne in enumerate(lignes):
            reponse = {}
            for question, options in OPTIONS.items():
                encodeur, _ = self.modeles[question]
                distribution = {o: 0.0 for o in options}
                for k, label in enumerate(encodeur.classes_):
                    distribution[str(label)] = float(probas[question][i][k])
                if question == "gravite":
                    reponse[question] = {"score": sum(int(o) * p for o, p in distribution.items()),
                                         "probabilities": distribution}
                else:
                    reponse[question] = {"probabilities": distribution}
            predictions[ligne["id"]] = reponse
        return predictions
```

- [ ] **Step 5 : Lancer les tests**

Run : `uv run pytest tests/test_xgb.py -q`
Attendu : 6 tests PASS

- [ ] **Step 6 : Commit**

```bash
git add src/irl_guard/evaluation/predicteurs.py src/irl_guard/evaluation/xgb.py tests/test_xgb.py
git commit -m "feat: prédicteurs d'évaluation (règles, Laya, fichiers) et témoin XGBoost" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9 : Rapport comparatif et commande `evaluer`

**Files:**
- Create: `src/irl_guard/evaluation/rapport.py`
- Modify: `src/irl_guard/cli.py` (commande `evaluer`)
- Test: `tests/test_rapport.py`

**Interfaces:**
- Consumes :
  - `Resultat` et `evaluer_predictions` (Task 7) ; `predire_avec_decider`, `charger_predictions` et `PredicteurXGBoost` (Task 8)
  - `lire_jeu` (Task 5) ; `RulesDecider` ; `LayaDecider` et `LayaIndisponible`
- Produces :
  - `APPROCHES = ("regles", "xgboost", "laya", "laya-ft", "qwen-ft")`, `JEUX_TEST = ("test_vues", "test_non_vues")`
  - `obtenir_predictions(approche, cfg, dossier_donnees, dossier_predictions, jeux) -> dict[str, dict | None]`
  - `ecrire_rapport(resultats: dict[str, dict[str, Resultat | None]], dossier) -> Path`
  - CLI `irl-guard evaluer`

- [ ] **Step 1 : Écrire les tests**

`tests/test_rapport.py` :

```python
import json

from irl_guard.config import SessionConfig
from irl_guard.donnees.lignes import ecrire_jeux, generer_jeux
from irl_guard.evaluation.metriques import Resultat
from irl_guard.evaluation.rapport import ecrire_rapport, obtenir_predictions

R = Resultat(10, 0.9, 0.8, 0.7, 0.05, 0.12, 1.5, [(0.55, 0.6, 4), (0.95, 0.9, 6)])
PETIT = {"train": 60, "val": 0, "test_vues": 10, "test_non_vues": 10}


def test_rapport_tableau_et_png(tmp_path):
    chemin = ecrire_rapport({"regles": {"test_vues": R, "test_non_vues": R},
                             "qwen-ft": {"test_vues": None, "test_non_vues": None}}, tmp_path)
    texte = chemin.read_text(encoding="utf-8")
    assert "| regles |" in texte and "90.0 %" in texte and "5.0 %" in texte
    assert "qwen-ft" in texte and "absent" in texte
    assert (tmp_path / "fiabilite_test_vues.png").exists()
    assert (tmp_path / "fiabilite_test_non_vues.png").exists()


def test_approche_absente_signalee(tmp_path):
    jeux = generer_jeux(0, PETIT)
    preds = obtenir_predictions("qwen-ft", SessionConfig(charte="x"), tmp_path, tmp_path / "predictions",
                                {j: jeux[j] for j in ("test_vues", "test_non_vues")})
    assert preds == {"test_vues": None, "test_non_vues": None}


def test_predictions_depuis_fichier(tmp_path):
    jeux = generer_jeux(0, PETIT)
    dossier = tmp_path / "predictions" / "laya-ft"
    dossier.mkdir(parents=True)
    with (dossier / "predictions_test_vues.jsonl").open("w", encoding="utf-8") as f:
        for l in jeux["test_vues"]:
            f.write(json.dumps({"id": l["id"], "answers": {"x": 1}}) + "\n")
    preds = obtenir_predictions("laya-ft", SessionConfig(charte="x"), tmp_path, tmp_path / "predictions",
                                {j: jeux[j] for j in ("test_vues", "test_non_vues")})
    assert len(preds["test_vues"]) == 10 and preds["test_non_vues"] is None


def test_laya_injoignable_donne_absent(tmp_path):
    jeux = generer_jeux(0, PETIT)
    cfg = SessionConfig(charte="x", laya={"url": "http://127.0.0.1:9", "timeout_s": 0.5})
    preds = obtenir_predictions("laya", cfg, tmp_path, tmp_path, {"test_vues": jeux["test_vues"]})
    assert preds == {"test_vues": None}


def test_cli_evaluer_regles_et_xgboost(tmp_path):
    from typer.testing import CliRunner

    from irl_guard import cli
    ecrire_jeux(generer_jeux(0, PETIT), tmp_path / "decision")
    config = tmp_path / "s.yaml"
    config.write_text("charte: x\n", encoding="utf-8")
    r = CliRunner().invoke(cli.app, ["evaluer", "--donnees", str(tmp_path / "decision"),
                                     "--predictions", str(tmp_path / "predictions"),
                                     "--approches", "regles,xgboost,qwen-ft", "--config", str(config),
                                     "--sortie", str(tmp_path / "resultats")])
    assert r.exit_code == 0, r.output
    texte = (tmp_path / "resultats" / "rapport.md").read_text(encoding="utf-8")
    assert "| regles |" in texte and "| xgboost |" in texte and "absent" in texte
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_rapport.py -q`
Attendu : échec par `ModuleNotFoundError`

- [ ] **Step 3 : Implémenter**

`src/irl_guard/evaluation/rapport.py` :

```python
"""Prédictions par approche et rapport Markdown comparatif avec diagrammes de fiabilité."""

from __future__ import annotations

import logging
from pathlib import Path

from irl_guard.config import SessionConfig
from irl_guard.decision.laya import LayaDecider, LayaIndisponible
from irl_guard.decision.rules import RulesDecider
from irl_guard.donnees.lignes import lire_jeu
from irl_guard.evaluation.metriques import Resultat
from irl_guard.evaluation.predicteurs import charger_predictions, predire_avec_decider
from irl_guard.evaluation.xgb import PredicteurXGBoost

log = logging.getLogger(__name__)
APPROCHES = ("regles", "xgboost", "laya", "laya-ft", "qwen-ft")
JEUX_TEST = ("test_vues", "test_non_vues")
TITRES = {"test_vues": "Chartes vues", "test_non_vues": "Chartes non vues"}


def obtenir_predictions(approche: str, cfg: SessionConfig, dossier_donnees: Path, dossier_predictions: Path,
                        jeux: dict[str, list[dict]]) -> dict[str, dict | None]:
    if approche == "regles":
        decider = RulesDecider(cfg)
        return {j: predire_avec_decider(decider, l) for j, l in jeux.items()}
    if approche == "xgboost":
        modele = PredicteurXGBoost().entrainer(lire_jeu(Path(dossier_donnees) / "train.jsonl"))
        return {j: modele.predire(l) for j, l in jeux.items()}
    if approche == "laya":
        decider = LayaDecider(cfg)
        try:
            decider.verifier()
        except LayaIndisponible as e:
            log.warning("Laya absent du rapport : %s", e)
            return {j: None for j in jeux}
        return {j: predire_avec_decider(decider, l) for j, l in jeux.items()}
    resultat = {}
    for j in jeux:
        chemin = Path(dossier_predictions) / approche / f"predictions_{j}.jsonl"
        resultat[j] = charger_predictions(chemin) if chemin.exists() else None
    return resultat


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def _diagramme(resultats: dict[str, dict[str, Resultat | None]], jeu: str, chemin: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="calibration parfaite")
    for approche, par_jeu in resultats.items():
        r = par_jeu.get(jeu)
        if r is not None and r.fiabilite:
            ax.plot([p[0] for p in r.fiabilite], [p[1] for p in r.fiabilite], marker="o", label=approche)
    ax.set_xlabel("confiance moyenne")
    ax.set_ylabel("précision observée")
    ax.set_title(f"Fiabilité — {TITRES.get(jeu, jeu)}")
    ax.legend(loc="lower right")
    fig.savefig(chemin, dpi=120, bbox_inches="tight")
    plt.close(fig)


def ecrire_rapport(resultats: dict[str, dict[str, Resultat | None]], dossier: str | Path) -> Path:
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    lignes = ["# Résultats du decision model", "",
              "Taux de fuite : part des cas à protéger (flouter ou pause attendu) que l'approche diffuse.", ""]
    jeux = sorted({j for par_jeu in resultats.values() for j in par_jeu}, key=lambda j: JEUX_TEST.index(j)
                  if j in JEUX_TEST else len(JEUX_TEST))
    for jeu in jeux:
        lignes += [f"## {TITRES.get(jeu, jeu)}", "",
                   "| Approche | n | Action | Risque | Gravité | Fuite | ECE | Coût moyen |",
                   "|---|---|---|---|---|---|---|---|"]
        for approche, par_jeu in resultats.items():
            r = par_jeu.get(jeu)
            if r is None:
                lignes.append(f"| {approche} | absent | | | | | | |")
            else:
                lignes.append(f"| {approche} | {r.n} | {_pct(r.precision_action)} | {_pct(r.precision_risque)} | "
                              f"{_pct(r.precision_gravite)} | {_pct(r.taux_fuite)} | {r.ece:.3f} | {r.cout_moyen:.2f} |")
        image = dossier / f"fiabilite_{jeu}.png"
        _diagramme(resultats, jeu, image)
        lignes += ["", f"![Fiabilité — {TITRES.get(jeu, jeu)}]({image.name})", ""]
    chemin = dossier / "rapport.md"
    chemin.write_text("\n".join(lignes), encoding="utf-8")
    return chemin
```

Dans `src/irl_guard/cli.py`, ajouter :

```python
@app.command()
def evaluer(
    donnees: Path = typer.Option(Path("data/decision"), "--donnees", exists=True, file_okay=False),
    predictions: Path = typer.Option(Path("data/predictions"), "--predictions"),
    approches: str = typer.Option("regles,xgboost,laya,laya-ft,qwen-ft", "--approches"),
    config: Path = typer.Option(Path("configs/exemple.yaml"), "--config", exists=True, dir_okay=False),
    sortie: Path = typer.Option(Path("docs/resultats"), "--sortie"),
) -> None:
    """Compare les approches sur les jeux de test et écrit le rapport."""
    from irl_guard.donnees.lignes import lire_jeu
    from irl_guard.evaluation.metriques import evaluer_predictions
    from irl_guard.evaluation.rapport import JEUX_TEST, ecrire_rapport, obtenir_predictions

    cfg = load_config(config)
    jeux = {j: lire_jeu(donnees / f"{j}.jsonl") for j in JEUX_TEST}
    resultats = {}
    for approche in [a.strip() for a in approches.split(",") if a.strip()]:
        typer.echo(f"Évaluation : {approche}")
        par_jeu = obtenir_predictions(approche, cfg, donnees, predictions, jeux)
        resultats[approche] = {
            j: evaluer_predictions(jeux[j], p, cfg.seuils, cfg.evaluation.couts) if p is not None else None
            for j, p in par_jeu.items()}
    typer.echo(f"Rapport : {ecrire_rapport(resultats, sortie)}")
```

- [ ] **Step 4 : Lancer les tests**

Run : `uv run pytest tests/test_rapport.py -q`
Attendu : 5 tests PASS

Run : `uv run pytest -q`
Attendu : toute la suite PASS

- [ ] **Step 5 : Commit**

```bash
git add src/irl_guard/evaluation/rapport.py src/irl_guard/cli.py tests/test_rapport.py
git commit -m "feat: rapport comparatif des décideurs et commande evaluer" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10 : Génération réelle, publication, évaluation de référence et documentation

**Files:**
- Modify: `README.md` (section « Fine-tuning du decision model »)
- Create: `docs/resultats/rapport.md` et PNG (générés)

**Interfaces:**
- Consumes : tout ce qui précède
- Produces : dataset publié, rapport de référence (règles, XGBoost, et Laya si Studio tourne), instructions Colab

- [ ] **Step 1 : Générer les jeux complets**

Run : `uv run irl-guard donnees generer`
Attendu : quatre lignes `data\decision\<jeu>.jsonl : <n> lignes` (4000, 500, 1000, 1000)

- [ ] **Step 2 : Vérifier la connexion au Hub**

Run : `uv run hf auth whoami`
Attendu : `MissawB`. Sinon, **s'arrêter** et demander à l'utilisateur de lancer `uv run hf auth login` avec un jeton en écriture. Ne jamais manipuler le jeton soi-même.

- [ ] **Step 3 : Publier le dataset**

Run : `uv run irl-guard donnees publier`
Attendu : `Publié en privé : https://huggingface.co/datasets/MissawB/irl-guard-decisions`

- [ ] **Step 4 : Évaluation de référence**

Si Unsloth Studio tourne avec la Decision API (`curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8888/openapi.json` renvoie 200), inclure `laya`. Sinon, l'omettre : le rapport l'indiquera.

Run : `uv run irl-guard evaluer --approches regles,xgboost,laya,laya-ft,qwen-ft`
Attendu : `Rapport : docs\resultats\rapport.md`. `laya-ft` et `qwen-ft` apparaissent « absent » (pas encore entraînés). Laya brut prend environ 20 minutes.

Lire le rapport et vérifier qu'il est plausible avant de continuer :
- XGBoost doit être nettement meilleur sur « chartes vues » que sur « chartes non vues » ;
- les règles doivent avoir à peu près le même score sur les deux jeux, puisqu'elles ne lisent pas la charte.

Si ce n'est pas le cas, consigner le constat sans maquiller les chiffres.

- [ ] **Step 5 : Documenter le fine-tuning dans le README**

Ajouter à la fin de `README.md` :

````markdown
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
````

- [ ] **Step 6 : Lancer toute la suite**

Run : `uv run pytest -q`
Attendu : tout PASS

- [ ] **Step 7 : Commit**

```bash
git add README.md docs/resultats
git commit -m "docs: rapport de référence et procédure de fine-tuning sur Colab" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Après le plan (manuel, côté utilisateur)

1. Lancer le notebook sur Colab avec `MODELE = "qwen"`, puis avec `MODELE = "laya"`. Si la cellule 0 échoue pour Laya, la branche est abandonnée.
2. Récupérer les prédictions (README, étape 3), puis relancer `uv run irl-guard evaluer` pour obtenir le tableau final à cinq approches.
