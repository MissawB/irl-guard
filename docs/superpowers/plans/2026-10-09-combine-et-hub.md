# Mode combiné et vitrine Hugging Face — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** IRL Guard décide par défaut en combinant règles et modèle fine-tuné (Laya en local, ou Qwen via Studio). Le dataset et les modèles sont publiés avec des fiches complètes, dans une Collection, avec un Space de démo sur ZeroGPU.

**Architecture :**
- Deux nouveaux décideurs respectent l'interface `Decider` : `LayaLocalDecider` (paquet `laya`, sur CPU) et `CombinedDecider` (le plus protecteur l'emporte).
- L'évaluation combine les prédictions existantes, sans nouvel appel de modèle.
- `irl_guard.hub` génère les fiches depuis le rapport, prépare le Space (code Gradio et copie filtrée du paquet) et publie avec `HfApi`, en exigeant `--public` pour changer la visibilité.

**Tech Stack :** Python 3.12, uv, `laya`, `huggingface_hub` (HfApi, Collections), Gradio et `spaces` (dans le Space seulement), `transformers` v5 et `joint_schema_model.py` du dépôt Qwen pour charger Qwen dans le Space.

**Spec :** `docs/superpowers/specs/2026-10-09-combine-et-hub-design.md` (à lire avec ce plan).

## Global Constraints

- **Dépôts publiés** :
  - dataset `MissawB/irl-guard-decisions` ;
  - modèles `MissawB/irl-guard-decision-qwen`, `MissawB/irl-guard-decision-qwen-lora` et `MissawB/irl-guard-decision-laya` ;
  - Space `MissawB/irl-guard-demo`, Collection « IRL Guard ».

  Les dépôts `*-checkpoints` restent **privés** et ne sont jamais modifiés.
- **Visibilité** : aucune visibilité ne change sans le drapeau `--public`. `--dry-run` n'appelle jamais le réseau.
- **Ordre de sévérité** : celui de `ACTIONS = ("diffuser", "flouter", "pause")`. En cas d'égalité, le modèle l'emporte. Un repli du modèle ne donne jamais `diffuser`.
- **Décideur par défaut** : `--decider combine`, et `decideur.modele = "laya-local"` dans la config.
- **Matériel du Space** : `zero-a10g`, Python 3.12. Le Space n'embarque que `irl_guard/__init__.py`, `irl_guard/config.py` et `irl_guard/decision/`.
- **Tokens** : aucun dans le code, les fiches ou le Space.
- **Tests** : pytest, sans réseau ni GPU par défaut. Les tests réseau sont marqués `gpu`.
- **Commits** : chacun se termine par `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Précisions apportées à la spec** :
1. **Chargement de Qwen dans le Space.** Le Space charge Qwen avec `load_release_model` et `systemone` du fichier `joint_schema_model.py` publié dans le dépôt Qwen (vérifié le 2026-10-09 : `transformers` seul, sans Unsloth). C'est ce que la spec appelait « essayer sans Unsloth », et c'est la seule voie.
2. **Collection.** La Collection est créée seulement avec `--public`, puisqu'elle est publique : on évite qu'un artefact public apparaisse sans confirmation.
3. **Logique du Space.** Elle vit dans `space/logique.py`, sans dépendance à Gradio, pour être testée en local. `space/app.py` ne contient que l'interface.

## Review Focus

1. **Le test CLI existant `test_laya_injoignable` n'indique pas `--decider`.** Avec le nouveau défaut `combine`, il chargerait le vrai modèle Laya depuis le Hub. On attend que le test passe `--decider laya` explicitement et reste hors réseau. Modifié en Task 3.
2. **Les règles disent `diffuser` et le modèle est en repli.** On attend `flouter`, jamais `diffuser`. Test : `test_repli_du_modele_jamais_diffuser` (Task 2).
3. **Le fichier de prédictions combiné est incomplet** (un id manque côté modèle). On attend que l'approche combinée soit marquée absente, sans plantage. Test : `test_combine_avec_predictions_modele_incompletes` (Task 4).
4. **Une fiche publiée contient un token ou un chemin local.** On attend aucun `hf_…` et aucun chemin `C:\` ou `/Users/`. Test : `test_cartes_sans_secret_ni_chemin_local` (Task 6).
5. **`hub publier` est lancé sans `--public`.** On attend aucun appel à `update_repo_settings` ni à `create_collection`, et un Space créé en privé. Test : `test_sans_public_rien_ne_devient_public` (Task 8).

---

## Structure des fichiers

```
src/irl_guard/
  config.py                     (modif.) + DecideurConfig, champ `decideur`
  decision/laya_local.py        LayaLocalDecider, _charger_laya
  decision/combine.py           CombinedDecider, plus_protectrice
  evaluation/predicteurs.py     (modif.) + combiner_predictions
  evaluation/rapport.py         (modif.) + approches combine-laya / combine-qwen
  cli.py                        (modif.) choix laya-local et combine, défaut combine ; groupe `hub`
  notebook.py                   (modif.) dossiers de sauvegarde par modèle
  hub/__init__.py
  hub/cartes.py                 lire_tableaux, stats_dataset, carte_dataset, carte_qwen, carte_qwen_lora, carte_laya
  hub/publication.py            DEPOTS_PUBLICS, SPACE, copier_paquet_space, construire, publier
space/
  app.py                        interface Gradio
  logique.py                    construire_state, decider_tout, DeciderQwen (sans Gradio)
  requirements.txt
  README.md                     frontmatter du Space
tests/test_laya_local.py  test_combine.py  test_cli.py (modif.)  test_rapport.py (modif.)
tests/test_notebook.py (modif.)  test_cartes.py  test_space.py  test_hub_publication.py
```

---

### Task 1 : `LayaLocalDecider`

**Files :**
- Modify : `pyproject.toml` (+ `"laya"`)
- Create : `src/irl_guard/decision/laya_local.py`
- Test : `tests/test_laya_local.py`

**Interfaces :**
- Consumes : `QUESTIONS`, `Decision`, `decision_repli` (`decision/base.py`) ; `parser_reponse`, `LayaIndisponible` (`decision/laya.py`) ; `SessionConfig`
- Produces : `LayaLocalDecider(cfg, modele, chargeur=None)` avec `.decide(state) -> Decision` (source `"laya-local"`, repli si erreur) et `.verifier()` (lève `LayaIndisponible`) ; `_charger_laya(modele)`

- [ ] **Step 1 : Ajouter la dépendance**

Ajouter `"laya",` à la fin de `dependencies` dans `pyproject.toml`.

Run : `uv sync`
Attendu : se termine sans erreur.

- [ ] **Step 2 : Écrire les tests**

`tests/test_laya_local.py` :

```python
import pytest

from irl_guard.config import SessionConfig
from irl_guard.decision.laya import LayaIndisponible
from irl_guard.decision.laya_local import LayaLocalDecider

CFG = SessionConfig(charte="x")
STATE = {"categorie": "IRL", "mode_domicile": True, "charte": "x",
         "cible": {"type": "texte", "texte": "Rue Nationale", "conf": 0.9, "taille": "grande"}, "contexte": []}


def _reponse(p_action=(0.2, 0.2, 0.6)):
    return {"answers": {
        "action": {"probabilities": dict(zip(("diffuser", "flouter", "pause"), p_action))},
        "risque": {"probabilities": {"localisation": 0.9, "donnee_perso": 0.025, "arme": 0.025,
                                     "propos": 0.025, "aucun": 0.025}},
        "gravite": {"score": 3, "probabilities": {"0": 0.1, "1": 0, "2": 0, "3": 0.9}},
    }}


class FauxAgent:
    def __init__(self, reponse=None, erreur=None):
        self.reponse, self.erreur, self.appels = reponse or _reponse(), erreur, []

    def predict(self, state, questions):
        self.appels.append((state, questions))
        if self.erreur:
            raise self.erreur
        return self.reponse


def test_decision_et_source():
    agent = FauxAgent()
    d = LayaLocalDecider(CFG, "MissawB/x", chargeur=lambda m: agent).decide(STATE)
    assert (d.action, d.risque, d.gravite, d.source) == ("pause", "localisation", 3, "laya-local")
    assert set(agent.appels[0][1]) == {"action", "risque", "gravite"}


def test_chargement_paresseux_et_unique():
    charges = []

    def chargeur(modele):
        charges.append(modele)
        return FauxAgent()

    decider = LayaLocalDecider(CFG, "MissawB/x", chargeur=chargeur)
    assert charges == []
    decider.decide(STATE)
    decider.decide(STATE)
    assert charges == ["MissawB/x"]


def test_erreur_de_prediction_donne_repli():
    d = LayaLocalDecider(CFG, "m", chargeur=lambda m: FauxAgent(erreur=RuntimeError("boum"))).decide(STATE)
    assert (d.action, d.source) == ("flouter", "fallback")


def test_reponse_invalide_donne_repli():
    d = LayaLocalDecider(CFG, "m", chargeur=lambda m: FauxAgent(reponse={"answers": {}})).decide(STATE)
    assert d.source == "fallback"


def test_verifier_leve_si_modele_introuvable():
    def chargeur(modele):
        raise OSError("dépôt introuvable")

    with pytest.raises(LayaIndisponible, match="MissawB/absent"):
        LayaLocalDecider(CFG, "MissawB/absent", chargeur=chargeur).verifier()


@pytest.mark.gpu
def test_vrai_modele_depuis_le_hub():
    d = LayaLocalDecider(CFG, "MissawB/irl-guard-decision-laya").decide(
        {**STATE, "charte": "Floute les insultes.", "cible": {**STATE["cible"], "texte": "Sale connard"},
         "mode_domicile": False})
    assert d.source == "laya-local" and d.action in ("flouter", "pause")
```

- [ ] **Step 3 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_laya_local.py -q`
Attendu : échec par `ModuleNotFoundError: No module named 'irl_guard.decision.laya_local'`

- [ ] **Step 4 : Implémenter**

`src/irl_guard/decision/laya_local.py` :

```python
"""Laya (ou un Laya fine-tuné) chargé dans le processus avec le paquet `laya` : CPU, sans serveur."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace

from irl_guard.config import SessionConfig
from irl_guard.decision.base import QUESTIONS, Decision, decision_repli
from irl_guard.decision.laya import LayaIndisponible, parser_reponse

log = logging.getLogger(__name__)


def _charger_laya(modele: str):
    import laya
    return laya.load(modele)


class LayaLocalDecider:
    def __init__(self, cfg: SessionConfig, modele: str, chargeur: Callable[[str], object] | None = None):
        self.cfg, self.modele = cfg, modele
        self._chargeur = chargeur or _charger_laya
        self._agent = None

    def _agent_charge(self):
        if self._agent is None:
            self._agent = self._chargeur(self.modele)
        return self._agent

    def _appeler(self, state: dict) -> Decision:
        reponse = self._agent_charge().predict(state, QUESTIONS)
        decision = parser_reponse(reponse, self.cfg.seuils.pause, self.cfg.seuils.flou)
        return replace(decision, source="laya-local")

    def decide(self, state: dict) -> Decision:
        try:
            return self._appeler(state)
        except Exception as e:  # échec du côté sûr, quelle que soit l'erreur du modèle local
            log.warning("Laya local en erreur, piste floutée par défaut : %s", e)
            return decision_repli()

    def verifier(self) -> None:
        try:
            self._appeler({"cible": {"type": "texte", "texte": "test"}})
        except Exception as e:
            raise LayaIndisponible(f"Modèle Laya local « {self.modele} » inutilisable : {e}. "
                                   "Vérifie l'identifiant du modèle et la connexion au Hub, "
                                   "ou relance avec --decider rules.") from e
```

- [ ] **Step 5 : Lancer les tests**

Run : `uv run pytest tests/test_laya_local.py -q`
Attendu : 5 tests PASS, 1 désélectionné

Run : `uv run pytest tests/test_laya_local.py -q -m gpu`
Attendu : 1 test PASS. Il télécharge le modèle depuis le Hub avec la connexion `hf` locale.

- [ ] **Step 6 : Commit**

```bash
git add pyproject.toml uv.lock src/irl_guard/decision/laya_local.py tests/test_laya_local.py
git commit -m "feat: décideur Laya local (paquet laya, CPU, sans serveur)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2 : `CombinedDecider`

**Files :**
- Create : `src/irl_guard/decision/combine.py`
- Test : `tests/test_combine.py`

**Interfaces :**
- Consumes : `ACTIONS`, `Decision`, `Decider` (`decision/base.py`)
- Produces : `plus_protectrice(regle: str, modele: str) -> str` (nom du gagnant, `"regle"` ou `"modele"`) ; `CombinedDecider(regles, modele)` avec `.decide(state) -> Decision` (source `"combine:<source du gagnant>"`)

- [ ] **Step 1 : Écrire les tests**

`tests/test_combine.py` :

```python
import pytest

from irl_guard.decision.base import Decision, decision_repli
from irl_guard.decision.combine import CombinedDecider, plus_protectrice


class Fixe:
    def __init__(self, decision):
        self.decision = decision

    def decide(self, state):
        return self.decision


def _d(action, source):
    return Decision(action, "localisation", 2, {}, source)


@pytest.mark.parametrize("regle, modele, gagnant", [
    ("diffuser", "diffuser", "modele"), ("diffuser", "flouter", "modele"), ("diffuser", "pause", "modele"),
    ("flouter", "diffuser", "regle"), ("flouter", "flouter", "modele"), ("flouter", "pause", "modele"),
    ("pause", "diffuser", "regle"), ("pause", "flouter", "regle"), ("pause", "pause", "modele"),
])
def test_plus_protectrice(regle, modele, gagnant):
    assert plus_protectrice(regle, modele) == gagnant


def test_source_du_gagnant():
    d = CombinedDecider(Fixe(_d("pause", "rules")), Fixe(_d("flouter", "laya-local"))).decide({})
    assert (d.action, d.source) == ("pause", "combine:rules")
    d = CombinedDecider(Fixe(_d("diffuser", "rules")), Fixe(_d("flouter", "laya-local"))).decide({})
    assert (d.action, d.source) == ("flouter", "combine:laya-local")


def test_repli_du_modele_jamais_diffuser():
    d = CombinedDecider(Fixe(_d("diffuser", "rules")), Fixe(decision_repli())).decide({})
    assert d.action == "flouter" and d.source == "combine:fallback"
    d = CombinedDecider(Fixe(_d("pause", "rules")), Fixe(decision_repli())).decide({})
    assert d.action == "pause" and d.source == "combine:rules"
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_combine.py -q`
Attendu : échec par `ModuleNotFoundError`

- [ ] **Step 3 : Implémenter**

`src/irl_guard/decision/combine.py` :

```python
"""Mode combiné : règles et modèle, la décision la plus protectrice l'emporte."""

from __future__ import annotations

from dataclasses import replace

from irl_guard.decision.base import ACTIONS, Decider, Decision


def plus_protectrice(regle: str, modele: str) -> str:
    """Nom du gagnant ; à égalité, le modèle (il porte des probabilités)."""
    return "modele" if ACTIONS.index(modele) >= ACTIONS.index(regle) else "regle"


class CombinedDecider:
    def __init__(self, regles: Decider, modele: Decider):
        self.regles, self.modele = regles, modele

    def decide(self, state: dict) -> Decision:
        regle = self.regles.decide(state)
        modele = self.modele.decide(state)
        if modele.source == "fallback":
            # Le repli vaut « flouter » : il ne peut que relever une règle qui diffuserait.
            gagnant = modele if regle.action == "diffuser" else regle
        else:
            gagnant = modele if plus_protectrice(regle.action, modele.action) == "modele" else regle
        return replace(gagnant, source=f"combine:{gagnant.source}")
```

- [ ] **Step 4 : Lancer les tests**

Run : `uv run pytest tests/test_combine.py -q`
Attendu : 11 tests PASS

- [ ] **Step 5 : Commit**

```bash
git add src/irl_guard/decision/combine.py tests/test_combine.py
git commit -m "feat: décideur combiné règles + modèle (le plus protecteur l'emporte)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3 : Configuration `decideur` et ligne de commande

**Files :**
- Modify : `src/irl_guard/config.py`, `src/irl_guard/cli.py`, `tests/test_cli.py`, `configs/exemple.yaml`
- Test : `tests/test_cli.py`

**Interfaces :**
- Consumes : `LayaLocalDecider` (Task 1), `CombinedDecider` (Task 2), `RulesDecider`, `LayaDecider`
- Produces :
  - `DecideurConfig(modele: Literal["laya-local", "studio"] = "laya-local", laya_local: str = "MissawB/irl-guard-decision-laya")` et `SessionConfig.decideur`
  - `ChoixDecider` avec `rules`, `laya`, `laya_local = "laya-local"`, `combine` ; `--decider` vaut `combine` par défaut
  - `creer_decider(choix, cfg) -> Decider`

- [ ] **Step 1 : Écrire les tests**

Dans `tests/test_cli.py`, remplacer `test_laya_injoignable` par la version suivante (qui indique explicitement `--decider laya`) et ajouter les tests :

```python
def test_laya_injoignable(make_video, tmp_path):
    entree = make_video(n=5)
    config = _config(tmp_path, laya={"url": "http://127.0.0.1:9", "timeout_s": 0.5})
    r = runner.invoke(cli.app, ["process", str(entree), "--config", str(config), "--out", str(tmp_path / "o.mp4"),
                                "--decider", "laya"])
    assert r.exit_code == 2
    assert "--decider rules" in r.output


class FauxAgentLaya:
    def predict(self, state, questions):
        return {"answers": {
            "action": {"probabilities": {"diffuser": 0.9, "flouter": 0.05, "pause": 0.05}},
            "risque": {"probabilities": {"localisation": 0.2, "donnee_perso": 0.2, "arme": 0.2,
                                         "propos": 0.2, "aucun": 0.2}},
            "gravite": {"score": 0, "probabilities": {"0": 1.0, "1": 0, "2": 0, "3": 0}},
        }}


def test_combine_est_le_defaut_et_garde_la_regle(make_video, tmp_path, monkeypatch):
    import json

    from irl_guard.decision import laya_local
    monkeypatch.setattr(laya_local, "_charger_laya", lambda modele: FauxAgentLaya())
    monkeypatch.setattr(cli, "construire_perception", lambda cfg: TexteFixe())
    entree = make_video(n=10, fps=10.0, size=(640, 240), draw=_dessiner)
    r = runner.invoke(cli.app, ["process", str(entree), "--config", str(_config(tmp_path)),
                                "--out", str(tmp_path / "out.mp4")])
    assert r.exit_code == 0, r.output
    lignes = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    appel = json.loads(lignes[0])["analyse"]["appels"][0]
    # Le modèle factice diffuse, la règle floute « Rue Nationale » : la règle l'emporte.
    assert appel["decision"]["action"] == "flouter"
    assert appel["decision"]["source"] == "combine:rules"


def test_laya_local_seul(make_video, tmp_path, monkeypatch):
    from irl_guard.decision import laya_local
    monkeypatch.setattr(laya_local, "_charger_laya", lambda modele: FauxAgentLaya())
    monkeypatch.setattr(cli, "construire_perception", lambda cfg: TexteFixe())
    entree = make_video(n=5, fps=10.0, size=(640, 240), draw=_dessiner)
    r = runner.invoke(cli.app, ["process", str(entree), "--config", str(_config(tmp_path)),
                                "--out", str(tmp_path / "out.mp4"), "--decider", "laya-local"])
    assert r.exit_code == 0, r.output


def test_config_decideur_par_defaut():
    from irl_guard.config import SessionConfig
    cfg = SessionConfig(charte="x")
    assert cfg.decideur.modele == "laya-local"
    assert cfg.decideur.laya_local == "MissawB/irl-guard-decision-laya"
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_cli.py -q`
Attendu : `test_combine_est_le_defaut_et_garde_la_regle`, `test_laya_local_seul` et `test_config_decideur_par_defaut` échouent (pas de `decideur` dans la config, choix `laya-local` refusé). `test_laya_injoignable` passe.

- [ ] **Step 3 : Ajouter `DecideurConfig`**

Dans `src/irl_guard/config.py`, avant `def _couts_par_defaut` :

```python
class DecideurConfig(_Strict):
    modele: Literal["laya-local", "studio"] = "laya-local"
    laya_local: str = "MissawB/irl-guard-decision-laya"
```

et dans `SessionConfig`, après `laya` :

```python
    decideur: DecideurConfig = Field(default_factory=DecideurConfig)
```

- [ ] **Step 4 : Mettre à jour la CLI**

Dans `src/irl_guard/cli.py`, ajouter les imports :

```python
from irl_guard.decision.combine import CombinedDecider
from irl_guard.decision.laya_local import LayaLocalDecider
```

remplacer `ChoixDecider` et `creer_decider` par :

```python
class ChoixDecider(str, Enum):
    rules = "rules"
    laya = "laya"
    laya_local = "laya-local"
    combine = "combine"


def creer_decider(choix: ChoixDecider, cfg: SessionConfig) -> Decider:
    if choix == ChoixDecider.rules:
        return RulesDecider(cfg)
    if choix == ChoixDecider.laya:
        decider = LayaDecider(cfg)
        decider.verifier()
        return decider
    if choix == ChoixDecider.laya_local:
        decider = LayaLocalDecider(cfg, cfg.decideur.laya_local)
        decider.verifier()
        return decider
    modele = ChoixDecider.laya if cfg.decideur.modele == "studio" else ChoixDecider.laya_local
    return CombinedDecider(RulesDecider(cfg), creer_decider(modele, cfg))
```

et, dans les signatures de `process` et `replay`, remplacer `typer.Option(ChoixDecider.laya, "--decider")` par `typer.Option(ChoixDecider.combine, "--decider")`.

Dans `configs/exemple.yaml`, ajouter à la fin :

```yaml

decideur:
  modele: laya-local          # laya-local (CPU, sans serveur) ou studio (Unsloth Studio, voir laya:)
  laya_local: MissawB/irl-guard-decision-laya
```

- [ ] **Step 5 : Lancer les tests**

Run : `uv run pytest tests/test_cli.py tests/test_config.py -q`
Attendu : tout PASS

Run : `uv run pytest -q`
Attendu : toute la suite PASS

- [ ] **Step 6 : Commit**

```bash
git add src/irl_guard/config.py src/irl_guard/cli.py tests/test_cli.py configs/exemple.yaml
git commit -m "feat: mode combiné par défaut dans la CLI, choix du modèle par la config" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4 : Évaluation des approches combinées

**Files :**
- Modify : `src/irl_guard/evaluation/predicteurs.py`, `src/irl_guard/evaluation/rapport.py`, `src/irl_guard/cli.py` (approches par défaut)
- Test : `tests/test_rapport.py`

**Interfaces :**
- Consumes : `plus_protectrice` (Task 2), `action_predite` (`evaluation/metriques.py`), `obtenir_predictions`
- Produces : `combiner_predictions(regles, modele, seuils) -> dict[str, dict]` ; les approches `combine-laya` et `combine-qwen` dans `APPROCHES` et dans la valeur par défaut de `--approches`

- [ ] **Step 1 : Écrire les tests**

Ajouter à `tests/test_rapport.py` :

```python
def _rep(action):
    p = {a: (1.0 if a == action else 0.0) for a in ("diffuser", "flouter", "pause")}
    return {"action": {"probabilities": p}, "risque": {"probabilities": {"aucun": 1.0}}, "gravite": {"score": 0}}


def test_combine_fuite_jamais_pire_que_chaque_source():
    from irl_guard.donnees.lignes import construire_gold
    from irl_guard.evaluation.metriques import evaluer_predictions
    from irl_guard.evaluation.predicteurs import combiner_predictions

    cfg = SessionConfig(charte="x")
    attendus = ["flouter", "pause", "diffuser", "flouter", "pause", "flouter"]
    regles = ["diffuser", "pause", "diffuser", "flouter", "diffuser", "diffuser"]
    modele = ["flouter", "diffuser", "diffuser", "diffuser", "diffuser", "diffuser"]
    lignes = [{"id": str(i), "gold": json.dumps(construire_gold(a, "localisation"))} for i, a in enumerate(attendus)]
    p_regles = {str(i): _rep(a) for i, a in enumerate(regles)}
    p_modele = {str(i): _rep(a) for i, a in enumerate(modele)}
    combine = combiner_predictions(p_regles, p_modele, cfg.seuils)
    fuite = lambda p: evaluer_predictions(lignes, p, cfg.seuils, cfg.evaluation.couts).taux_fuite
    assert fuite(combine) <= min(fuite(p_regles), fuite(p_modele))
    assert combine["0"] is p_modele["0"] and combine["1"] is p_regles["1"]


def test_combine_depuis_fichiers(tmp_path):
    jeux = generer_jeux(0, PETIT)
    dossier = tmp_path / "predictions" / "laya-ft"
    dossier.mkdir(parents=True)
    with (dossier / "predictions_test_vues.jsonl").open("w", encoding="utf-8") as f:
        for l in jeux["test_vues"]:
            f.write(json.dumps({"id": l["id"], "answers": _rep("diffuser")}) + "\n")
    preds = obtenir_predictions("combine-laya", SessionConfig(charte="x"), tmp_path, tmp_path / "predictions",
                                {j: jeux[j] for j in ("test_vues", "test_non_vues")})
    assert len(preds["test_vues"]) == 10 and preds["test_non_vues"] is None


def test_combine_avec_predictions_modele_incompletes(tmp_path):
    from typer.testing import CliRunner

    from irl_guard import cli
    jeux = generer_jeux(0, PETIT)
    ecrire_jeux(jeux, tmp_path / "decision")
    dossier = tmp_path / "predictions" / "qwen-ft"
    dossier.mkdir(parents=True)
    with (dossier / "predictions_test_vues.jsonl").open("w", encoding="utf-8") as f:
        for l in jeux["test_vues"][:3]:  # seulement 3 lignes sur 10
            f.write(json.dumps({"id": l["id"], "answers": _rep("pause")}) + "\n")
    config = tmp_path / "s.yaml"
    config.write_text("charte: x\n", encoding="utf-8")
    r = CliRunner().invoke(cli.app, ["evaluer", "--donnees", str(tmp_path / "decision"),
                                     "--predictions", str(tmp_path / "predictions"),
                                     "--approches", "regles,combine-qwen", "--config", str(config),
                                     "--sortie", str(tmp_path / "resultats")])
    assert r.exit_code == 0, r.output
    assert "| combine-qwen | absent |" in (tmp_path / "resultats" / "rapport.md").read_text(encoding="utf-8")
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_rapport.py -q`
Attendu : les 3 nouveaux tests échouent (`ImportError` sur `combiner_predictions`, approche inconnue)

- [ ] **Step 3 : Implémenter**

Dans `src/irl_guard/evaluation/predicteurs.py`, ajouter :

```python
def combiner_predictions(regles: dict[str, dict], modele: dict[str, dict], seuils) -> dict[str, dict]:
    """Réponse la plus protectrice ligne par ligne (règle de `CombinedDecider`).

    Une ligne absente côté modèle est omise : l'évaluation marquera l'approche absente.
    """
    from irl_guard.decision.combine import plus_protectrice
    from irl_guard.evaluation.metriques import action_predite
    combine = {}
    for id_, reponse_regle in regles.items():
        reponse_modele = modele.get(id_)
        if reponse_modele is None:
            continue
        gagnant = plus_protectrice(action_predite(reponse_regle, seuils), action_predite(reponse_modele, seuils))
        combine[id_] = reponse_modele if gagnant == "modele" else reponse_regle
    return combine
```

Dans `src/irl_guard/evaluation/rapport.py` :
- remplacer la ligne `APPROCHES = ...` par :

```python
APPROCHES = ("toujours-diffuser", "regles", "xgboost", "laya", "laya-ft", "qwen-ft", "combine-laya", "combine-qwen")
SOURCES_COMBINEES = {"combine-laya": "laya-ft", "combine-qwen": "qwen-ft"}
```

- importer `combiner_predictions` avec les autres fonctions de `predicteurs` ;
- au début de `obtenir_predictions`, ajouter :

```python
    if approche in SOURCES_COMBINEES:
        modele = obtenir_predictions(SOURCES_COMBINEES[approche], cfg, dossier_donnees, dossier_predictions, jeux)
        regles = obtenir_predictions("regles", cfg, dossier_donnees, dossier_predictions, jeux)
        return {j: combiner_predictions(regles[j], modele[j], cfg.seuils) if modele[j] is not None else None
                for j in jeux}
```

Dans `src/irl_guard/cli.py`, remplacer la valeur par défaut de `--approches` de `evaluer` par `"toujours-diffuser,regles,xgboost,laya,laya-ft,qwen-ft,combine-laya,combine-qwen"`.

- [ ] **Step 4 : Lancer les tests**

Run : `uv run pytest tests/test_rapport.py -q`
Attendu : 11 tests PASS

- [ ] **Step 5 : Commit**

```bash
git add src/irl_guard/evaluation/predicteurs.py src/irl_guard/evaluation/rapport.py src/irl_guard/cli.py tests/test_rapport.py
git commit -m "feat: évaluation des approches combinées règles + modèle fine-tuné" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5 : Correction du notebook (dossiers de sauvegarde par modèle)

**Files :**
- Modify : `src/irl_guard/notebook.py`, `notebooks/finetune_decision.ipynb` (régénéré)
- Test : `tests/test_notebook.py`

**Interfaces :**
- Consumes : rien de nouveau
- Produces : une cellule de publication qui utilise `lora-{MODELE}` et `merged-{MODELE}`, vidés avant sauvegarde, et qui vérifie `model_type` avant l'envoi

- [ ] **Step 1 : Écrire le test**

Ajouter à `tests/test_notebook.py` :

```python
def test_sauvegardes_isolees_par_modele():
    texte = "\n".join(_code(construire_notebook()))
    assert 'f"lora-{MODELE}"' in texte and 'f"merged-{MODELE}"' in texte
    assert "shutil.rmtree" in texte
    assert '"lora"' not in texte and '"merged"' not in texte
    assert 'config["model_type"]' in texte
```

- [ ] **Step 2 : Lancer le test pour vérifier qu'il échoue**

Run : `uv run pytest tests/test_notebook.py -q`
Attendu : `test_sauvegardes_isolees_par_modele` FAIL

- [ ] **Step 3 : Remplacer la cellule de publication**

Dans `src/irl_guard/notebook.py`, remplacer la cellule de code qui commence par `from huggingface_hub import HfApi` par :

```python
    ("code", """import json
import os
import shutil

from huggingface_hub import HfApi

api = HfApi()
DOSSIER_LORA, DOSSIER_MERGED = f"lora-{MODELE}", f"merged-{MODELE}"
for dossier in (DOSSIER_LORA, DOSSIER_MERGED):
    shutil.rmtree(dossier, ignore_errors=True)  # jamais de fichiers d'un entraînement précédent
model.save_pretrained(DOSSIER_LORA)
model.save_pretrained_merged(DOSSIER_MERGED)

with open(os.path.join(DOSSIER_MERGED, "config.json"), encoding="utf-8") as f:
    config = json.load(f)
attendu = "laya" if MODELE == "laya" else "qwen"
assert config["model_type"].startswith(attendu), f"config.json inattendu pour {MODELE} : {config['model_type']}"

for dossier, repo in ((DOSSIER_LORA, REPO_MODELE + "-lora"), (DOSSIER_MERGED, REPO_MODELE)):
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=dossier, repo_id=repo)
    print("Publié :", repo, sorted(api.list_repo_files(repo)))"""),
```

- [ ] **Step 4 : Régénérer le notebook et lancer les tests**

Run : `uv run python -m irl_guard.notebook && uv run pytest tests/test_notebook.py -q`
Attendu : `notebooks\finetune_decision.ipynb`, puis 6 tests PASS

- [ ] **Step 5 : Commit**

```bash
git add src/irl_guard/notebook.py notebooks/finetune_decision.ipynb tests/test_notebook.py
git commit -m "fix: notebook — dossiers de sauvegarde propres à chaque modèle, vérification avant envoi" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6 : Fiches Hugging Face

**Files :**
- Create : `src/irl_guard/hub/__init__.py`, `src/irl_guard/hub/cartes.py`
- Test : `tests/test_cartes.py`

**Interfaces :**
- Consumes : `lire_jeu`, `JEUX` (`donnees/lignes.py`)
- Produces :
  - `lire_tableaux(rapport_md: str) -> dict[str, list[dict[str, str]]]` (section, puis lignes ; clés = en-têtes)
  - `stats_dataset(dossier) -> dict` (tailles par jeu, répartition des actions)
  - `carte_dataset(stats) -> str`, `carte_qwen(tableaux) -> str`, `carte_qwen_lora() -> str`, `carte_laya(tableaux) -> str`, chacune avec un frontmatter YAML

- [ ] **Step 1 : Écrire les tests**

`tests/test_cartes.py` :

```python
import re

import yaml

from irl_guard.donnees.lignes import ecrire_jeux, generer_jeux
from irl_guard.hub.cartes import (
    carte_dataset, carte_laya, carte_qwen, carte_qwen_lora, lire_tableaux, stats_dataset,
)

RAPPORT = """# Résultats du decision model

## Chartes vues

| Approche | n | Action | F1 macro | Risque | Gravité | Fuite | ECE | Coût moyen |
|---|---|---|---|---|---|---|---|---|
| regles | 1000 | 38.5 % | 0.331 | 80.4 % | 38.5 % | 23.7 % | 0.615 | 3.37 |
| qwen-ft | 1000 | 94.6 % | 0.945 | 99.6 % | 94.5 % | 1.8 % | 0.015 | 0.20 |
| laya-ft | absent | | | | | | | | |

## Chartes non vues

| Approche | n | Action | F1 macro | Risque | Gravité | Fuite | ECE | Coût moyen |
|---|---|---|---|---|---|---|---|---|
| qwen-ft | 1000 | 84.6 % | 0.831 | 99.5 % | 84.8 % | 13.6 % | 0.067 | 1.19 |
"""


def _frontmatter(carte):
    assert carte.startswith("---\n")
    return yaml.safe_load(carte.split("---\n")[1])


def test_lire_tableaux():
    t = lire_tableaux(RAPPORT)
    assert list(t) == ["Chartes vues", "Chartes non vues"]
    assert t["Chartes vues"][1]["Approche"] == "qwen-ft" and t["Chartes vues"][1]["Fuite"] == "1.8 %"
    assert t["Chartes vues"][2]["n"] == "absent"


def test_stats_dataset(tmp_path):
    ecrire_jeux(generer_jeux(0, {"train": 20, "val": 2, "test_vues": 4, "test_non_vues": 4}), tmp_path)
    s = stats_dataset(tmp_path)
    assert s["tailles"] == {"train": 20, "val": 2, "test_vues": 4, "test_non_vues": 4}
    assert abs(sum(s["actions"]["train"].values()) - 1.0) < 1e-9


def test_carte_dataset():
    carte = carte_dataset({"tailles": {"train": 4000, "val": 500, "test_vues": 1000, "test_non_vues": 1000},
                           "actions": {"train": {"diffuser": 0.46, "flouter": 0.25, "pause": 0.29}}})
    meta = _frontmatter(carte)
    assert meta["license"] == "cc-by-4.0" and meta["language"] == ["fr"]
    assert "Avertissement" in carte and "Faker" in carte and "4000" in carte


def test_cartes_modeles():
    tableaux = lire_tableaux(RAPPORT)
    qwen, laya = carte_qwen(tableaux), carte_laya(tableaux)
    assert _frontmatter(qwen)["base_model"] == "unsloth/Qwen3.5-0.8B"
    assert _frontmatter(laya)["base_model"] == "convaiinnovations/laya-multilingual"
    for carte in (qwen, laya):
        meta = _frontmatter(carte)
        assert meta["license"] == "apache-2.0" and meta["datasets"] == ["MissawB/irl-guard-decisions"]
        assert "84.6 %" in carte and "| qwen-ft |" in carte
    assert 'laya.load("MissawB/irl-guard-decision-laya")' in laya
    assert _frontmatter(carte_qwen_lora())["library_name"] == "peft"


def test_cartes_sans_secret_ni_chemin_local():
    tableaux = lire_tableaux(RAPPORT)
    stats = {"tailles": {"train": 1}, "actions": {"train": {"diffuser": 1.0}}}
    for carte in (carte_dataset(stats), carte_qwen(tableaux), carte_qwen_lora(), carte_laya(tableaux)):
        assert not re.search(r"hf_[A-Za-z0-9]{20,}", carte)
        assert "C:\\" not in carte and "/Users/" not in carte
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_cartes.py -q`
Attendu : échec par `ModuleNotFoundError: No module named 'irl_guard.hub'`

- [ ] **Step 3 : Implémenter**

`src/irl_guard/hub/__init__.py` :

```python
"""Vitrine Hugging Face : fiches, Space de démo et publication."""
```

`src/irl_guard/hub/cartes.py` :

````python
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
        "combine-laya": "Règles + Laya fine-tuné", "combine-qwen": "Règles + Qwen fine-tuné"}
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
    return "
".join(blocs)


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
````

- [ ] **Step 4 : Lancer les tests**

Run : `uv run pytest tests/test_cartes.py -q`
Attendu : 5 tests PASS

- [ ] **Step 5 : Commit**

```bash
git add src/irl_guard/hub tests/test_cartes.py
git commit -m "feat: fiches Hugging Face du dataset et des modèles, générées depuis le rapport" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7 : Space de démo

**Files :**
- Create : `space/logique.py`, `space/app.py`, `space/requirements.txt`, `space/README.md`
- Test : `tests/test_space.py`

**Interfaces :**
- Consumes : `SessionConfig`, `QUESTIONS`, `RulesDecider`, `LayaLocalDecider`, `CombinedDecider`, `parser_reponse`
- Produces :
  - `construire_state(charte, categorie, mode_domicile, type_cible, texte, classe, taille) -> dict`
  - `ligne(nom, decision) -> list[str]`
  - `decider_tout(state, regles, laya, qwen) -> list[list[str]]` (`qwen` peut être `None`)
  - `DeciderQwen(repo, device)` et `charger_qwen(device) -> DeciderQwen | None`
  - `EXEMPLES`

- [ ] **Step 1 : Écrire les tests**

`tests/test_space.py` :

```python
import ast
import sys
from pathlib import Path

from irl_guard.config import SessionConfig
from irl_guard.decision.base import Decision
from irl_guard.decision.rules import RulesDecider

sys.path.insert(0, str(Path("space").resolve()))
import logique  # noqa: E402


class Fixe:
    def __init__(self, action):
        self.action = action

    def decide(self, state):
        return Decision(self.action, "localisation", 2,
                        {"action": {"diffuser": 0.1, "flouter": 0.2, "pause": 0.7}}, "laya-local")


def test_construire_state():
    s = logique.construire_state("Floute les insultes.", "IRL", True, "texte", "Sale connard", "couteau", "grande")
    assert s == {"categorie": "IRL", "mode_domicile": True, "charte": "Floute les insultes.",
                 "cible": {"type": "texte", "texte": "Sale connard", "conf": 0.9, "taille": "grande"},
                 "contexte": []}
    o = logique.construire_state("x", "Cuisine", False, "objet", "", "couteau", "moyenne")
    assert o["cible"] == {"type": "objet", "classe": "couteau", "conf": 0.9, "taille": "moyenne"}


def test_decider_tout_sans_qwen():
    regles = RulesDecider(SessionConfig(charte="x"))
    state = logique.construire_state("x", "IRL", False, "texte", "Rue Nationale", "", "grande")
    lignes = logique.decider_tout(state, regles, Fixe("pause"), None)
    noms = [l[0] for l in lignes]
    assert noms == ["Règles", "Laya fine-tuné", "Règles + Laya", "Qwen fine-tuné"]
    assert lignes[2][1] == "pause"
    assert lignes[3][1] == "indisponible en ligne"


def test_decider_tout_avec_qwen():
    regles = RulesDecider(SessionConfig(charte="x"))
    state = logique.construire_state("x", "IRL", False, "texte", "SOLDES -50 %", "", "petite")
    lignes = logique.decider_tout(state, regles, Fixe("diffuser"), Fixe("flouter"))
    assert [l[0] for l in lignes][-2:] == ["Qwen fine-tuné", "Règles + Qwen"]
    assert lignes[-1][1] == "flouter"


def test_exemples_complets():
    assert len(logique.EXEMPLES) >= 4
    assert all(len(e) == 7 for e in logique.EXEMPLES)


def test_app_syntaxe_et_readme():
    ast.parse(Path("space/app.py").read_text(encoding="utf-8"))
    readme = Path("space/README.md").read_text(encoding="utf-8")
    assert "sdk: gradio" in readme and "app_file: app.py" in readme
    exigences = Path("space/requirements.txt").read_text(encoding="utf-8")
    for paquet in ("gradio", "spaces", "laya", "transformers", "pydantic"):
        assert paquet in exigences
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_space.py -q`
Attendu : échec par `ModuleNotFoundError: No module named 'logique'`

- [ ] **Step 3 : Implémenter la logique**

`space/logique.py` :

```python
"""Logique du Space IRL Guard, sans Gradio (testable en local)."""

from __future__ import annotations

import logging
import sys

from irl_guard.config import SessionConfig
from irl_guard.decision.base import QUESTIONS, Decision
from irl_guard.decision.combine import CombinedDecider
from irl_guard.decision.laya import parser_reponse

log = logging.getLogger(__name__)
REPO_QWEN = "MissawB/irl-guard-decision-qwen"
REPO_LAYA = "MissawB/irl-guard-decision-laya"
CHARTE_DEFAUT = ("Live tout public. Floute les insultes. En mode domicile, coupe le live si on voit les adresses. "
                 "Sauf en live cuisine, masque les objets tranchants.")
EXEMPLES = [
    [CHARTE_DEFAUT, "IRL", True, "texte", "Rue Nationale", "couteau", "grande"],
    [CHARTE_DEFAUT, "IRL", False, "texte", "Rue Nationale", "couteau", "grande"],
    [CHARTE_DEFAUT, "Cuisine", False, "objet", "", "couteau", "moyenne"],
    [CHARTE_DEFAUT, "IRL", False, "objet", "", "couteau", "moyenne"],
    ["Ambiance détendue. Je n'ai aucun problème avec les gros mots.", "Gaming", False, "texte", "Sale connard",
     "couteau", "moyenne"],
]


def construire_state(charte: str, categorie: str, mode_domicile: bool, type_cible: str, texte: str,
                     classe: str, taille: str) -> dict:
    cible: dict = {"type": type_cible}
    if type_cible == "objet":
        cible["classe"] = classe
    elif texte:
        cible["texte"] = texte
    cible["conf"] = 0.9
    cible["taille"] = taille
    return {"categorie": categorie, "mode_domicile": bool(mode_domicile), "charte": charte, "cible": cible,
            "contexte": []}


def ligne(nom: str, decision: Decision) -> list[str]:
    p = decision.probabilites.get("action") or {}
    probas = " / ".join(f"{p.get(a, 0):.2f}" for a in ("diffuser", "flouter", "pause")) if p else "règle"
    return [nom, decision.action, probas, decision.risque, str(decision.gravite)]


def decider_tout(state: dict, regles, laya, qwen) -> list[list[str]]:
    lignes = [ligne("Règles", regles.decide(state)),
              ligne("Laya fine-tuné", laya.decide(state)),
              ligne("Règles + Laya", CombinedDecider(regles, laya).decide(state))]
    if qwen is None:
        lignes.append(["Qwen fine-tuné", "indisponible en ligne", "", "", ""])
    else:
        lignes += [ligne("Qwen fine-tuné", qwen.decide(state)),
                   ligne("Règles + Qwen", CombinedDecider(regles, qwen).decide(state))]
    return lignes


class DeciderQwen:
    """Qwen fine-tuné chargé avec `joint_schema_model.py` du dépôt (transformers seul, sans Unsloth)."""

    def __init__(self, cfg: SessionConfig, device: str = "cuda", repo: str = REPO_QWEN):
        from huggingface_hub import snapshot_download
        chemin = snapshot_download(repo)
        sys.path.insert(0, chemin)
        from joint_schema_model import load_release_model, systemone
        self._systemone = systemone
        self.model, self.processor = load_release_model(chemin, device=device)
        self.cfg = cfg

    def decide(self, state: dict) -> Decision:
        reponse = self._systemone(self.model, self.processor,
                                  {"model": "irl-guard-qwen", "state": state, "questions": QUESTIONS})
        decision = parser_reponse(reponse, self.cfg.seuils.pause, self.cfg.seuils.flou)
        return Decision(decision.action, decision.risque, decision.gravite, decision.probabilites, "qwen")


def charger_qwen(cfg: SessionConfig, device: str = "cuda") -> DeciderQwen | None:
    try:
        return DeciderQwen(cfg, device)
    except Exception as e:  # le Space reste utile sans Qwen
        log.warning("Qwen indisponible dans le Space : %s", e)
        return None
```

- [ ] **Step 4 : Écrire l'interface et les fichiers du Space**

`space/app.py` :

```python
"""Démo IRL Guard : comment chaque approche applique une charte à un élément vu à l'écran."""

import gradio as gr

try:
    import spaces
    gpu = spaces.GPU
except ImportError:  # exécution locale sans ZeroGPU
    def gpu(fonction):
        return fonction

from irl_guard.config import SessionConfig
from irl_guard.decision.laya_local import LayaLocalDecider
from irl_guard.decision.rules import RulesDecider

import logique

CFG = SessionConfig(charte="x")
REGLES = RulesDecider(CFG)
LAYA = LayaLocalDecider(CFG, logique.REPO_LAYA)
QWEN = logique.charger_qwen(CFG, device="cuda")
SCORES = ("Sur les chartes **jamais vues** du jeu de test : Qwen fine-tuné 84,6 % de précision, Laya fine-tuné "
          "79,9 %, XGBoost 45,5 %, Laya brut 35,9 % — voir les fiches des modèles.")


@gpu
def decider(charte, categorie, mode_domicile, type_cible, texte, classe, taille):
    state = logique.construire_state(charte, categorie, mode_domicile, type_cible, texte, classe, taille)
    return logique.decider_tout(state, REGLES, LAYA, QWEN)


with gr.Blocks(title="IRL Guard — décisions guidées par une charte") as demo:
    gr.Markdown("# IRL Guard\nÉcris la charte de ta chaîne, décris ce que la caméra voit, et compare les décisions.")
    with gr.Row():
        with gr.Column():
            charte = gr.Textbox(label="Charte de la chaîne", lines=4, value=logique.CHARTE_DEFAUT)
            categorie = gr.Dropdown(["IRL", "Cuisine", "Gaming", "Musique", "Sport"], value="IRL", label="Catégorie")
            mode_domicile = gr.Checkbox(label="Mode domicile")
            type_cible = gr.Radio(["texte", "objet"], value="texte", label="Ce que la caméra voit")
            texte = gr.Textbox(label="Texte lu (si texte)", value="Rue Nationale")
            classe = gr.Dropdown(["couteau", "ciseaux", "telephone", "ordinateur"], value="couteau",
                                 label="Objet (si objet)")
            taille = gr.Radio(["petite", "moyenne", "grande"], value="grande", label="Taille à l'écran")
            bouton = gr.Button("Décider", variant="primary")
        with gr.Column():
            resultats = gr.Dataframe(headers=["Approche", "Action", "P(diffuser / flouter / pause)", "Risque",
                                              "Gravité"], interactive=False)
            gr.Markdown(SCORES)
    entrees = [charte, categorie, mode_domicile, type_cible, texte, classe, taille]
    gr.Examples(logique.EXEMPLES, entrees)
    bouton.click(decider, entrees, resultats)

demo.launch()
```

`space/requirements.txt` :

```text
gradio
spaces
laya
transformers>=5
torch
safetensors
huggingface_hub
pydantic>=2.7
pyyaml>=6.0
httpx>=0.27
```

`space/README.md` (remplacer `<VERSION>` par la version affichée par `uv run python -m pip index versions gradio`, première ligne) :

```markdown
---
title: IRL Guard
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: <VERSION>
python_version: "3.12"
app_file: app.py
license: apache-2.0
models:
  - MissawB/irl-guard-decision-qwen
  - MissawB/irl-guard-decision-laya
datasets:
  - MissawB/irl-guard-decisions
short_description: Décisions de modération guidées par une charte de live IRL
---

Démo d'IRL Guard : un *decision model* fine-tuné applique la charte d'une chaîne de live IRL à ce que la caméra
voit (texte lu ou objet détecté) et décide de diffuser, flouter ou mettre en pause. Comparaison des règles, de Laya
et de Qwen fine-tunés, seuls ou combinés avec les règles.
```

- [ ] **Step 5 : Lancer les tests**

Run : `uv run pytest tests/test_space.py -q`
Attendu : 5 tests PASS

- [ ] **Step 6 : Commit**

```bash
git add space tests/test_space.py
git commit -m "feat: Space de démo (Gradio, ZeroGPU) comparant règles, Laya et Qwen fine-tunés" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8 : Publication sur le Hub

**Files :**
- Create : `src/irl_guard/hub/publication.py`
- Modify : `src/irl_guard/cli.py` (groupe `hub`)
- Test : `tests/test_hub_publication.py`

**Interfaces :**
- Consumes : les fonctions de `hub/cartes.py` (Task 6) ; `space/` (Task 7)
- Produces :
  - `DEPOTS_PUBLICS: list[tuple[str, str]]` (repo_id, repo_type), `SPACE = "MissawB/irl-guard-demo"`, `COLLECTION = "IRL Guard"`
  - `copier_paquet_space(source_irl_guard, destination) -> None`
  - `construire(dossier_build, rapport, donnees, source_space, source_paquet) -> dict[str, Path]`
  - `publier(api, construits, public) -> list[str]` (journal des actions)
  - CLI `irl-guard hub publier [--dry-run] [--public]`

- [ ] **Step 1 : Écrire les tests**

`tests/test_hub_publication.py` :

```python
import subprocess
import sys
from pathlib import Path

from irl_guard.donnees.lignes import ecrire_jeux, generer_jeux
from irl_guard.hub import publication

RAPPORT = Path("docs/resultats/rapport.md")


class FauxApi:
    def __init__(self):
        self.appels = []

    def __getattr__(self, nom):
        def appel(*args, **kwargs):
            self.appels.append((nom, args, kwargs))

            class Collection:
                slug = "MissawB/irl-guard-123"
            return Collection()
        return appel

    def noms(self):
        return [a[0] for a in self.appels]


def _construire(tmp_path):
    ecrire_jeux(generer_jeux(0, {"train": 10, "val": 2, "test_vues": 2, "test_non_vues": 2}), tmp_path / "d")
    return publication.construire(tmp_path / "build", RAPPORT, tmp_path / "d", Path("space"), Path("src/irl_guard"))


def test_copie_filtree_du_paquet(tmp_path):
    publication.copier_paquet_space(Path("src/irl_guard"), tmp_path / "irl_guard")
    fichiers = {p.relative_to(tmp_path / "irl_guard").as_posix() for p in (tmp_path / "irl_guard").rglob("*.py")}
    assert "config.py" in fichiers and "decision/combine.py" in fichiers
    assert not any(f.startswith(("perception", "evaluation", "donnees", "hub")) for f in fichiers)
    assert "cli.py" not in fichiers and "pipeline.py" not in fichiers
    r = subprocess.run([sys.executable, "-c", "import irl_guard.decision.combine, irl_guard.decision.rules"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_construire(tmp_path):
    construits = _construire(tmp_path)
    assert set(construits) == {r for r, _ in publication.DEPOTS_PUBLICS} | {publication.SPACE}
    for repo, chemin in construits.items():
        if repo != publication.SPACE:
            assert chemin.name == "README.md" and chemin.read_text(encoding="utf-8").startswith("---")
    space = construits[publication.SPACE]
    assert (space / "app.py").exists() and (space / "irl_guard" / "decision" / "combine.py").exists()


def test_sans_public_rien_ne_devient_public(tmp_path):
    api = FauxApi()
    publication.publier(api, _construire(tmp_path), public=False)
    assert "update_repo_settings" not in api.noms() and "create_collection" not in api.noms()
    creation_space = [a for a in api.appels if a[0] == "create_repo"][0]
    assert creation_space[2]["private"] is True
    assert ("request_space_hardware", (publication.SPACE, "zero-a10g"), {}) in api.appels


def test_public_ne_touche_que_les_depots_listes(tmp_path):
    api = FauxApi()
    publication.publier(api, _construire(tmp_path), public=True)
    rendus_publics = [a[1][0] for a in api.appels if a[0] == "update_repo_settings"]
    assert set(rendus_publics) == {r for r, _ in publication.DEPOTS_PUBLICS} | {publication.SPACE}
    assert not any("checkpoints" in r for r in rendus_publics)
    elements = [a[2]["item_id"] for a in api.appels if a[0] == "add_collection_item"]
    assert set(elements) == {r for r, _ in publication.DEPOTS_PUBLICS} | {publication.SPACE}


def test_cli_dry_run_sans_reseau(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from irl_guard import cli

    def interdit(*a, **k):
        raise AssertionError("aucun appel réseau en --dry-run")

    monkeypatch.setattr(publication, "publier", interdit)
    ecrire_jeux(generer_jeux(0, {"train": 10, "val": 2, "test_vues": 2, "test_non_vues": 2}), tmp_path / "d")
    r = CliRunner().invoke(cli.app, ["hub", "publier", "--dry-run", "--donnees", str(tmp_path / "d"),
                                     "--build", str(tmp_path / "build")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "build" / "irl-guard-decision-laya" / "README.md").exists()
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_hub_publication.py -q`
Attendu : échec par `ImportError` (`publication` est absent de `irl_guard.hub`)

- [ ] **Step 3 : Implémenter**

`src/irl_guard/hub/publication.py` :

```python
"""Construction locale et publication de la vitrine Hugging Face (fiches, Space, visibilité, Collection)."""

from __future__ import annotations

import shutil
from pathlib import Path

from irl_guard.hub.cartes import (
    carte_dataset, carte_laya, carte_qwen, carte_qwen_lora, lire_tableaux, stats_dataset,
)

DEPOTS_PUBLICS = [
    ("MissawB/irl-guard-decisions", "dataset"),
    ("MissawB/irl-guard-decision-qwen", "model"),
    ("MissawB/irl-guard-decision-qwen-lora", "model"),
    ("MissawB/irl-guard-decision-laya", "model"),
]
SPACE = "MissawB/irl-guard-demo"
COLLECTION = "IRL Guard"
DESCRIPTION_COLLECTION = ("Bloqueur de contenu pour lives IRL : dataset de décisions guidées par une charte, "
                          "decision models fine-tunés et démo.")
# Le Space n'a besoin que de la décision et de la config : pas de perception vidéo ni d'évaluation.
INCLUS_SPACE = ("__init__.py", "config.py", "decision")


def copier_paquet_space(source: Path, destination: Path) -> None:
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    for nom in INCLUS_SPACE:
        chemin = source / nom
        if chemin.is_dir():
            shutil.copytree(chemin, destination / nom, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(chemin, destination / nom)


def construire(dossier_build: Path, rapport: Path, donnees: Path, source_space: Path,
               source_paquet: Path) -> dict[str, Path]:
    tableaux = lire_tableaux(Path(rapport).read_text(encoding="utf-8"))
    cartes = {
        "MissawB/irl-guard-decisions": carte_dataset(stats_dataset(donnees)),
        "MissawB/irl-guard-decision-qwen": carte_qwen(tableaux),
        "MissawB/irl-guard-decision-qwen-lora": carte_qwen_lora(),
        "MissawB/irl-guard-decision-laya": carte_laya(tableaux),
    }
    construits: dict[str, Path] = {}
    for repo, carte in cartes.items():
        chemin = Path(dossier_build) / repo.split("/")[1] / "README.md"
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text(carte, encoding="utf-8")
        construits[repo] = chemin
    space = Path(dossier_build) / "space"
    shutil.rmtree(space, ignore_errors=True)
    shutil.copytree(source_space, space, ignore=shutil.ignore_patterns("__pycache__"))
    copier_paquet_space(Path(source_paquet), space / "irl_guard")
    construits[SPACE] = space
    return construits


def publier(api, construits: dict[str, Path], public: bool) -> list[str]:
    journal = []
    for repo, repo_type in DEPOTS_PUBLICS:
        api.upload_file(path_or_fileobj=str(construits[repo]), path_in_repo="README.md", repo_id=repo,
                        repo_type=repo_type, commit_message="Fiche IRL Guard")
        journal.append(f"fiche envoyée : {repo}")
    api.create_repo(SPACE, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(construits[SPACE]), repo_id=SPACE, repo_type="space",
                      commit_message="Space IRL Guard")
    api.request_space_hardware(SPACE, "zero-a10g")
    journal.append(f"Space déployé (ZeroGPU) : {SPACE}")
    if not public:
        journal.append("visibilité inchangée (relancer avec --public pour publier)")
        return journal
    elements = DEPOTS_PUBLICS + [(SPACE, "space")]
    for repo, repo_type in elements:
        api.update_repo_settings(repo, private=False, repo_type=repo_type)
        journal.append(f"public : {repo}")
    collection = api.create_collection(title=COLLECTION, description=DESCRIPTION_COLLECTION, private=False,
                                       exists_ok=True)
    for repo, repo_type in elements:
        api.add_collection_item(collection.slug, item_id=repo, item_type=repo_type, exists_ok=True)
    journal.append(f"Collection : https://huggingface.co/collections/{collection.slug}")
    return journal
```

Dans `src/irl_guard/cli.py`, ajouter après le groupe `donnees` :

```python
hub_app = typer.Typer(help="Vitrine Hugging Face : fiches, Space et Collection.")
app.add_typer(hub_app, name="hub")


@hub_app.command("publier")
def hub_publier(
    dry_run: bool = typer.Option(False, "--dry-run", help="Génère les fiches et le Space en local, sans réseau"),
    public: bool = typer.Option(False, "--public", help="Rend publics le dataset, les modèles et le Space"),
    rapport: Path = typer.Option(Path("docs/resultats/rapport.md"), "--rapport", exists=True, dir_okay=False),
    donnees: Path = typer.Option(Path("data/decision"), "--donnees", exists=True, file_okay=False),
    build: Path = typer.Option(Path("build/hub"), "--build"),
) -> None:
    """Prépare puis publie la vitrine IRL Guard sur Hugging Face."""
    from irl_guard.hub import publication
    construits = publication.construire(build, rapport, donnees, Path("space"), Path("src/irl_guard"))
    for repo, chemin in construits.items():
        typer.echo(f"{repo} → {chemin}")
    if dry_run:
        typer.echo("--dry-run : rien n'a été envoyé.")
        return
    from huggingface_hub import HfApi
    for ligne in publication.publier(HfApi(), construits, public):
        typer.echo(ligne)
```

Ajouter `build/` à `.gitignore`.

- [ ] **Step 4 : Lancer les tests**

Run : `uv run pytest tests/test_hub_publication.py -q`
Attendu : 5 tests PASS

Run : `uv run pytest -q`
Attendu : toute la suite PASS

- [ ] **Step 5 : Commit**

```bash
git add src/irl_guard/hub/publication.py src/irl_guard/cli.py .gitignore tests/test_hub_publication.py
git commit -m "feat: publication de la vitrine Hugging Face (fiches, Space ZeroGPU, Collection, --public)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9 : Rapport combiné, relecture des fiches et publication

**Files :**
- Modify : `docs/resultats/` (régénéré), `README.md`

**Interfaces :**
- Consumes : tout ce qui précède

- [ ] **Step 1 : Rapport avec les approches combinées**

Run : `uv run irl-guard evaluer`
Attendu : `Rapport : docs\resultats\rapport.md`, avec les lignes `combine-laya` et `combine-qwen`. Environ 20 minutes si Unsloth Studio tourne (Laya brut). Sinon, Laya brut apparaît « absent » : le signaler.

Vérifier le critère n° 2 de la spec : pour chaque jeu, la fuite de `combine-laya` est inférieure ou égale à la plus petite des fuites de `regles` et `laya-ft`, et de même pour `combine-qwen` avec `qwen-ft`. Si ce n'est pas le cas, s'arrêter et diagnostiquer.

- [ ] **Step 2 : Construire en local et faire relire**

Run : `uv run irl-guard hub publier --dry-run`
Attendu : la liste des 5 éléments construits, puis `--dry-run : rien n'a été envoyé.`

**S'arrêter et demander à l'utilisateur** de relire `build/hub/*/README.md` et `build/hub/space/`, puis de confirmer explicitement le passage en public. Ne rien publier sans cette réponse.

- [ ] **Step 3 : Publier (après confirmation)**

Run : `uv run hf auth whoami`
Attendu : `MissawB`. Sinon, s'arrêter.

Run : `uv run irl-guard hub publier --public`
Attendu : les fiches envoyées, `Space déployé (ZeroGPU)`, 5 lignes `public : …`, puis l'URL de la Collection.

- [ ] **Step 4 : Vérifier le Space**

Attendre la fin de la construction du Space : `uv run hf spaces info MissawB/irl-guard-demo`, ou sa page web. Si la construction échoue, lire les journaux (`hf spaces logs MissawB/irl-guard-demo`), corriger `space/`, republier sans `--public` (la visibilité est déjà acquise) et consigner la correction. Si seul Qwen est indisponible, le noter dans le compte rendu : le reste de la démo doit fonctionner.

- [ ] **Step 5 : README**

Ajouter à la fin de `README.md` :

```markdown
## Mode combiné et démo en ligne

Par défaut, `irl-guard process` décide en **mode combiné** : les règles et Laya fine-tuné (chargé en local, sur
CPU) donnent chacun leur décision, et la plus protectrice l'emporte. Pour utiliser Qwen fine-tuné via Unsloth
Studio, régler `decideur.modele: studio` dans la config.

- Démo : https://huggingface.co/spaces/MissawB/irl-guard-demo
- Dataset et modèles : Collection « IRL Guard » sur https://huggingface.co/MissawB
- Republier les fiches après une nouvelle évaluation : `uv run irl-guard hub publier`
```

- [ ] **Step 6 : Commit**

```bash
git add docs/resultats README.md
git commit -m "docs: rapport avec approches combinées, liens vers la démo et la Collection" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
