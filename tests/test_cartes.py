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
