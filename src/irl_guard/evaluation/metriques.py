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
