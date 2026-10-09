"""Précision, taux de fuite, ECE et coût moyen sur un jeu de test."""

from __future__ import annotations

import json
from dataclasses import dataclass

from irl_guard.config import SeuilsConfig
from irl_guard.decision.base import ACTIONS, NIVEAUX_GRAVITE, action_depuis_probas


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
    f1_macro_action: float = 0.0


def _argmax(probas: dict[str, float]) -> str:
    return max(probas, key=probas.get)


def action_predite(reponse: dict, seuils: SeuilsConfig) -> str:
    return action_depuis_probas(reponse["action"]["probabilities"], seuils.pause, seuils.flou)


def _index_niveau(cle: str) -> int:
    return NIVEAUX_GRAVITE.index(cle) if cle in NIVEAUX_GRAVITE else int(cle)


def niveau_gravite(reponse: dict) -> int:
    """Niveau le plus probable ; l'espérance (`score`) n'est qu'un repli : arrondie, elle tombe sur des
    niveaux intermédiaires jamais attendus (0,6 sur 0 et 0,4 sur 3 donnent 1,2)."""
    probas = reponse["gravite"].get("probabilities")
    if probas:
        return _index_niveau(_argmax(probas))
    return min(max(round(reponse["gravite"]["score"]), 0), 3)


def f1_macro(attendues: list[str], predites: list[str]) -> float:
    classes = [c for c in ACTIONS if c in attendues or c in predites]
    scores = []
    for c in classes:
        vp = sum(a == c and p == c for a, p in zip(attendues, predites))
        fp = sum(a != c and p == c for a, p in zip(attendues, predites))
        fn = sum(a == c and p != c for a, p in zip(attendues, predites))
        scores.append(2 * vp / (2 * vp + fp + fn) if vp else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


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
    attendues, predites = [], []
    for ligne in lignes:
        gold = json.loads(ligne["gold"])
        reponse = predictions[ligne["id"]]
        attendue = gold["action"]["label"]
        predite = action_predite(reponse, seuils)
        ok_action += predite == attendue
        attendues.append(attendue)
        predites.append(predite)
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
                    fuites / a_proteger if a_proteger else 0.0, ece, cout / n, points,
                    f1_macro(attendues, predites))
