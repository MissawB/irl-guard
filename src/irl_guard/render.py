"""Rendu d'une image : masquage par zone, écran de pause, overlay de debug."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from irl_guard.analysis import Analyse
from irl_guard.planning import PlanImage
from irl_guard.types import Box

POLICE = cv2.FONT_HERSHEY_SIMPLEX
COULEURS = {"diffuser": (0, 180, 0), "flouter": (0, 140, 255), "pause": (0, 0, 255)}


def masquer(image: np.ndarray, boite: Box, mode: str) -> None:
    h, w = image.shape[:2]
    x1, y1 = max(0, int(boite.x1)), max(0, int(boite.y1))
    x2, y2 = min(w, math.ceil(boite.x2)), min(h, math.ceil(boite.y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = image[y1:y2, x1:x2]
    rw, rh = x2 - x1, y2 - y1
    if mode == "pixel":
        # Au plus une dizaine de blocs sur le grand côté : aucun caractère ne reste lisible.
        bloc = max(8, max(rw, rh) // 10)
        petit = cv2.resize(roi, (max(1, rw // bloc), max(1, rh // bloc)), interpolation=cv2.INTER_AREA)
        roi[:] = cv2.resize(petit, (rw, rh), interpolation=cv2.INTER_NEAREST)
    else:
        k = max(31, (max(rw, rh) // 2) | 1)
        roi[:] = cv2.GaussianBlur(roi, (k, k), 0)


def ecran_pause(largeur: int, hauteur: int, chemin: Path | None = None) -> np.ndarray:
    if chemin is not None:
        image = cv2.imread(str(chemin))
        if image is None:
            raise FileNotFoundError(f"Écran de pause introuvable : {chemin}")
        return cv2.resize(image, (largeur, hauteur))
    image = np.full((hauteur, largeur, 3), 30, np.uint8)
    texte = "Pause de securite"
    echelle = max(0.5, largeur / 800)
    (tw, th), _ = cv2.getTextSize(texte, POLICE, echelle, 2)
    cv2.putText(image, texte, ((largeur - tw) // 2, (hauteur + th) // 2), POLICE, echelle,
                (255, 255, 255), 2, cv2.LINE_AA)
    return image


def dessiner_overlay(image: np.ndarray, analyse: Analyse, en_pause: bool) -> None:
    for p in analyse.pistes:
        x1, y1, x2, y2 = (int(v) for v in p.boite.to_tuple())
        couleur = COULEURS.get(p.action, (200, 200, 200))
        cv2.rectangle(image, (x1, y1), (x2, y2), couleur, 2)
        etiquette = f"#{p.id} {p.classe} {p.action}"
        if p.decision is not None:
            etiquette += f" P(pause)={p.decision.p_pause:.2f}"
        if p.texte:
            etiquette += f" '{p.texte[:30]}'"
        cv2.putText(image, etiquette, (x1, max(12, y1 - 4)), POLICE, 0.45, couleur, 1, cv2.LINE_AA)
    etat = "PAUSE" if en_pause else "DIFFUSION"
    bandeau = f"{etat}  score={analyse.score:.2f}  analyse #{analyse.frame_index}"
    cv2.putText(image, bandeau, (8, 20), POLICE, 0.6, COULEURS["pause" if en_pause else "diffuser"], 2, cv2.LINE_AA)


def rendre(image: np.ndarray, plan: PlanImage, masquage: str, ecran: np.ndarray, debug: bool = False) -> np.ndarray:
    if plan.pause:
        sortie = ecran.copy()
    else:
        sortie = image.copy()
        for boite in plan.masques:
            masquer(sortie, boite, masquage)
    if debug and plan.analyse is not None:
        dessiner_overlay(sortie, plan.analyse, plan.pause)
    return sortie
