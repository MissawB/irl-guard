"""Construction du `state` JSON envoyé au décideur pour une piste."""

from __future__ import annotations

import json
from collections import Counter

from irl_guard.config import SessionConfig, TailleConfig
from irl_guard.tracking import Piste
from irl_guard.types import Box

CONTEXTE_TEXTE_MAX = 40


def taille_relative(boite: Box, largeur: int, hauteur: int, cfg: TailleConfig) -> str:
    part = boite.area / float(largeur * hauteur)
    if part <= cfg.petite_max:
        return "petite"
    if part <= cfg.moyenne_max:
        return "moyenne"
    return "grande"


def resume_contexte(cible: Piste, pistes: list[Piste]) -> list[str]:
    autres = [p for p in pistes if p.id != cible.id and not p.perdu]
    comptes = Counter(p.classe for p in autres if p.classe != "texte")
    resume = [f"{classe} x{n}" for classe, n in sorted(comptes.items())]
    resume += [f"texte: {p.texte[:CONTEXTE_TEXTE_MAX]}" for p in autres if p.classe == "texte" and p.texte]
    return resume


def estimer_tokens(obj) -> int:
    """Estimation prudente, faute du tokenizer exact de Laya : environ 3 caractères par token."""
    return len(json.dumps(obj, ensure_ascii=False)) // 3 + 1


def construire_state(cible: Piste, pistes: list[Piste], cfg: SessionConfig, largeur: int, hauteur: int) -> dict:
    description: dict = {"type": "texte" if cible.classe == "texte" else "objet"}
    if cible.classe != "texte":
        description["classe"] = cible.classe
    if cible.texte:
        description["texte"] = cible.texte
    description["conf"] = round(cible.conf, 2)
    description["taille"] = taille_relative(cible.boite, largeur, hauteur, cfg.taille)

    state = {
        "categorie": cfg.categorie,
        "mode_domicile": cfg.mode_domicile,
        "charte": cfg.charte,
        "cible": description,
        "contexte": resume_contexte(cible, pistes),
    }
    # Seul le contexte est tronqué : la cible et la charte ne le sont jamais.
    while estimer_tokens(state) > cfg.budget_tokens_state and state["contexte"]:
        state["contexte"].pop()
    return state
