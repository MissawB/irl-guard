"""Assemblage des lignes (state, questions, gold) et des quatre jeux de données (spec §2.3 à §2.5)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from faker import Faker

from irl_guard.decision.base import ACTIONS, QUESTIONS, RISQUES
from irl_guard.donnees.chartes import rendre_charte, tirer_clauses
from irl_guard.donnees.cibles import NEUTRES, bruiter, tirer_cible
from irl_guard.donnees.clauses import RISQUES_CHARTE, appliquer_charte
from irl_guard.donnees.gabarits import Composants, partitionner

JEUX = ("train", "val", "test_vues", "test_non_vues")
TAILLES_DEFAUT = {"train": 4000, "val": 500, "test_vues": 1000, "test_non_vues": 1000}
CATEGORIES_STREAM = ("IRL", "Cuisine", "Gaming", "Musique", "Sport")
TAILLES = ("petite", "moyenne", "grande")
LISSAGE = 0.9
GRAVITE_PAR_ACTION = {"diffuser": 0, "flouter": 2, "pause": 3}
NIVEAUX = ("0", "1", "2", "3")
# Équilibrage (revue finale C1) : sans lui, ~70 % des labels valaient « diffuser ».
P_AUCUN = 0.12        # cible sans risque
P_MENTIONNE = 0.75    # sinon, cible d'un risque cité par la charte
P_REJET_DIFFUSER = 0.35


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
    while True:
        clauses = tirer_clauses(rng, reservee)
        tirage = rng.random()
        if tirage < P_AUCUN:
            risque = "aucun"
        elif tirage < P_AUCUN + (1 - P_AUCUN) * P_MENTIONNE:
            risque = rng.choice([c.risque for c in clauses])
        else:
            risque = rng.choice(RISQUES_CHARTE)
        categorie = rng.choice(CATEGORIES_STREAM)
        mode_domicile = rng.random() < 1 / 3
        action = appliquer_charte(clauses, risque, categorie, mode_domicile)
        if action != "diffuser" or risque == "aucun" or rng.random() >= P_REJET_DIFFUSER:
            break
    charte = rendre_charte(clauses, rng, train, test, mode)
    cible = tirer_cible(rng, faker, risque)
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


def generer_jeu_manuscrit(graine: int = 0, n: int = 1000) -> list[dict]:
    """Jeu de test sur les 20 chartes écrites à la main (même tirage de cibles et de contexte que les autres)."""
    from irl_guard.donnees.chartes_manuscrites import CHARTES
    rng = random.Random(graine + 1_000_003)  # flux distinct des quatre jeux synthétiques
    faker = Faker("fr_FR")
    faker.seed_instance(graine + 1_000_003)
    lignes = []
    for i in range(n):
        charte = CHARTES[i % len(CHARTES)]
        clauses = list(charte.clauses)
        while True:
            tirage = rng.random()
            if tirage < P_AUCUN:
                risque = "aucun"
            elif tirage < P_AUCUN + (1 - P_AUCUN) * P_MENTIONNE:
                risque = rng.choice([c.risque for c in clauses])
            else:
                risque = rng.choice(RISQUES_CHARTE)
            categorie = rng.choice(CATEGORIES_STREAM)
            mode_domicile = rng.random() < 1 / 3
            action = appliquer_charte(clauses, risque, categorie, mode_domicile)
            if action != "diffuser" or risque == "aucun" or rng.random() >= P_REJET_DIFFUSER:
                break
        cible = tirer_cible(rng, faker, risque)
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
        state = {"categorie": categorie, "mode_domicile": mode_domicile, "charte": charte.texte,
                 "cible": description, "contexte": tirer_contexte(rng)}
        lignes.append({
            "id": f"test_manuscrites_{i:06d}",
            "state": json.dumps(state, ensure_ascii=False),
            "questions": json.dumps(QUESTIONS, ensure_ascii=False),
            "gold": json.dumps(construire_gold(action, cible.risque), ensure_ascii=False),
            "clauses": json.dumps([c.to_dict() for c in clauses], ensure_ascii=False),
        })
    return lignes
