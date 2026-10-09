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


def tirer_cible(rng: random.Random, faker: Faker, risque: str | None = None) -> CibleTiree:
    if risque is None:
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
