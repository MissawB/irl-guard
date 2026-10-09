"""Formulations françaises des clauses : préfixe de condition + verbe de sanction + sujet de risque.

Environ 30 % de chaque liste de composants est réservé au test « chartes non vues » : une
formulation de test contient au moins un composant réservé, donc jamais vue à l'entraînement.
Aucun composant n'est sous-chaîne d'un autre de la même liste : garder cette propriété en ajoutant
des gabarits (les tests de fuite cherchent les composants réservés dans le texte).
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
