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

# Sujets génériques : chacun couvre toutes les cibles tirées pour son risque (cibles.py), sinon le label
# contredirait le texte (« floute les armes à feu » ne dit rien d'un couteau).
SUJETS = {
    "localisation": ("les adresses", "les indications de lieu", "les repères géographiques",
                     "les infos de localisation", "les éléments qui me localisent"),
    "donnee_perso": ("les infos personnelles", "les données privées", "les coordonnées des gens",
                     "les informations perso", "les éléments qui identifient quelqu'un"),
    "arme": ("les armes", "les objets tranchants", "les objets dangereux", "les lames",
             "les objets qui peuvent blesser"),
    "propos": ("les insultes", "les gros mots", "les injures", "les grossièretés", "les propos insultants"),
}

# v2 : environ 20 formulations par sanction et 8 préfixes par condition (les 5 et 4-5 de la v1 sont conservés).
# Aucune expression des chartes écrites à la main n'est reprise (voir tests/test_gabarits.py, LISTE_NOIRE).
VERBES = {
    "diffuser": (
        "Tu peux montrer {sujet}.", "Je n'ai aucun problème avec {sujet}.", "{Sujet} : c'est autorisé.",
        "Aucune raison de dissimuler {sujet}.", "Laisse passer {sujet}.",
        "{Sujet} ne me dérangent pas.", "Pas besoin d'agir sur {sujet}.", "Pas de souci pour {sujet}.",
        "{Sujet} : aucun traitement.", "Je tolère {sujet}.", "Ne filtre pas {sujet}.", "{Sujet} restent visibles.",
        "Garde {sujet} tels quels.", "On accepte {sujet}.", "Ça ne me gêne pas de voir {sujet}.",
        "Diffuse {sujet} normalement.", "{Sujet} sont tolérés.", "Je n'interviens pas sur {sujet}.",
        "Affiche {sujet} sans retouche.", "{Sujet}, je les autorise.",
    ),
    "flouter": (
        "Floute {sujet}.", "Cache {sujet} à l'écran.", "Masque {sujet}.", "Je ne veux pas voir {sujet} en clair.",
        "{Sujet} : flou obligatoire.",
        "Escamote {sujet}.", "Ajoute un filtre flou par-dessus {sujet}.", "Estompe {sujet}.",
        "Noircis {sujet}.", "Recouvre {sujet}.", "Occulte {sujet}.", "Transforme {sujet} en tache floue.",
        "{Sujet} doivent disparaître de l'image.", "Efface {sujet} de l'écran.", "Assombris {sujet}.",
        "{Sujet} jamais en clair.", "Trouble {sujet}.", "Floutage obligatoire pour {sujet}.",
        "Il faut que {sujet} soient floutés.", "Grise {sujet}.",
    ),
    "pause": (
        "Coupe le live si on voit {sujet}.", "{Sujet} : mets le stream en pause.",
        "Passe en écran de pause dès que {sujet} apparaissent à l'écran.",
        "Interromps la diffusion si {sujet} apparaissent.", "Dès qu'on voit {sujet}, pause immédiate.",
        "Stoppe le live si {sujet} apparaissent.", "Arrête la diffusion quand {sujet} sont visibles.",
        "Suspends le stream si {sujet} se montrent.", "{Sujet} = live suspendu.",
        "Mets la diffusion sur pause si {sujet} s'affichent.", "Interruption du stream si {sujet} sont à l'image.",
        "Arrêt total du live si {sujet} passent à l'écran.", "Gèle le stream dès que {sujet} surgissent.",
        "Le live s'arrête net quand {sujet} se montrent.", "{Sujet} : on met tout en pause.",
        "Bascule sur l'écran d'attente si {sujet} sont visibles.", "Fige la diffusion quand {sujet} débarquent.",
        "Pause obligatoire si {sujet} sont filmés.", "Le stream doit s'interrompre si {sujet} apparaissent.",
        "Écran noir si {sujet} sont filmés.",
    ),
}

PREFIXES = {
    "toujours": ("", "Dans tous les cas, ", "Quel que soit le contexte, ", "En toutes circonstances, ",
                 "Sans exception, ", "Peu importe la situation, ", "À chaque stream, ", "Systématiquement, "),
    "domicile": ("En mode domicile, ", "Quand je suis près de chez moi, ", "Près de mon domicile, ",
                 "Lorsque le mode domicile est activé, ", "Autour de chez moi, ",
                 "Lorsque je streame depuis mon logement, ", "En direct de mon domicile, ",
                 "Quand le mode domicile tourne, ", "Si je streame depuis ma maison, "),
    "sauf_categorie": ("Sauf en live {cat}, ", "Hors stream {cat}, ", "Excepté pendant les lives {cat}, ",
                       "En dehors des streams {cat}, ", "Pour tout stream qui n'est pas {cat}, ",
                       "Hormis les lives {cat}, ", "Excepté en catégorie {cat}, ",
                       "Pour les streams autres que {cat}, "),
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
