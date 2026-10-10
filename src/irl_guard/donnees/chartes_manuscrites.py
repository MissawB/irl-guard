"""Vingt chartes écrites à la main, hors gabarits, annotées en clauses structurées.

Elles mesurent si un modèle applique une charte rédigée comme un vrai streamer l'écrirait (ton familier,
abréviations, formulations indirectes), et non dans le style des gabarits d'entraînement. Les clauses sont
l'annotation : le label se calcule avec `appliquer_charte`, comme pour le dataset synthétique.
"""

from __future__ import annotations

from dataclasses import dataclass

from irl_guard.donnees.clauses import Clause


@dataclass(frozen=True)
class CharteManuscrite:
    texte: str
    clauses: tuple[Clause, ...]


def _c(risque: str, sanction: str, condition: str = "toujours") -> Clause:
    return Clause(risque, sanction, condition)


CHARTES = (
    CharteManuscrite(
        "Salut la team ! Ici c'est familial : pas de gros mots à l'image, tu mets du flou dessus. Et quand je suis "
        "chez moi, si on aperçoit mon adresse, on coupe tout.",
        (_c("propos", "flouter"), _c("localisation", "pause", "domicile"))),
    CharteManuscrite(
        "Quand je fais un live cuisine, les couteaux c'est ok. Sinon, pixelise-les. Personne ne doit voir mon "
        "numéro de tel ou mes mails.",
        (_c("arme", "flouter", "sauf_categorie:cuisine"), _c("donnee_perso", "flouter"))),
    CharteManuscrite(
        "Règle n°1 : JAMAIS mon adresse ni le nom de ma rue, peu importe où je suis. Écran de pause immédiat.",
        (_c("localisation", "pause"),)),
    CharteManuscrite(
        "jsp pk mais les gens arrêtent pas d'écrire des trucs vulgaires sur des pancartes. si ça arrive, "
        "brouille-les stp",
        (_c("propos", "flouter"),)),
    CharteManuscrite(
        "Je stream en extérieur et je ne veux pas qu'on puisse me retrouver. Les noms de rues et numéros, tu les "
        "rends illisibles. Si je suis près de mon appart, carrément pause.",
        (_c("localisation", "flouter"), _c("localisation", "pause", "domicile"))),
    CharteManuscrite(
        "Pas de couteau ni de ciseaux visibles en stream, quel que soit le jeu ou l'activité, sauf quand je "
        "cuisine. Dans ce cas ils restent nets.",
        (_c("arme", "flouter", "sauf_categorie:cuisine"),)),
    CharteManuscrite(
        "Insulter c'est permis ici, on est entre adultes. Mais les mails et numéros de téléphone des gens, on ne "
        "se contente pas de flouter : on coupe carrément le live.",
        (_c("propos", "diffuser"), _c("donnee_perso", "pause"))),
    CharteManuscrite(
        "Mode maison activé = zéro info qui permette de savoir où je vis. Dans ce cas, pause immédiate.",
        (_c("localisation", "pause", "domicile"),)),
    CharteManuscrite(
        "Tout public. Armes réelles interdites à l'écran (floutage), propos haineux aussi. Le reste passe.",
        (_c("arme", "flouter"), _c("propos", "flouter"))),
    CharteManuscrite(
        "Je fais du sport en salle : on s'en fiche des objets autour, rien à cacher de ce côté. Par contre si "
        "quelqu'un montre une adresse mail ou un numéro, tu le rends flou.",
        (_c("arme", "diffuser"), _c("donnee_perso", "flouter"))),
    CharteManuscrite(
        "Pendant les sessions gaming, rien à flouter côté couteaux, c'est des jeux. En dehors du gaming, un "
        "couteau qui traîne = flou.",
        (_c("arme", "flouter", "sauf_categorie:gaming"),)),
    CharteManuscrite(
        "Quand je suis chez moi, je veux aucune info perso visible (nom, téléphone, mail) : coupe le stream. "
        "Ailleurs, juste un flou.",
        (_c("donnee_perso", "pause", "domicile"), _c("donnee_perso", "flouter"))),
    CharteManuscrite(
        "Chat poli, écran poli : si un texte injurieux apparaît, mets le direct en pause le temps qu'il disparaisse.",
        (_c("propos", "pause"),)),
    CharteManuscrite(
        "Je veux pas qu'on devine mon quartier. Plaques de rue, numéros de bâtiment, codes postaux : flou partout.",
        (_c("localisation", "flouter"),)),
    CharteManuscrite(
        "Stream musique en terrasse. Les gens peuvent montrer ce qu'ils veulent, sauf leurs coordonnées perso "
        "que tu brouilles, et zéro lame ou couteau à l'image, sinon pause.",
        (_c("donnee_perso", "flouter"), _c("arme", "pause"))),
    CharteManuscrite(
        "Ma seule exigence : quand je suis à domicile, tout ce qui révèle où je me trouve doit être illisible. "
        "Pour le reste je m'en fiche.",
        (_c("localisation", "flouter", "domicile"),)),
    CharteManuscrite(
        "Jurer, on s'en fout, c'est un live entre potes. Un couteau ou n'importe quelle arme à l'écran : écran "
        "de pause direct.",
        (_c("propos", "diffuser"), _c("arme", "pause"))),
    CharteManuscrite(
        "En cuisine les couteaux font partie du décor. Dans les autres lives, si un couteau apparaît, je préfère "
        "qu'on coupe. Et jamais de numéro de téléphone visible, du flou même en cuisine.",
        (_c("arme", "pause", "sauf_categorie:cuisine"), _c("donnee_perso", "flouter"))),
    CharteManuscrite(
        "Quand le mode domicile est ON : pause dès qu'une rue ou une adresse est lisible, et flou sur les mails "
        "et téléphones. Sinon, rien de spécial.",
        (_c("localisation", "pause", "domicile"), _c("donnee_perso", "flouter", "domicile"))),
    CharteManuscrite(
        "Live familial le dimanche : aucune grossièreté, aucune insulte, rien de violent. Si ça arrive, flou. Et "
        "rien qui donne mon adresse, jamais : pause.",
        (_c("propos", "flouter"), _c("arme", "flouter"), _c("localisation", "pause"))),
)
