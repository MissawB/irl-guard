import random

from irl_guard.donnees.clauses import RISQUES_CHARTE, SANCTIONS, TYPES_CONDITION, Clause
from irl_guard.donnees.gabarits import (
    PREFIXES, SUJETS, VERBES, choisir_composants, composer, partitionner,
)


def test_composer():
    assert composer("les noms de rue", "Floute {sujet}.", "", "") == "Floute les noms de rue."
    assert composer("les noms de rue", "Floute {sujet}.", "En mode domicile, ", "") == \
        "En mode domicile, floute les noms de rue."
    assert composer("les couteaux", "{Sujet} : c'est autorisé.", "Sauf en live {cat}, ", "cuisine") == \
        "Sauf en live cuisine, les couteaux : c'est autorisé."


def test_partition_deterministe():
    assert partitionner(0) == partitionner(0)
    assert partitionner(0) != partitionner(1)


def test_partition_disjointe():
    train, test = partitionner(0)
    for source, cle in ((SUJETS, "sujets"), (VERBES, "verbes"), (PREFIXES, "prefixes")):
        for nom, elements in source.items():
            tr, te = getattr(train, cle)[nom], getattr(test, cle)[nom]
            assert set(tr) | set(te) == set(elements)
            assert not set(tr) & set(te)
            assert len(te) >= 1 and len(tr) >= 1


def test_au_moins_5_formulations_par_combinaison_et_par_partition():
    train, test = partitionner(0)
    for r in RISQUES_CHARTE:
        for s in SANCTIONS:
            for t in TYPES_CONDITION:
                total = len(SUJETS[r]) * len(VERBES[s]) * len(PREFIXES[t])
                n_train = len(train.sujets[r]) * len(train.verbes[s]) * len(train.prefixes[t])
                assert n_train >= 5
                assert total - n_train >= 5  # formulations contenant au moins une partie réservée


def test_train_n_utilise_que_des_composants_train():
    train, test = partitionner(0)
    rng = random.Random(1)
    clause = Clause("localisation", "flouter", "domicile")
    for _ in range(200):
        sujet, verbe, prefixe = choisir_composants(clause, rng, train, test, "train")
        assert sujet in train.sujets["localisation"]
        assert verbe in train.verbes["flouter"]
        assert prefixe in train.prefixes["domicile"]


def test_test_utilise_au_moins_un_composant_reserve():
    train, test = partitionner(0)
    rng = random.Random(2)
    clause = Clause("propos", "pause", "toujours")
    for _ in range(200):
        sujet, verbe, prefixe = choisir_composants(clause, rng, train, test, "test")
        assert (sujet in test.sujets["propos"] or verbe in test.verbes["pause"]
                or prefixe in test.prefixes["toujours"])


def test_sujets_generiques_couvrent_toutes_les_cibles_du_risque():
    # Un sujet précis (« les armes à feu ») étiquèterait « flouter » un couteau : seuls des sujets génériques.
    specifiques = ("couteau", "ciseaux", "à feu", "blanches", "codes postaux", "numéros de maison", "noms de rue",
                   "plaques de rue", "panneaux de rue", "e-mail", "téléphone", "noms des gens")
    for sujets in SUJETS.values():
        for sujet in sujets:
            assert not any(s in sujet for s in specifiques), sujet


def test_aucun_composant_n_est_sous_chaine_d_un_autre():
    fragments = [s for t in SUJETS.values() for s in t]
    fragments += [v.split("{")[0].strip().lower() for t in VERBES.values() for v in t if not v.startswith("{")]
    fragments += [p.split("{")[0].strip().lower() for t in PREFIXES.values() for p in t if p]
    for a in fragments:
        for b in fragments:
            if a != b:
                assert a not in b, (a, b)


# Expressions des 20 chartes écrites à la main : aucun nouveau gabarit ne peut les reprendre,
# sinon le jeu « test_manuscrites » mesurerait la mémorisation et non la généralisation.
LISTE_NOIRE = ("pixelis", "brouill", "illisible", "flou partout", "du flou", "un flou", "= flou", "flou sur",
               "rends flou", "coupe tout", "on coupe", "coupe le stream", "coupe le direct", "qu'on coupe",
               "écran de pause", "pause immédiat", "pause direct", "sinon pause", "en pause le temps", "c'est ok",
               "restent nets", "permis", "on s'en fiche", "on s'en fout", "rien à cacher", "rien à flouter",
               "le reste passe", "chez moi", "appart", "mode maison", "à domicile", "est on", "live cuisine",
               "quand je cuisine", "session", "en dehors du", "en cuisine", "autres lives", "si ça arrive")
# Gabarits v1, antérieurs à l'écriture des chartes manuscrites : autorisés même s'ils contiennent une expression.
GABARITS_V1_AUTORISES = {"Passe en écran de pause dès que {sujet} apparaissent à l'écran.",
                         "Dès qu'on voit {sujet}, pause immédiate.", "Quand je suis près de chez moi, ",
                         "Autour de chez moi, "}


def test_aucun_gabarit_ne_reprend_les_chartes_manuscrites():
    composants = [c for source in (SUJETS, VERBES, PREFIXES) for t in source.values() for c in t]
    for composant in composants:
        if composant in GABARITS_V1_AUTORISES:
            continue
        for expression in LISTE_NOIRE:
            assert expression not in composant.lower(), (expression, composant)


def test_formulations_enrichies():
    assert all(len(v) >= 20 for v in VERBES.values())
    assert all(len(p) >= 8 for p in PREFIXES.values())
