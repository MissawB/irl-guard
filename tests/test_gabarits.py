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
