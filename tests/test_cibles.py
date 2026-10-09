import random

from faker import Faker

from irl_guard.config import SessionConfig
from irl_guard.decision.rules import RulesDecider
from irl_guard.donnees.cibles import bruiter, tirer_cible


def _faker():
    f = Faker("fr_FR")
    f.seed_instance(0)
    return f


def test_tous_les_risques_sont_tires():
    rng, faker = random.Random(0), _faker()
    cibles = [tirer_cible(rng, faker) for _ in range(500)]
    assert {c.risque for c in cibles} == {"localisation", "donnee_perso", "arme", "propos", "aucun"}


def test_coherence_type_texte_classe():
    rng, faker = random.Random(1), _faker()
    for _ in range(300):
        c = tirer_cible(rng, faker)
        if c.risque == "arme":
            assert c.type == "objet" and c.classe in ("couteau", "ciseaux") and c.texte is None
        else:
            assert c.type == "texte" and c.classe is None and c.texte


def test_cibles_reconnues_par_les_regles():
    # Les textes générés sans bruit doivent être cohérents avec leur risque (contrôle de vraisemblance).
    rules = RulesDecider(SessionConfig(charte="x"))
    rng, faker = random.Random(2), _faker()
    for _ in range(300):
        c = tirer_cible(rng, faker)
        if c.risque in ("localisation", "propos"):
            state = {"categorie": "IRL", "mode_domicile": False, "charte": "x",
                     "cible": {"type": "texte", "texte": c.texte, "conf": 0.9, "taille": "moyenne"}, "contexte": []}
            assert rules.decide(state).risque == c.risque, c.texte


def test_bruit_modifie_le_texte_le_plus_souvent():
    rng = random.Random(3)
    modifies = sum(bruiter("Rue Nationale", rng) != "Rue Nationale" for _ in range(200))
    assert modifies > 150


def test_bruit_sur_textes_courts():
    rng = random.Random(4)
    for texte in ("12", "75011", "a", "Bd"):
        for _ in range(100):
            assert bruiter(texte, rng)


def test_bruit_deterministe():
    assert bruiter("Rue Nationale", random.Random(5)) == bruiter("Rue Nationale", random.Random(5))
