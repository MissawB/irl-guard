from irl_guard.tracking import TrackManager, fusionner_texte
from irl_guard.types import Box, Detection


def _texte(texte, x=100):
    return Detection("texte", 0.95, Box(x, 100, x + 200, 140), texte)


def test_fusionner_texte():
    assert fusionner_texte("", "Rue") == "Rue"
    assert fusionner_texte("Rue Nationale", "Nationale") == "Rue Nationale"
    assert fusionner_texte("Rue Nat", "Rue Nationale") == "Rue Nationale"
    assert fusionner_texte("Rue Nationale", "75011") == "Rue Nationale | 75011"
    assert fusionner_texte("Rue", "") == "Rue"
    assert len(fusionner_texte("a" * 150, "b" * 150)) == 200


def test_id_stable_sur_plusieurs_analyses():
    tm = TrackManager(maintien=3)
    ids, nouveaux = [], []
    for _ in range(4):
        pistes = tm.update([_texte("Rue Nationale")])
        vivantes = [p for p in pistes if not p.perdu]
        assert len(vivantes) == 1
        ids.append(vivantes[0].id)
        nouveaux.append(vivantes[0].nouveau)
    assert len(set(ids)) == 1
    assert nouveaux == [True, False, False, False]


def test_texte_accumule_et_drapeau_modifie():
    tm = TrackManager(maintien=3)
    tm.update([_texte("Rue Nat")])
    p = [p for p in tm.update([_texte("Rue Nationale")]) if not p.perdu][0]
    assert p.texte == "Rue Nationale" and p.modifie is True
    p = [p for p in tm.update([_texte("Nationale")]) if not p.perdu][0]
    assert p.texte == "Rue Nationale" and p.modifie is False


def test_piste_perdue_maintenue_puis_supprimee():
    tm = TrackManager(maintien=2)
    tm.update([_texte("Rue Nationale")])
    for _ in range(2):
        pistes = tm.update([])
        assert len(pistes) == 1 and pistes[0].perdu is True
    assert tm.update([]) == []


def test_deux_objets_et_deux_classes_ont_des_ids_distincts():
    tm = TrackManager(maintien=3)
    pistes = tm.update([
        _texte("Rue A", x=10),
        _texte("Rue B", x=400),
        Detection("visage", 0.9, Box(10, 100, 210, 140)),
    ])
    assert len({p.id for p in pistes}) == 3


def test_les_copies_ne_modifient_pas_l_etat_interne():
    tm = TrackManager(maintien=3)
    p = tm.update([_texte("Rue")])[0]
    p.texte = "modifié"
    assert tm.update([_texte("Rue")])[0].texte == "Rue"
