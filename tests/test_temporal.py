from irl_guard.temporal import Hysteresis


def test_entree_immediate_sortie_apres_k_calmes():
    h = Hysteresis(haut=0.7, bas=0.3, k=3)
    sequence = [0.1, 0.8, 0.2, 0.2, 0.5, 0.2, 0.2, 0.2, 0.1]
    assert [h.update(s) for s in sequence] == [False, True, True, True, True, True, True, False, False]


def test_valeur_entre_les_seuils_ne_declenche_pas():
    h = Hysteresis(haut=0.7, bas=0.3, k=1)
    assert [h.update(s) for s in (0.5, 0.69, 0.7, 0.29)] == [False, False, True, False]
