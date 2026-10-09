from irl_guard.types import Box, Detection


def test_area_et_iou():
    a = Box(0, 0, 10, 10)
    b = Box(5, 0, 15, 10)
    assert a.area == 100
    assert a.iou(b) == 50 / 150
    assert a.iou(Box(20, 20, 30, 30)) == 0.0


def test_expand_elargit_et_reste_dans_l_image():
    assert Box(10, 10, 30, 50).expand(0.5, 100, 100) == Box(5, 0, 35, 60)
    assert Box(0, 0, 100, 100).expand(0.2, 100, 100) == Box(0, 0, 100, 100)


def test_lerp():
    a, b = Box(0, 0, 10, 10), Box(10, 20, 30, 40)
    assert a.lerp(b, 0.5) == Box(5, 10, 20, 25)
    assert a.lerp(b, 0.0) == a


def test_detection_aller_retour_dict():
    d = Detection("texte", 0.912, Box(10.5, 20, 30, 40.5), "Rue Nationale")
    assert Detection.from_dict(d.to_dict()) == Detection("texte", 0.912, Box(10.5, 20, 30, 40.5), "Rue Nationale")
