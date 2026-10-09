import pytest
from pydantic import ValidationError

from irl_guard.config import SessionConfig, load_config


def test_valeurs_par_defaut():
    cfg = SessionConfig(charte="Tout public.")
    assert cfg.categorie == "IRL"
    assert cfg.mode_domicile is False
    assert cfg.seuils.pause == 0.5
    assert cfg.seuils.flou == 0.4
    assert cfg.masquage == "pixel"
    assert cfg.laya.modele == "laya-multilingual"
    assert cfg.detecteurs.plaques.repo == "morsetechlab/yolov11-license-plate-detection"


def test_chargement_yaml_avec_surcharge(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("charte: Pas de rue.\nmode_domicile: true\nseuils:\n  pause: 0.6\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.mode_domicile is True
    assert cfg.seuils.pause == 0.6
    assert cfg.seuils.flou == 0.4


def test_cle_inconnue_refusee():
    with pytest.raises(ValidationError):
        SessionConfig(charte="x", mode_domicil=True)


def test_charte_obligatoire(tmp_path):
    p = tmp_path / "vide.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(p)


def test_hysteresis_incoherente_refusee():
    with pytest.raises(ValidationError):
        SessionConfig(charte="x", seuils={"hysteresis_haut": 0.3, "hysteresis_bas": 0.5})
