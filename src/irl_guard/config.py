"""Configuration d'une session : chargée depuis un YAML et validée par pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SeuilsConfig(_Strict):
    pause: float = Field(0.5, ge=0, le=1)
    flou: float = Field(0.4, ge=0, le=1)
    hysteresis_haut: float = Field(0.7, ge=0, le=1)
    hysteresis_bas: float = Field(0.3, ge=0, le=1)
    sortie_k: int = Field(3, ge=1)
    nudite: float = Field(0.6, ge=0, le=1)
    diff_max: int = Field(6, ge=0, le=255)
    max_reutilisations: int = Field(2, ge=0)

    @model_validator(mode="after")
    def _bas_sous_haut(self) -> SeuilsConfig:
        if self.hysteresis_bas >= self.hysteresis_haut:
            raise ValueError("hysteresis_bas doit être strictement inférieur à hysteresis_haut")
        return self


class TailleConfig(_Strict):
    petite_max: float = 0.01
    moyenne_max: float = 0.05


class LayaConfig(_Strict):
    url: str = "http://localhost:8888"
    modele: str = "laya-multilingual"
    cle: str | None = None
    timeout_s: float = Field(5.0, gt=0)


class ReglesConfig(_Strict):
    categories_couteau_ok: list[str] = Field(default_factory=lambda: ["cuisine"])
    mots_offensants: list[str] = Field(default_factory=list)


class PoidsHF(_Strict):
    repo: str
    fichier: str


class DetecteursConfig(_Strict):
    objets: str = "yolo11n.pt"
    visages: PoidsHF = Field(default_factory=lambda: PoidsHF(
        repo="AdamCodd/YOLOv11n-face-detection", fichier="model.pt"))
    plaques: PoidsHF = Field(default_factory=lambda: PoidsHF(
        repo="morsetechlab/yolov11-license-plate-detection", fichier="license-plate-finetune-v1n.pt"))
    conf_min: float = Field(0.3, ge=0, le=1)
    ocr_conf_min: float = Field(0.5, ge=0, le=1)
    device: str | None = None


def _couts_par_defaut() -> dict[str, dict[str, float]]:
    # couts[action prédite][action attendue]
    return {
        "diffuser": {"diffuser": 0, "flouter": 10, "pause": 20},
        "flouter": {"diffuser": 1, "flouter": 0, "pause": 5},
        "pause": {"diffuser": 3, "flouter": 1, "pause": 0},
    }


class EvaluationConfig(_Strict):
    couts: dict[str, dict[str, float]] = Field(default_factory=_couts_par_defaut)


class SessionConfig(_Strict):
    categorie: str = "IRL"
    charte: str
    mode_domicile: bool = False
    fps_analyse: float = Field(3.0, gt=0, le=30)
    seuils: SeuilsConfig = Field(default_factory=SeuilsConfig)
    taille: TailleConfig = Field(default_factory=TailleConfig)
    masquage: Literal["pixel", "flou"] = "pixel"
    marge: float = Field(0.15, ge=0, le=1)
    maintien: int = Field(3, ge=0)
    couper_audio_en_pause: bool = True
    budget_tokens_state: int = Field(700, ge=100)
    ecran_pause: Path | None = None
    laya: LayaConfig = Field(default_factory=LayaConfig)
    regles: ReglesConfig = Field(default_factory=ReglesConfig)
    detecteurs: DetecteursConfig = Field(default_factory=DetecteursConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def load_config(path: str | Path) -> SessionConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return SessionConfig.model_validate(data)
