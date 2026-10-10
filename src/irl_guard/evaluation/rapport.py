"""Prédictions par approche et rapport Markdown comparatif avec diagrammes de fiabilité."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from irl_guard.config import SessionConfig
from irl_guard.decision.laya import LayaDecider, LayaIndisponible
from irl_guard.decision.base import Decision
from irl_guard.decision.rules import RulesDecider
from irl_guard.donnees.lignes import lire_jeu
from irl_guard.evaluation.metriques import Resultat
from irl_guard.evaluation.predicteurs import (
    charger_predictions, combiner_predictions, predire_avec_decider, reponses_depuis_decision,
)
from irl_guard.evaluation.xgb import PredicteurXGBoost

log = logging.getLogger(__name__)
APPROCHES = ("toujours-diffuser", "regles", "xgboost", "laya", "laya-ft", "qwen-ft", "combine-laya", "combine-qwen")
SOURCES_COMBINEES = {"combine-laya": "laya-ft", "combine-qwen": "qwen-ft",
                     "combine-laya-v2": "laya-ft-v2", "combine-qwen-v2": "qwen-ft-v2"}
SEUIL_REPLIS = 0.05  # au-delà, les réponses de Laya ne mesurent plus Laya mais le repli
JEUX_TEST = ("test_vues", "test_non_vues", "test_manuscrites")
TITRES = {"test_vues": "Chartes vues", "test_non_vues": "Chartes non vues",
          "test_manuscrites": "Chartes écrites à la main"}


def obtenir_predictions(approche: str, cfg: SessionConfig, dossier_donnees: Path, dossier_predictions: Path,
                        jeux: dict[str, list[dict]]) -> dict[str, dict | None]:
    if approche in SOURCES_COMBINEES:
        modele = obtenir_predictions(SOURCES_COMBINEES[approche], cfg, dossier_donnees, dossier_predictions, jeux)
        regles = obtenir_predictions("regles", cfg, dossier_donnees, dossier_predictions, jeux)
        return {j: combiner_predictions(regles[j], modele[j], cfg.seuils) if modele[j] is not None else None
                for j in jeux}
    if approche == "toujours-diffuser":
        reponse = reponses_depuis_decision(Decision("diffuser", "aucun", 0))
        return {j: {l["id"]: reponse for l in lignes} for j, lignes in jeux.items()}
    if approche == "regles":
        decider = RulesDecider(cfg)
        return {j: predire_avec_decider(decider, l) for j, l in jeux.items()}
    if approche == "xgboost":
        modele = PredicteurXGBoost().entrainer(lire_jeu(Path(dossier_donnees) / "train.jsonl"))
        return {j: modele.predire(l) for j, l in jeux.items()}
    if approche == "laya":
        decider = LayaDecider(cfg)
        try:
            decider.verifier()
        except LayaIndisponible as e:
            log.warning("Laya absent du rapport : %s", e)
            return {j: None for j in jeux}
        resultat = {}
        for j, lignes in jeux.items():
            decisions = {l["id"]: decider.decide(json.loads(l["state"])) for l in lignes}
            replis = sum(d.source == "fallback" for d in decisions.values())
            if lignes and replis / len(lignes) > SEUIL_REPLIS:
                log.warning("Laya absent pour %s : %d réponses sur %d en repli", j, replis, len(lignes))
                resultat[j] = None
            else:
                resultat[j] = {i: reponses_depuis_decision(d) for i, d in decisions.items()}
        return resultat
    resultat = {}
    for j in jeux:
        chemin = Path(dossier_predictions) / approche / f"predictions_{j}.jsonl"
        resultat[j] = charger_predictions(chemin) if chemin.exists() else None
    return resultat


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def _diagramme(resultats: dict[str, dict[str, Resultat | None]], jeu: str, chemin: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="calibration parfaite")
    for approche, par_jeu in resultats.items():
        r = par_jeu.get(jeu)
        if r is not None and r.fiabilite:
            ax.plot([p[0] for p in r.fiabilite], [p[1] for p in r.fiabilite], marker="o", label=approche)
    ax.set_xlabel("confiance moyenne")
    ax.set_ylabel("précision observée")
    ax.set_title(f"Fiabilité — {TITRES.get(jeu, jeu)}")
    ax.legend(loc="lower right")
    fig.savefig(chemin, dpi=120, bbox_inches="tight")
    plt.close(fig)


def ecrire_rapport(resultats: dict[str, dict[str, Resultat | None]], dossier: str | Path) -> Path:
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    lignes = ["# Résultats du decision model", "",
              "Taux de fuite : part des cas à protéger (flouter ou pause attendu) que l'approche diffuse.",
              "« toujours-diffuser » est le plancher : une approche qui ne le dépasse pas n'a rien appris.",
              "ECE : calibration de la question action (option la plus probable), sur 10 classes de confiance.", ""]
    jeux = sorted({j for par_jeu in resultats.values() for j in par_jeu}, key=lambda j: JEUX_TEST.index(j)
                  if j in JEUX_TEST else len(JEUX_TEST))
    for jeu in jeux:
        lignes += [f"## {TITRES.get(jeu, jeu)}", "",
                   "| Approche | n | Action | F1 macro | Risque | Gravité | Fuite | ECE | Coût moyen |",
                   "|---|---|---|---|---|---|---|---|---|"]
        for approche, par_jeu in resultats.items():
            r = par_jeu.get(jeu)
            if r is None:
                lignes.append(f"| {approche} | absent | | | | | | | |")
            else:
                lignes.append(f"| {approche} | {r.n} | {_pct(r.precision_action)} | {r.f1_macro_action:.3f} | {_pct(r.precision_risque)} | "
                              f"{_pct(r.precision_gravite)} | {_pct(r.taux_fuite)} | {r.ece:.3f} | {r.cout_moyen:.2f} |")
        image = dossier / f"fiabilite_{jeu}.png"
        _diagramme(resultats, jeu, image)
        lignes += ["", f"![Fiabilité — {TITRES.get(jeu, jeu)}]({image.name})", ""]
    chemin = dossier / "rapport.md"
    chemin.write_text("\n".join(lignes), encoding="utf-8")
    return chemin
