"""Ligne de commande : `irl-guard process` et `irl-guard replay`."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from pathlib import Path

import typer

from irl_guard.config import SessionConfig, load_config
from irl_guard.decision.base import Decider
from irl_guard.decision.laya import LayaDecider, LayaIndisponible
from irl_guard.decision.rules import RulesDecider
from irl_guard.journal import lire_detections
from irl_guard.perception.base import CachedPerception, DetecteurIndisponible, JournalIncompatible
from irl_guard.perception.factory import construire_perception
from irl_guard.pipeline import traiter_video

app = typer.Typer(add_completion=False, help="Bloqueur de contenu pour lives IRL (V1 hors ligne).")
donnees_app = typer.Typer(help="Données synthétiques pour fine-tuner le decision model.")
app.add_typer(donnees_app, name="donnees")


@donnees_app.command("generer")
def donnees_generer(
    sortie: Path = typer.Option(Path("data/decision"), "--sortie"),
    graine: int = typer.Option(0, "--graine"),
    train: int = typer.Option(4000, "--train"),
    val: int = typer.Option(500, "--val"),
    test_vues: int = typer.Option(1000, "--test-vues"),
    test_non_vues: int = typer.Option(1000, "--test-non-vues"),
) -> None:
    """Génère les quatre jeux JSONL (train, val, test_vues, test_non_vues)."""
    from irl_guard.donnees.lignes import ecrire_jeux, generer_jeux
    jeux = generer_jeux(graine, {"train": train, "val": val, "test_vues": test_vues, "test_non_vues": test_non_vues})
    for chemin in ecrire_jeux(jeux, sortie):
        typer.echo(f"{chemin} : {len(jeux[chemin.stem])} lignes")


@donnees_app.command("publier")
def donnees_publier(
    dossier: Path = typer.Option(Path("data/decision"), "--dossier", exists=True, file_okay=False),
    repo: str = typer.Option("MissawB/irl-guard-decisions", "--repo"),
) -> None:
    """Publie les jeux en dataset privé sur le Hub (nécessite `hf auth login`)."""
    from irl_guard.donnees.publication import publier
    publier(dossier, repo)
    typer.echo(f"Publié en privé : https://huggingface.co/datasets/{repo}")


class ChoixDecider(str, Enum):
    laya = "laya"
    rules = "rules"


def creer_decider(choix: ChoixDecider, cfg: SessionConfig) -> Decider:
    if choix == ChoixDecider.rules:
        return RulesDecider(cfg)
    decider = LayaDecider(cfg)
    decider.verifier()
    return decider


def _executer(entree: Path, out: Path, config: Path, debug_overlay: bool, choix: ChoixDecider,
              fabrique_perception: Callable[[SessionConfig], object]) -> None:
    cfg = load_config(config)
    try:
        decider = creer_decider(choix, cfg)
    except LayaIndisponible as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2)
    try:
        perception = fabrique_perception(cfg)
    except DetecteurIndisponible as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(3)
    try:
        res = traiter_video(entree, out, cfg, perception, decider, debug=debug_overlay)
    except JournalIncompatible as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(4)
    typer.echo(f"Vidéo protégée : {res.sortie}")
    if res.sortie_debug:
        typer.echo(f"Vidéo de debug : {res.sortie_debug}")
    typer.echo(f"Journal : {res.journal}")
    typer.echo(f"{res.images} images, {res.analyses} analyses, {res.duree_traitement_s:.1f} s "
               f"— débit {res.debit:.2f}x temps réel")


@app.command()
def process(
    entree: Path = typer.Argument(..., exists=True, dir_okay=False, help="Vidéo à protéger"),
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False, help="YAML de session"),
    out: Path = typer.Option(..., "--out", help="Vidéo protégée en sortie"),
    debug_overlay: bool = typer.Option(False, "--debug-overlay", help="Produit aussi <out>_debug.mp4"),
    decider: ChoixDecider = typer.Option(ChoixDecider.laya, "--decider"),
) -> None:
    """Analyse une vidéo et produit sa version protégée."""
    _executer(entree, out, config, debug_overlay, decider, construire_perception)


@app.command()
def replay(
    journal: Path = typer.Argument(..., exists=True, dir_okay=False, help="Journal JSONL d'une analyse"),
    entree: Path = typer.Argument(..., exists=True, dir_okay=False, help="Vidéo d'origine"),
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    out: Path = typer.Option(..., "--out"),
    debug_overlay: bool = typer.Option(False, "--debug-overlay"),
    decider: ChoixDecider = typer.Option(ChoixDecider.laya, "--decider"),
) -> None:
    """Rejoue une analyse avec d'autres réglages, sans refaire la perception."""
    if out.with_suffix(".jsonl").resolve() == journal.resolve():
        typer.echo("La sortie écraserait le journal source : choisis un autre --out.", err=True)
        raise typer.Exit(1)
    cache = lire_detections(journal)
    _executer(entree, out, config, debug_overlay, decider, lambda cfg: CachedPerception(cache))


@app.command()
def evaluer(
    donnees: Path = typer.Option(Path("data/decision"), "--donnees", exists=True, file_okay=False),
    predictions: Path = typer.Option(Path("data/predictions"), "--predictions"),
    approches: str = typer.Option("toujours-diffuser,regles,xgboost,laya,laya-ft,qwen-ft", "--approches"),
    config: Path = typer.Option(Path("configs/exemple.yaml"), "--config", exists=True, dir_okay=False),
    sortie: Path = typer.Option(Path("docs/resultats"), "--sortie"),
) -> None:
    """Compare les approches sur les jeux de test et écrit le rapport."""
    from irl_guard.donnees.lignes import lire_jeu
    from irl_guard.evaluation.metriques import evaluer_predictions
    from irl_guard.evaluation.rapport import JEUX_TEST, ecrire_rapport, obtenir_predictions

    cfg = load_config(config)
    jeux = {j: lire_jeu(donnees / f"{j}.jsonl") for j in JEUX_TEST}
    resultats = {}
    for approche in [a.strip() for a in approches.split(",") if a.strip()]:
        typer.echo(f"Évaluation : {approche}")
        par_jeu = obtenir_predictions(approche, cfg, donnees, predictions, jeux)
        resultats[approche] = {}
        for j, p in par_jeu.items():
            try:
                resultats[approche][j] = (evaluer_predictions(jeux[j], p, cfg.seuils, cfg.evaluation.couts)
                                          if p is not None else None)
            except (ValueError, KeyError, TypeError) as e:  # fichier partiel ou mal formé : approche absente
                typer.echo(f"  {approche} / {j} : prédictions inutilisables ({e}), marquée absente", err=True)
                resultats[approche][j] = None
    typer.echo(f"Rapport : {ecrire_rapport(resultats, sortie)}")
