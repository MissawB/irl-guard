import json

from irl_guard.analysis import Analyse
from irl_guard.journal import Journal, lire_detections
from irl_guard.types import Box, Detection


def test_aller_retour(tmp_path):
    chemin = tmp_path / "sous" / "j.jsonl"
    dets = [Detection("texte", 0.9, Box(1, 2, 3, 4), "Rue Nationale é")]
    with Journal(chemin) as j:
        j.ecrire(Analyse(0, (), 0.0, False), dets)
        j.ecrire(Analyse(5, (), 0.2, False, reutilisee=True), [])
    lignes = chemin.read_text(encoding="utf-8").splitlines()
    assert len(lignes) == 2
    assert json.loads(lignes[1])["analyse"]["reutilisee"] is True
    assert lire_detections(chemin) == {0: dets, 5: []}
