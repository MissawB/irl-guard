import ast
import re

import nbformat

from irl_guard.notebook import construire_notebook, ecrire_notebook


def _code(nb):
    return [c.source for c in nb.cells if c.cell_type == "code"]


def test_notebook_valide_et_parametre_en_tete(tmp_path):
    chemin = ecrire_notebook(tmp_path / "nb.ipynb")
    nb = nbformat.read(chemin, as_version=4)
    nbformat.validate(nb)
    assert 'MODELE = "qwen"' in _code(nb)[0]


def test_cellules_python_syntaxiquement_valides():
    for source in _code(construire_notebook()):
        lignes = [l for l in source.splitlines() if not l.lstrip().startswith(("!", "%"))]
        ast.parse("\n".join(lignes))


def test_aucun_token_en_clair():
    texte = "\n".join(c.source for c in construire_notebook().cells)
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", texte)
    assert 'userdata.get("HF_TOKEN")' in texte


def test_etapes_cles_presentes():
    texte = "\n".join(_code(construire_notebook()))
    for attendu in ("FastDecisionModel.from_pretrained", "get_peft_model", "build_dataset", "DecisionTrainer",
                    "resume_from_checkpoint", "calibrate", "save_pretrained_merged", "private=True",
                    "predictions_", "test_non_vues"):
        assert attendu in texte, attendu


def test_le_fichier_du_depot_est_a_jour():
    depot = nbformat.read("notebooks/finetune_decision.ipynb", as_version=4)
    assert [c.source for c in depot.cells] == [c.source for c in construire_notebook().cells]


def test_sauvegardes_isolees_par_modele():
    texte = "\n".join(_code(construire_notebook()))
    assert 'f"lora-{MODELE}"' in texte and 'f"merged-{MODELE}"' in texte
    assert "shutil.rmtree" in texte
    assert '"lora"' not in texte and '"merged"' not in texte
    assert 'config["model_type"]' in texte
