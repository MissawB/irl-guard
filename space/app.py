"""Démo IRL Guard : comment chaque approche applique une charte à un élément vu à l'écran."""

import gradio as gr

try:
    import spaces
    gpu = spaces.GPU
except ImportError:  # exécution locale sans ZeroGPU
    def gpu(fonction):
        return fonction

from irl_guard.config import SessionConfig
from irl_guard.decision.laya_local import LayaLocalDecider
from irl_guard.decision.rules import RulesDecider

import logique

CFG = SessionConfig(charte="x")
REGLES = RulesDecider(CFG)
LAYA = LayaLocalDecider(CFG, logique.REPO_LAYA, revision=CFG.decideur.laya_revision)
LAYA._agent_charge()  # chargé au démarrage, sur CPU : pas pendant le créneau ZeroGPU d'un visiteur
QWEN = logique.charger_qwen(CFG, device="cuda")
SCORES = ("Sur 20 chartes **écrites à la main** (jamais vues à l'entraînement) : Qwen v2 seul 79,1 % de précision "
          "et 29 % de fuite ; combiné aux règles 71,9 % et 5,5 % de fuite. Laya v2 55,0 %, Laya brut 40,5 % — "
          "voir les fiches des modèles.")


@gpu
def decider(charte, categorie, mode_domicile, type_cible, texte, classe, taille):
    state = logique.construire_state(charte, categorie, mode_domicile, type_cible, texte, classe, taille)
    return logique.decider_tout(state, REGLES, LAYA, QWEN)


with gr.Blocks() as demo:
    gr.Markdown("# IRL Guard\nÉcris la charte de ta chaîne, décris ce que la caméra voit, et compare les décisions.")
    with gr.Row():
        with gr.Column():
            charte = gr.Textbox(label="Charte de la chaîne", lines=4, value=logique.CHARTE_DEFAUT)
            categorie = gr.Dropdown(["IRL", "Cuisine", "Gaming", "Musique", "Sport"], value="IRL", label="Catégorie")
            mode_domicile = gr.Checkbox(label="Mode domicile")
            type_cible = gr.Radio(["texte", "objet"], value="texte", label="Ce que la caméra voit")
            texte = gr.Textbox(label="Texte lu (si texte)", value="Rue Nationale")
            classe = gr.Dropdown(["couteau", "ciseaux", "telephone", "ordinateur"], value="couteau",
                                 label="Objet (si objet)")
            taille = gr.Radio(["petite", "moyenne", "grande"], value="grande", label="Taille à l'écran")
            bouton = gr.Button("Décider", variant="primary")
        with gr.Column():
            resultats = gr.Dataframe(headers=["Approche", "Action", "P(diffuser / flouter / pause)", "Risque",
                                              "Gravité"], interactive=False)
            gr.Markdown(SCORES)
    entrees = [charte, categorie, mode_domicile, type_cible, texte, classe, taille]
    gr.Examples(logique.EXEMPLES, entrees)
    bouton.click(decider, entrees, resultats)

demo.launch()
