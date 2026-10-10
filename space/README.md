---
title: IRL Guard
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
license: apache-2.0
models:
  - MissawB/irl-guard-decision-qwen
  - MissawB/irl-guard-decision-laya
datasets:
  - MissawB/irl-guard-decisions
short_description: Décisions de modération guidées par une charte de live IRL
---

Démo d'IRL Guard : un *decision model* fine-tuné applique la charte d'une chaîne de live IRL à ce que la caméra
voit (texte lu ou objet détecté) et décide de diffuser, flouter ou mettre en pause. Comparaison des règles, de Laya
et de Qwen fine-tunés, seuls ou combinés avec les règles.
