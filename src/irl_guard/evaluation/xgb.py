"""Témoin tabulaire : XGBoost sur features structurées + TF-IDF de la charte (jamais les clauses)."""

from __future__ import annotations

import json
import re

from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from irl_guard.decision.base import ACTIONS, RISQUES
from irl_guard.decision.rules import CODE_POSTAL, EMAIL, MOTS_OFFENSANTS_BASE, NUMERO, TELEPHONE, VOIE, normaliser

INSULTE = re.compile(r"\b(" + "|".join(re.escape(m) for m in MOTS_OFFENSANTS_BASE) + r")\b")
OPTIONS = {"action": ACTIONS, "risque": RISQUES, "gravite": ("0", "1", "2", "3")}


def features_structurees(state: dict) -> dict[str, float]:
    cible = state["cible"]
    texte = normaliser(cible.get("texte") or "")
    return {
        f"type={cible['type']}": 1, f"classe={cible.get('classe') or 'aucune'}": 1,
        f"taille={cible.get('taille', 'petite')}": 1, f"categorie={state['categorie'].lower()}": 1,
        "domicile": int(bool(state["mode_domicile"])),
        "voie": int(bool(VOIE.search(texte))), "numero": int(bool(NUMERO.match(texte))),
        "code_postal": int(bool(CODE_POSTAL.search(texte))), "email": int(bool(EMAIL.search(texte))),
        "telephone": int(bool(TELEPHONE.search(texte))), "insulte": int(bool(INSULTE.search(texte))),
    }


class PredicteurXGBoost:
    def __init__(self, graine: int = 0):
        self.graine = graine
        self.dv = DictVectorizer()
        self.tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.modeles: dict[str, tuple[LabelEncoder, XGBClassifier | None]] = {}

    def _matrice(self, states: list[dict], entrainement: bool):
        structurees = [features_structurees(s) for s in states]
        chartes = [s["charte"] for s in states]
        if entrainement:
            return hstack([self.dv.fit_transform(structurees), self.tfidf.fit_transform(chartes)]).tocsr()
        return hstack([self.dv.transform(structurees), self.tfidf.transform(chartes)]).tocsr()

    def entrainer(self, lignes: list[dict]) -> PredicteurXGBoost:
        states = [json.loads(l["state"]) for l in lignes]
        golds = [json.loads(l["gold"]) for l in lignes]
        x = self._matrice(states, entrainement=True)
        for question in OPTIONS:
            encodeur = LabelEncoder().fit([g[question]["label"] for g in golds])
            y = encodeur.transform([g[question]["label"] for g in golds])
            if len(encodeur.classes_) < 2:
                self.modeles[question] = (encodeur, None)  # une seule classe : prédiction constante
                continue
            modele = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   random_state=self.graine, n_jobs=-1, verbosity=0)
            modele.fit(x, y)
            self.modeles[question] = (encodeur, modele)
        return self

    def predire(self, lignes: list[dict]) -> dict[str, dict]:
        x = self._matrice([json.loads(l["state"]) for l in lignes], entrainement=False)
        probas = {}
        for question, (encodeur, modele) in self.modeles.items():
            probas[question] = modele.predict_proba(x) if modele is not None else [[1.0]] * len(lignes)
        predictions = {}
        for i, ligne in enumerate(lignes):
            reponse = {}
            for question, options in OPTIONS.items():
                encodeur, _ = self.modeles[question]
                distribution = {o: 0.0 for o in options}
                for k, label in enumerate(encodeur.classes_):
                    distribution[str(label)] = float(probas[question][i][k])
                if question == "gravite":
                    reponse[question] = {"score": sum(int(o) * p for o, p in distribution.items()),
                                         "probabilities": distribution}
                else:
                    reponse[question] = {"probabilities": distribution}
            predictions[ligne["id"]] = reponse
        return predictions
