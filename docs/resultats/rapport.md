# Résultats du decision model

Taux de fuite : part des cas à protéger (flouter ou pause attendu) que l'approche diffuse.

## Chartes vues

| Approche | n | Action | Risque | Gravité | Fuite | ECE | Coût moyen |
|---|---|---|---|---|---|---|---|
| regles | 1000 | 41.8 % | 83.0 % | 41.8 % | 22.1 % | 0.582 | 1.95 |
| xgboost | 1000 | 71.4 % | 85.2 % | 47.1 % | 53.7 % | 0.060 | 2.70 |
| laya | 1000 | 39.2 % | 25.2 % | 5.2 % | 37.4 % | 0.171 | 2.58 |
| laya-ft | absent | | | | | | |
| qwen-ft | absent | | | | | | |

![Fiabilité — Chartes vues](fiabilite_test_vues.png)

## Chartes non vues

| Approche | n | Action | Risque | Gravité | Fuite | ECE | Coût moyen |
|---|---|---|---|---|---|---|---|
| regles | 1000 | 40.7 % | 83.0 % | 40.7 % | 18.9 % | 0.593 | 2.01 |
| xgboost | 1000 | 69.5 % | 83.6 % | 43.4 % | 62.2 % | 0.070 | 3.22 |
| laya | 1000 | 44.7 % | 29.8 % | 5.2 % | 56.3 % | 0.128 | 3.35 |
| laya-ft | absent | | | | | | |
| qwen-ft | absent | | | | | | |

![Fiabilité — Chartes non vues](fiabilite_test_non_vues.png)
