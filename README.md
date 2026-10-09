# credit-scoring-project
# Credit Scoring — Évaluation du Risque de Crédit

Projet de data science de bout en bout : prédiction du risque de défaut de paiement d'un emprunteur, à partir du dataset German Credit Data.

## Objectif

Développer un pipeline complet — nettoyage, feature engineering, modélisation, évaluation — pour prédire si un client va rembourser son crédit (bon payeur) ou faire défaut (mauvais payeur), en respectant les bonnes pratiques méthodologiques d'un projet de scoring réel.

## Dataset

**Source** : [German Credit Data (with Credit Risk)](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk), version enrichie du dataset original UCI/Statlog incluant la vraie variable cible.

- 954 clients, 10 colonnes (9 features + 1 cible)
- Variable cible `Credit Risk` : `0` = bon payeur, `1` = défaut (remappée depuis 1/2)
- Déséquilibre de classes réaliste : **~70% bons payeurs / 30% défauts**

| Variable | Type | Description |
|---|---|---|
| Age | Numérique | Âge du client |
| Sex | Catégorielle | Sexe |
| Job | Catégorielle (codée) | Niveau de qualification |
| Housing | Catégorielle | Type de logement (own / rent / free) |
| Saving accounts | Catégorielle | Niveau du compte épargne |
| Checking account | Catégorielle | Niveau du compte courant |
| Credit amount | Numérique | Montant du crédit demandé |
| Duration | Numérique | Durée du crédit (mois) |
| Purpose | Catégorielle | Motif du crédit |

## Pipeline du projet

```
data/raw/               → Dataset brut
      │
      ▼ preprocess.py    (nettoyage : suppression colonnes inutiles, gestion des NA)
data/processed/clean.csv
      │
      ▼ features.py      (feature engineering : ratio, remapping cible, encodage)
data/processed/features.csv
      │
      ▼ train_logistic.py / train_rf.py   (split, standardisation, entraînement, évaluation)
reports/                 (courbes ROC, métriques)
```

## Point méthodologique clé : détection et correction d'un data leakage

Un des enjeux principaux de ce projet a été la détection d'une **fuite de données (data leakage)** :

1. Le dataset Kaggle initialement utilisé ne contenait **aucune vraie colonne cible**. Une cible avait été fabriquée artificiellement à partir d'un seuil sur `Credit amount` et `Duration` — deux variables réutilisées ensuite comme features. Résultat : un AUC-ROC de **0.998**, bien au-delà de ce que rapporte la littérature sur ce dataset (~0.75-0.80), signe évident que le modèle « lisait » une formule plutôt qu'un vrai signal financier.
2. Une seconde fuite du même type a été trouvée dans le script d'entraînement lui-même (`Target_Class` dérivée de `Installment_Per_Month`).
3. Le dataset a été remplacé par une version contenant la vraie cible historique (`Credit Risk`), et les deux blocs fautifs ont été supprimés.
4. Un traitement plus rigoureux des valeurs manquantes a aussi été mis en place : `Saving accounts` et `Checking account` avaient ~55% de leurs `NA` supprimés par un `dropna()` initial, ce qui faisait perdre 46% des lignes du dataset et déformait le ratio de classes (70/30 → 55/45). Ces `NA` reflètent en réalité l'absence du type de compte chez le client, pas une donnée manquante — ils ont donc été remplacés par la catégorie explicite `"none"` plutôt que supprimés.

Après correction, le modèle de régression logistique obtient un AUC-ROC de **0.756**, cohérent avec les benchmarks publiés sur ce dataset.

## Analyse exploratoire — principaux enseignements

- **`Checking account = none`** est le signal le plus discriminant du dataset (corrélation de -0.33 avec le risque). Contre-intuitivement, l'absence de compte courant est associée à un taux de défaut plus faible (~12% contre ~30% en moyenne) — probablement le signe d'une situation financière stable ne nécessitant pas de gestion de trésorerie courante, plutôt qu'une exclusion bancaire.
- **`Duration`** et **`Credit amount`** sont positivement corrélés au risque : des crédits plus longs et plus élevés sont associés à davantage de défauts.
- **`Housing = own`** (propriétaire) est associé à un risque plus faible que la location.
- **`Age`** et **`Sex`** ont un pouvoir discriminant faible pris isolément.

## Modélisation

Deux modèles ont été entraînés et comparés sur le même split train/test (80/20, stratifié, `random_state=42`) :

| Modèle | AUC-ROC | Notes |
|---|---|---|
| Régression Logistique | 0.756 | `StandardScaler` (fit sur train uniquement) + `class_weight='balanced'` |
| Random Forest (tuné) | **0.770** | `n_estimators=200`, `max_depth=6`, `min_samples_leaf=10`, `class_weight='balanced'` |

Le Random Forest non contraint (profondeur illimitée) obtenait un AUC plus faible (0.74) que la régression logistique — un cas classique de surapprentissage sur un dataset de petite taille (954 lignes). Limiter la profondeur des arbres et la taille minimale des feuilles a permis de dépasser la régression logistique.

Les deux modèles convergent vers les mêmes variables les plus importantes (`Checking account_none`, `Duration`, `Credit amount`), ce qui renforce la fiabilité de ces signaux.

### Calibration et seuil de décision

Avec `class_weight='balanced'`, le Random Forest surestime le risque : probabilité de défaut moyenne de **44,6 %** pour un taux réel de **30,4 %**. Une calibration sigmoid (Platt scaling, `CalibratedClassifierCV`, cv=5) corrige ce biais sans toucher au classement des clients (l'AUC ne change pas). Sigmoid a été préféré à isotonic, plus sujet au surapprentissage sur 954 lignes.

Des probabilités fiables permettent de fixer le seuil à partir des coûts : selon la matrice de coût du German Credit Data, accorder un prêt à un futur défaut coûte 5 fois plus que refuser un bon payeur, d'où un seuil optimal de 1/(1+5) ≈ **0,167**.

| | p moyenne | Brier | AUC | Coût moyen / client |
|---|---|---|---|---|
| Non calibré, seuil 0,5 | 0,446 | 0,195 | 0,770 | 0,618 |
| Calibré, seuil 0,167 | **0,302** | **0,170** | 0,769 | **0,518** (−16 %) |

![Courbe de calibration](reports/calibration_curve_rf.png)

## Structure du projet

```
credit-scoring-project/
├── api/
│   └── main.py            # API FastAPI de scoring
├── data/
│   ├── raw/               # Dataset brut
│   └── processed/         # Données nettoyées et features
├── notebooks/
│   ├── 01_eda.ipynb       # Analyse exploratoire
│   └── 02_shap_explainability.ipynb  # Interprétabilité SHAP
├── src/
│   ├── preprocess.py      # Nettoyage (ETL)
│   ├── features.py        # Feature engineering
│   ├── train_logistic.py  # Entraînement régression logistique
│   └── train_rf.py        # Entraînement Random Forest
├── models/                 # Modèle entraîné (généré par train_rf.py)
├── reports/                # Courbes ROC générées
├── requirements.txt
└── README.md
```

## Installation et exécution

```bash
pip install -r requirements.txt

python src/preprocess.py
python src/features.py
python src/train_logistic.py
python src/train_rf.py     # sauvegarde aussi models/rf_model.joblib
```

## API de scoring

L'API FastAPI sert le Random Forest entraîné. Elle reçoit les données brutes d'un client et applique le même feature engineering que `features.py`.

```bash
uvicorn api.main:app --reload
```

Documentation interactive : http://127.0.0.1:8000/docs

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | État de l'API et du modèle |
| POST | `/predict` | Score d'un client |
| POST | `/predict/batch` | Score d'une liste de clients |

```bash
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"age": 35, "sex": "male", "job": 2, "housing": "own", "saving_accounts": "little", "checking_account": "moderate", "credit_amount": 3000, "duration": 24, "purpose": "car"}'
```

```json
{"probability_of_default": 0.3622, "prediction": "default", "decision": "reject", "threshold": 0.1667}
```

## Prochaines étapes
- **Suivi d'expériences** (MLflow ou équivalent léger)

## Stack technique

Python · Pandas · Scikit-learn · Matplotlib · Git
