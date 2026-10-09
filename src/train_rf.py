import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score, roc_curve

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "german_credit_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.joblib")

# Matrice de cout du German Credit Data : accorder un pret a un futur defaut
# coute 5 fois plus que refuser un bon payeur. Avec des probabilites calibrees,
# le seuil optimal est donc 1 / (1 + 5).
COST_FALSE_NEGATIVE = 5
COST_FALSE_POSITIVE = 1
DECISION_THRESHOLD = COST_FALSE_POSITIVE / (COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE)


def build_rf():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced"
    )


def average_cost(y_true, y_prob, threshold):
    y_pred = y_prob >= threshold
    false_neg = ((~y_pred) & (y_true == 1)).sum()
    false_pos = (y_pred & (y_true == 0)).sum()
    return (COST_FALSE_NEGATIVE * false_neg + COST_FALSE_POSITIVE * false_pos) / len(y_true)


def train_and_evaluate_rf(data_path):
    if not os.path.exists(data_path):
        print(f"Erreur : Fichier introuvable -> {data_path}")
        return

    df = pd.read_csv(data_path)
    print("Donnees chargees pour la modelisation.")

    target_column = "Credit Risk"

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Meme split que pour la regression logistique (random_state=42 pour comparer equitablement)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pas de standardisation necessaire pour Random Forest
    print("\nEntrainement du Random Forest...")
    model = build_rf()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Rapport de Classification (Random Forest) ---")
    print(classification_report(y_test, y_pred))

    auc_score = roc_auc_score(y_test, y_prob)
    print(f"Score AUC-ROC : {auc_score:.4f}")

    # Importance des variables (equivalent des coefficients en regression logistique)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n--- Top 10 variables les plus importantes ---")
    print(importances.head(10))

    # Courbe ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc_score:.2f})", color="green")
    plt.plot([0, 1], [0, 1], "k--", label="Hasard (AUC = 0.5)")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.title("Courbe ROC - Random Forest")
    plt.legend(loc="lower right")

    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
    roc_path = os.path.join(BASE_DIR, "reports", "roc_curve_rf.png")
    plt.savefig(roc_path)
    print(f"\nCourbe ROC sauvegardee dans : {roc_path}")

    # Calibration (Platt scaling) : class_weight="balanced" fait surestimer le risque.
    # Sigmoid plutot qu'isotonic, plus robuste sur un petit dataset (954 lignes).
    print("\nCalibration des probabilites (sigmoid, cv=5)...")
    calibrated = CalibratedClassifierCV(build_rf(), method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    y_prob_cal = calibrated.predict_proba(X_test)[:, 1]

    print("\n--- Calibration ---")
    print(f"Taux de defaut reel (test) : {y_test.mean():.3f}")
    print(f"{'':12s} {'p moyenne':>10s} {'Brier':>7s} {'AUC':>6s} {'Cout moyen':>11s}")
    print(f"{'Non calibre':12s} {y_prob.mean():10.3f} {brier_score_loss(y_test, y_prob):7.3f} "
          f"{auc_score:6.3f} {average_cost(y_test, y_prob, 0.5):11.3f}  (seuil 0.5)")
    print(f"{'Calibre':12s} {y_prob_cal.mean():10.3f} {brier_score_loss(y_test, y_prob_cal):7.3f} "
          f"{roc_auc_score(y_test, y_prob_cal):6.3f} {average_cost(y_test, y_prob_cal, DECISION_THRESHOLD):11.3f}"
          f"  (seuil {DECISION_THRESHOLD:.3f})")

    # Courbe de calibration
    fig, ax = plt.subplots(figsize=(8, 6))
    CalibrationDisplay.from_predictions(y_test, y_prob, n_bins=8, name="Non calibre", ax=ax)
    CalibrationDisplay.from_predictions(y_test, y_prob_cal, n_bins=8, name="Calibre (sigmoid)", ax=ax)
    ax.set_xlabel("Probabilite de defaut predite")
    ax.set_ylabel("Taux de defaut observe")
    ax.set_title("Courbe de calibration - Random Forest")
    calib_path = os.path.join(BASE_DIR, "reports", "calibration_curve_rf.png")
    fig.savefig(calib_path)
    print(f"\nCourbe de calibration sauvegardee dans : {calib_path}")

    # Sauvegarde du modele calibre + ordre des colonnes + seuil (utilises par l'API)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "model": calibrated,
            "feature_columns": X.columns.tolist(),
            "threshold": DECISION_THRESHOLD,
        },
        MODEL_PATH,
    )
    print(f"Modele sauvegarde dans : {MODEL_PATH}")


if __name__ == "__main__":
    train_and_evaluate_rf(FEATURE_DATA_PATH)

