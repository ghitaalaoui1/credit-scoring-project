import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "german_credit_features.csv"
)


def train_and_evaluate_model(data_path):
  if not os.path.exists(data_path):
    print(f"Erreur : Fichier introuvable -> {data_path}")
    return

  df = pd.read_csv(data_path)
  print("Donnees chargees pour la modelisation.")

  target_column = "Credit Risk"

  # 1. Separation des features (X) et de la cible (y)
  X = df.drop(columns=[target_column])
  y = df[target_column]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # 2. Standardisation (fit sur train seulement, transform sur train et test)
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # 3. Modelisation avec Regression Logistique
  print("\nEntrainement de la Regression Logistique...")
  model = LogisticRegression(
      max_iter=1000, random_state=42, class_weight="balanced"
  )
  model.fit(X_train_scaled, y_train)

  # 4. Predictions et Evaluation
  y_pred = model.predict(X_test_scaled)
  y_prob = model.predict_proba(X_test_scaled)[:, 1]

  print("\n--- Rapport de Classification ---")
  print(classification_report(y_test, y_pred))

  # 5. Calcul du score AUC-ROC
  auc_score = roc_auc_score(y_test, y_prob)
  print(f"Score AUC-ROC : {auc_score:.4f}")

  # 6. Sauvegarde de la courbe ROC
  fpr, tpr, thresholds = roc_curve(y_test, y_prob)
  plt.figure(figsize=(8, 6))
  plt.plot(
      fpr,
      tpr,
      label=f"Regression Logistique (AUC = {auc_score:.2f})",
      color="blue",
  )
  plt.plot([0, 1], [0, 1], "k--", label="Hasard (AUC = 0.5)")
  plt.xlabel("Taux de Faux Positifs")
  plt.ylabel("Taux de Vrais Positifs")
  plt.title("Courbe ROC - Modele de Scoring")
  plt.legend(loc="lower right")

  os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
  roc_path = os.path.join(BASE_DIR, "reports", "roc_curve.png")
  plt.savefig(roc_path)
  print(f"\nCourbe ROC sauvegardee dans : {roc_path}")


if __name__ == "__main__":
  train_and_evaluate_model(FEATURE_DATA_PATH)