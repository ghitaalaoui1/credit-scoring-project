import pandas as pd
import os

# Récupère le dossier racine du projet proprement
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "german_credit_clean.csv")
FINAL_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "german_credit_features.csv")

def engineer_features(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Erreur : Fichier introuvable -> {input_path}")
        return None
    
    df = pd.read_csv(input_path)
    
    if 'Credit amount' in df.columns and 'Duration' in df.columns:
        df['Installment_Per_Month'] = df['Credit amount'] / df['Duration']
    
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_encoded.to_csv(output_path, index=False)
    
    print(f"Feature engineering réussi ! Fichier sauvegardé.")
    return df_encoded

if __name__ == "__main__":
    df_result = engineer_features(PROCESSED_DATA_PATH, FINAL_DATA_PATH)
    if df_result is not None:
        print(f"Nombre total de colonnes finales : {df_result.shape[1]}")
        print("Colonnes :", df_result.columns.tolist())