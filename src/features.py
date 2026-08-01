import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "german_credit_clean_updated.csv")
FINAL_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "german_credit_features.csv")

def engineer_features(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Erreur : Fichier introuvable -> {input_path}")
        return None
    
    df = pd.read_csv(input_path)
    
    # 1. Creation de la variable 'Installment_Per_Month'
    if 'Credit amount' in df.columns and 'Duration' in df.columns:
        df['Installment_Per_Month'] = df['Credit amount'] / df['Duration']
    
    # 2. Remapping de la vraie cible : 1 (bon payeur) -> 0, 2 (mauvais payeur/defaut) -> 1
    df['Credit Risk'] = df['Credit Risk'].map({1: 0, 2: 1})
    
    # 3. Encodage des variables categorielles (Credit Risk reste numerique, non affectee)
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # 4. Sauvegarde du fichier final
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_encoded.to_csv(output_path, index=False)
    
    print("Feature engineering reussi ! Fichier sauvegarde.")
    return df_encoded

if __name__ == "__main__":
    df_result = engineer_features(PROCESSED_DATA_PATH, FINAL_DATA_PATH)
    if df_result is not None:
        print(f"Nombre total de colonnes finales : {df_result.shape[1]}")
        print("Colonnes :", df_result.columns.tolist())
        print("\nDistribution de la cible :")
        print(df_result['Credit Risk'].value_counts())