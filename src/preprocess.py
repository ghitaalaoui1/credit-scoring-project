import pandas as pd
import os

RAW_DATA_PATH = os.path.join("data", "raw", "german_credit_data.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "german_credit_clean.csv")

def preprocess_data(input_path, output_path):
    """Charge, nettoie et sauvegarde les données."""
    if not os.path.exists(input_path):
        print(f"Erreur : Fichier introuvable -> {input_path}")
        return
    
    df = pd.read_csv(input_path)
    
    # 1. Suppression d'une colonne inutile si elle existe (souvent une colonne d'index 'Unnamed: 0')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # 2. Gestion simple des valeurs manquantes (remplacement par la médiane pour les numériques, ou suppression)
    print(f"Valeurs manquantes avant nettoyage :\n{df.isnull().sum()}")
    df = df.dropna() # Ou imputation selon ton dataset
    
    # 3. Sauvegarde dans le dossier processed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nDonnées nettoyées et sauvegardées avec succès dans : {output_path}")
    print(f"Lignes finales : {df.shape[0]} | Colonnes finales : {df.shape[1]}")

if __name__ == "__main__":
    preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)