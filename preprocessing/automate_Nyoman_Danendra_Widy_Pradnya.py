import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def preprocess_to_csv(file_path, target_col='is_fraud', output_prefix='processed'):
    # Muat Dataset
    df = pd.read_csv(file_path)
    
    # Pembersihan Kolom
    # transaction_id dihapus, cardholder_age tetap disimpan karena akan diproses
    drop_cols = ['transaction_id', 'cardholder_age']
    df_clean = df.drop(columns=drop_cols)
    
    # Pisahkan fitur (X) dan target (y)
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # Definisikan Kolom
    numerical_cols = ['amount', 'transaction_hour', 'device_trust_score', 
                      'velocity_last_24h']
    categorical_cols = ['merchant_category']
    binary_cols = ['foreign_transaction', 'location_mismatch']
    
    # Buat Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Transformasi
    X_processed_array = preprocessor.fit_transform(X)
    
    # Rekonstruksi DataFrame
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numerical_cols + cat_names + binary_cols
    X_processed = pd.DataFrame(X_processed_array, columns=all_feature_names)
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Gabungkan kembali dengan Target untuk disimpan ke CSV
    train_final = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_final = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
    
    # Simpan ke CSV
    train_path = f"./preprocessing/{output_prefix}_train.csv"
    test_path = f"./preprocessing/{output_prefix}_test.csv"
    
    train_final.to_csv(train_path, index=False)
    test_final.to_csv(test_path, index=False)
    
    print(f"File berhasil disimpan: {train_path} dan {test_path}")

if __name__ == "__main__":
    preprocess_to_csv('credit_card_fraud_raw.csv')