import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_col='is_fraud', drop_cols=['transaction_id', 'cardholder_age']):  
  # Hapus kolom yang tidak diperlukan (seperti ID)
  df_clean = df.drop(columns=drop_cols)
  
  # Pisahkan fitur (X) dan target (y)
  X = df_clean.drop(columns=[target_col])
  y = df_clean[target_col]
  
  # Definisikan pengelompokan kolom berdasarkan eksperimen notebook
  numerical_cols = ['amount', 'transaction_hour', 'device_trust_score', 
                    'velocity_last_24h', 'cardholder_age']
  categorical_cols = ['merchant_category']
  # Kolom biner ditangani oleh 'remainder=passthrough' dalam ColumnTransformer
  binary_cols = ['foreign_transaction', 'location_mismatch']
  
  # Buat ColumnTransformer
  # - StandardScaler untuk data numerik
  # - OneHotEncoder untuk data kategori
  preprocessor = ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), numerical_cols),
          ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
      ],
      remainder='passthrough' # Untuk kolom biner
  )
  
  # Transformasi Fitur
  X_processed_array = preprocessor.fit_transform(X)
  
  # Rekonstruksi DataFrame (Opsional, agar tetap memiliki nama kolom)
  cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
  all_feature_names = numerical_cols + cat_feature_names + binary_cols
  X_processed = pd.DataFrame(X_processed_array, columns=all_feature_names)
  
  # 7. Split Data (Training & Testing)
  X_train, X_test, y_train, y_test = train_test_split(
      X_processed, y, test_size=0.2, random_state=42, stratify=y
  )
  
  return X_train, X_test, y_train, y_test, preprocessor