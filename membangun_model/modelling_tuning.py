import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

def train_model_manual_logging():
#  KONFIGURASI DAGSHUB & MLFLOW 
    REPO_OWNER = "danendrapradnya" # Ganti dengan username Anda
    REPO_NAME = "my-first-repo" # Ganti dengan nama repo Anda
    
    # Inisialisasi Dagshub
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME)
    
    # Atur Tracking URI secara manual ke alamat Dagshub
    tracking_url = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(tracking_url)
    
    # Set nama eksperimen
    mlflow.set_experiment("Credit_Card_Fraud_Manual_Logging")

    # Loading Data
    preprocessed_path = "credit_card_fraud_processed.csv"
    if not os.path.exists(preprocessed_path):
        print("Data processed tidak ditemukan!")
        return

    df = pd.read_csv(preprocessed_path)
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    # Split Data 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Grid Search
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3, 
        scoring='f1', 
        n_jobs=-1
    )

    # Memulai MLflow
    with mlflow.start_run(run_name="RF_GridSearch_Manual_Log") as run:
        print(f"Eksperimen berjalan (Manual Log): {run.info.run_name}")
        
        # 1. Log Parameter Input (Sebelum Fit)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("sampling_method", "SMOTE")
        mlflow.log_param("grid_search_param_grid", str(param_grid))
        mlflow.log_param("cv_folds", 3)

        # Proses Fit
        grid_search.fit(X_train_res, y_train_res)
        best_model = grid_search.best_estimator_

        # Log Best Parameters dari GridSearch
        for param_name, param_value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{param_name}", param_value)

        # Prediksi untuk evaluasi
        y_pred = best_model.predict(X_test)

        # Log Metrik Standar (Sesuai autologging)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        # Log 2 Metrik Tambahan
        mcc = matthews_corrcoef(y_test, y_pred)
        b_acc = balanced_accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("mcc", mcc)
        mlflow.log_metric("balanced_accuracy", b_acc)

        # Manual Logging
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="model",
            registered_model_name="RF_Fraud_Model_Manual"
        )

        print("\nPelatihan selesai dan tercatat di Dagshub!")
        print(f"Best Params: {grid_search.best_params_}")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model_manual_logging()