from __future__ import annotations

"""
Module d'entraînement et d'enregistrement d'un modèle de churn.
"""
from mlflow.tracking import MlflowClient

from datetime import datetime
from pathlib import Path
from typing import Any, Final
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import mlflow
import mlflow.sklearn

# ---------------------------------------------------------------------------
# Chemins et constantes globales
# ---------------------------------------------------------------------------
ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_PATH: Final[Path] = ROOT / "data" / "processed.csv"
MODELS_DIR: Final[Path] = ROOT / "models"
REGISTRY_DIR: Final[Path] = ROOT / "registry"
CURRENT_MODEL_PATH: Final[Path] = REGISTRY_DIR / "current_model.txt"
METADATA_PATH: Final[Path] = REGISTRY_DIR / "metadata.json"
MODEL_NAME: Final[str] = "churn_model"


# ---------------------------------------------------------------------------
# Fonctions pour la gestion des métadonnées
# ---------------------------------------------------------------------------

def load_metadata() -> list[dict[str, Any]]:
    if not METADATA_PATH.exists():
        return []
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)

def save_metadata(items: list[dict[str, Any]]) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(items, file, indent=2)

# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def compute_baseline_f1(y_true: pd.Series | list[int]) -> float:
    y_pred = [0] * len(y_true)
    return float(f1_score(y_true, y_pred, zero_division=0))

def build_preprocessing_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])
    return preprocessor

def build_model_pipeline(preprocessor: ColumnTransformer, seed: int) -> Pipeline:
    classifier = LogisticRegression(max_iter=400, random_state=seed)
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", classifier),
    ])
    return pipe

# ---------------------------------------------------------------------------
# Fonction principale d'entraînement
# ---------------------------------------------------------------------------

def main(version: str = "v1", seed: int = 42, gate_f1: float = 0.6) -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Fichier processed.csv introuvable. "
            "Veuillez exécuter d'abord le script de préparation des données."
        )

    df = pd.read_csv(DATA_PATH)
    target_col = "churn"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    numeric_cols = ["tenure_months", "num_complaints", "avg_session_minutes"]
    categorical_cols = ["plan_type", "region"]

    # Construction du pipeline complet
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    pipe = build_model_pipeline(preprocessor, seed)

    # Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # Entraînement
    pipe.fit(X_train, y_train)

    # Prédictions sur le test
    y_pred = pipe.predict(X_test)

    # Calcul des métriques
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "baseline_f1": compute_baseline_f1(y_test),
    }

    # Sauvegarde du modèle timestampé
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_filename = f"churn_model_{version}_{timestamp}.joblib"
    model_path = MODELS_DIR / model_filename
    joblib.dump(pipe, model_path)

    # Création de l'entrée de métadonnées
    entry: dict[str, Any] = {
        "model_file": model_filename,
        "version": version,
        "trained_at_utc": timestamp,
        "data_file": DATA_PATH.name,
        "seed": seed,
        "metrics": metrics,
        "gate_f1": gate_f1,
        "passed_gate": metrics["f1"] >= gate_f1 and metrics["f1"] >= metrics["baseline_f1"],
    }

    # Mise à jour des métadonnées
    items = load_metadata()
    items.append(entry)
    save_metadata(items)

    # Logs
    print("[METRICS]", json.dumps(metrics, indent=2))
    print(f"[OK] Modèle sauvegardé : {model_path}")

    # Déploiement du modèle (registry minimaliste + stable alias pour DVC)
    if entry["passed_gate"]:
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        CURRENT_MODEL_PATH.write_text(model_filename, encoding="utf-8")

        # Alias stable
        stable_model_path = MODELS_DIR / "model.joblib"
        joblib.dump(pipe, stable_model_path)

        # Associe le run à une expérience logique. Les exécutions seront regroupées sous ce nom dans l'interface MLflow.
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("mlops-lab-01")

        # Crée une exécution traçable correspondant à l'entraînement courant. Le nom du run porte ici la version logique passée au script.
        with mlflow.start_run(run_name=f"train-{version}") as run:
            run_id = run.info.run_id
            
            # Enregistre le contexte de l'entraînement. Les paramètres sont essentiels pour comprendre et reproduire une exécution.
            mlflow.log_param("version", version)
            mlflow.log_param("seed", seed)
            mlflow.log_param("gate_f1", gate_f1)
            
            # Enregistre l'ensemble des métriques calculées par le script (F1, accuracy, etc.). 
            # Chaque métrique devient comparable entre runs dans l'UI.
            mlflow.log_metrics(metrics)
            
            # Ajoute des informations descriptives (métadonnées) destinées à faciliter la lecture humaine 
            # quel fichier de données a été utilisé, quel fichier modèle a été produit.
            mlflow.set_tag("data_file", DATA_PATH.name)
            mlflow.set_tag("model_file", model_filename)
            
            # Attache le fichier modèle exporté (celui déjà généré via joblib.dump) au run MLflow. Il apparaît dans la section Artifacts du run.
            mlflow.log_artifact(str(model_path), artifact_path="exported_models")
            
            # Enregistre le pipeline entraîné comme un modèle MLflow et le publie dans le Model Registry sous un nom stable (churn_model). 
            # Chaque exécution crée une nouvelle version (v1, v2, …) associée au run.
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )
            
            # Ajoute des tags sur la version du modèle dans le Model Registry
            client = MlflowClient()
            mvs = client.search_model_versions(f"name='{MODEL_NAME}' and run_id='{run_id}'")
            if mvs:
                v = mvs[0].version
                client.set_model_version_tag(MODEL_NAME, v, "data_file", DATA_PATH.name)
                client.set_model_version_tag(MODEL_NAME, v, "model_file", model_filename)
                client.set_model_version_tag(MODEL_NAME, v, "gate_f1", str(gate_f1))

        print(f"[DEPLOY] Modèle activé : {model_filename}")
        print(f"[DEPLOY] Alias stable : {stable_model_path}")
        print(f"[MLFLOW] Run ID : {run_id}")
        print(f"[MLFLOW] Modèle enregistré : {MODEL_NAME}")
    else:
        print("[DEPLOY] Refusé : F1 insuffisante ou baseline non battue.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(version=sys.argv[1])
    else:
        main()