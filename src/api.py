from __future__ import annotations

"""
API FastAPI de prédiction de churn pour le lab MLOps.

Ce service :
- charge dynamiquement le modèle courant depuis MLflow via l'alias 'production' ;
- expose des endpoints Kubernetes probes (/health, /startup, /ready) ;
- expose un endpoint `/predict` pour faire une prédiction de churn ;
- journalise chaque requête de prédiction dans `logs/predictions.log`.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ---------------------------------------------------------------------------
# Constantes de chemin
# ---------------------------------------------------------------------------

ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = ROOT / "models"
REGISTRY_DIR: Path = ROOT / "registry"
CURRENT_MODEL_PATH: Path = REGISTRY_DIR / "current_model.txt"
LOG_PATH: Path = ROOT / "logs" / "predictions.log"

# ---------------------------------------------------------------------------
# Constantes MLflow
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "churn_model"
ALIAS = "production"
MODEL_URI = f"models:/{MODEL_NAME}@{ALIAS}"

# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="MLOps Lab 01 - Churn API")

# ---------------------------------------------------------------------------
# Schéma d'entrée (Pydantic)
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Modèle de requête pour l'endpoint /predict.
    """

    tenure_months: int = Field(..., ge=0, le=200)
    num_complaints: int = Field(..., ge=0, le=50)
    avg_session_minutes: float = Field(..., ge=0.0, le=500.0)
    plan_type: str
    region: str
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Cache de modèle en mémoire
# ---------------------------------------------------------------------------

_model_cache: dict[str, Any] = {"name": None, "model": None}

# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def get_current_model_name() -> str:
    """
    Récupère le nom du modèle courant depuis MLflow via l'alias 'production'.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        return f"{MODEL_NAME}@{ALIAS} (v{mv.version})"
    except Exception as exc:
        raise FileNotFoundError(
            f"Impossible de récupérer le modèle avec l'alias '{ALIAS}'. "
            f"Assurez-vous qu'un modèle est activé via rollback.py. Erreur : {exc}"
        ) from exc


def load_model_if_needed() -> tuple[str, Any]:
    """
    Charge le modèle courant depuis MLflow en mémoire si nécessaire.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Cache key = model URI (alias), not local filename
    cache_key = MODEL_URI

    if _model_cache["name"] == cache_key and _model_cache["model"] is not None:
        return cache_key, _model_cache["model"]

    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
    except Exception as exc:
        raise FileNotFoundError(
            f"Impossible de charger le modèle depuis MLflow URI: {MODEL_URI}. "
            f"Erreur : {exc}"
        ) from exc

    _model_cache["name"] = cache_key
    _model_cache["model"] = model
    return cache_key, model


def log_prediction(payload: dict[str, Any]) -> None:
    """
    Ajoute une ligne de log JSON dans le fichier de prédictions.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Endpoints Kubernetes probes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    """
    Liveness probe.
    """
    try:
        model_name = get_current_model_name()
        return {"status": "ok", "current_model": model_name}
    except Exception as exc:  # pragma: no cover
        return {"status": "error", "detail": str(exc)}


@app.get("/startup")
def startup() -> dict[str, Any]:
    """
    Startup probe.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        
        # Vérifier que le modèle existe dans MLflow
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
            model_info = f"{MODEL_NAME}@{ALIAS} (v{mv.version})"
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Aucun modèle avec l'alias '{ALIAS}'. Exécutez rollback.py d'abord.",
            ) from exc

        return {
            "status": "ok",
            "current_model": model_info,
            "mlflow_uri": MLFLOW_TRACKING_URI,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Erreur lors de la vérification du modèle MLflow : {exc}",
        ) from exc


@app.get("/ready")
def ready() -> dict[str, Any]:
    """
    Readiness probe.
    """
    try:
        model_name = get_current_model_name()
        return {"status": "ready", "current_model": model_name}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

# ---------------------------------------------------------------------------
# Endpoint métier
# ---------------------------------------------------------------------------

@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    """
    Endpoint de prédiction de churn.
    """
    try:
        model_name, model = load_model_if_needed()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    features = {
        "tenure_months": req.tenure_months,
        "num_complaints": req.num_complaints,
        "avg_session_minutes": req.avg_session_minutes,
        "plan_type": req.plan_type.strip().lower(),
        "region": req.region.strip().upper(),
    }

    X_df = pd.DataFrame([features])

    start = time.perf_counter()
    try:
        proba = float(model.predict_proba(X_df)[0][1])
        pred = int(proba >= 0.5)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur de prédiction : {exc}",
        ) from exc

    latency_ms = (time.perf_counter() - start) * 1000.0

    out: dict[str, Any] = {
        "request_id": req.request_id,
        "model_version": model_name,
        "prediction": pred,
        "probability": round(proba, 6),
        "latency_ms": round(latency_ms, 3),
        "features": features,
        "ts": int(time.time()),
    }

    log_prediction(out)
    return out