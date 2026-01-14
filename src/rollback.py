from __future__ import annotations

"""
Script utilitaire de gestion du Model Registry via MLflow.

Objectif principal :
- Lister les versions du modèle enregistré dans MLflow.
- Mettre à jour l'alias "production" pour activer :
  - une version spécifique (via target),
  - ou, par défaut, la version précédente (rollback).

Le registry local (metadata.json, current_model.txt) n'est plus utilisé.
MLflow devient la source de vérité.
"""

from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient


MODEL_NAME = "churn_model"
ALIAS = "production"


def _list_versions(client: MlflowClient) -> list[int]:
    """Retourne la liste des versions existantes (triées)."""
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    out = sorted({int(v.version) for v in versions})
    return out


def _get_current_version(client: MlflowClient) -> Optional[int]:
    """
    Retourne la version actuellement pointée par l'alias production.
    Retourne None si l'alias n'existe pas.
    """
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        return int(mv.version)
    except Exception:
        return None


def _set_alias(client: MlflowClient, version: int) -> None:
    """Fait pointer l'alias production vers la version demandée."""
    client.set_registered_model_alias(MODEL_NAME, ALIAS, str(version))


def main(target: Optional[str] = None) -> None:
    """
    Active une version spécifique ou effectue un rollback vers la version précédente.

    Paramètres
    ----------
    target : str, optionnel
        Version à activer explicitement (ex: "3").
        Si None, effectue un rollback vers la version précédente de l'alias production.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()

    versions = _list_versions(client)
    if not versions:
        raise FileNotFoundError(
            f"Aucune version MLflow trouvée pour le modèle '{MODEL_NAME}'. "
            "Lancer train.py au moins une fois."
        )

    # Activation explicite : target = numéro de version
    if target is not None:
        try:
            v = int(target)
        except ValueError as e:
            raise ValueError(
                "target doit être un numéro de version (ex: '2')."
            ) from e

        if v not in versions:
            raise ValueError(
                f"Version inconnue : v{v}. Versions disponibles : {versions}"
            )

        _set_alias(client, v)
        print(f"[OK] activation => {MODEL_NAME}@{ALIAS} = v{v}")
        return

    # Rollback automatique : version précédente par rapport à l'alias courant
    current = _get_current_version(client)

    # Si l'alias n'existe pas encore, on considère la dernière version comme courante
    if current is None:
        current = versions[-1]

    idx = versions.index(current)
    if idx == 0:
        raise ValueError(
            f"Rollback impossible : {MODEL_NAME}@{ALIAS} est déjà sur "
            f"la plus ancienne version (v{current})."
        )

    previous = versions[idx - 1]
    _set_alias(client, previous)
    print(
        f"[OK] rollback => {MODEL_NAME}@{ALIAS} : v{current} -> v{previous}"
    )


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)