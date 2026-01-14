import mlflow
from mlflow.tracking import MlflowClient


MODEL_NAME = "churn_model"
ALIAS = "production"


mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()


# Cherche toutes les versions et prend la plus récente
mvs = client.search_model_versions(f"name='{MODEL_NAME}'")
if not mvs:
    raise SystemExit(f"Aucune version trouvée pour {MODEL_NAME}. Lance train.py d'abord.")


latest_version = max(int(mv.version) for mv in mvs)


client.set_registered_model_alias(MODEL_NAME, ALIAS, str(latest_version))
print(f"Modèle activé : {MODEL_NAME}@{ALIAS} -> v{latest_version}")