# src/train.py
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error, r2_score

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from data_loader import load_data  # noqa: E402
from models import get_model  # noqa: E402

PROJECT_ROOT = current_dir.parent


def train(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "params.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    X_train, X_test, y_train, y_test = load_data(config_path)

    # Keep MLflow data at the project root, wherever the script is launched from.
    # Recent MLflow versions no longer allow the plain ./mlruns file store, so use SQLite.
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(config["experiment_name"]) is None:
        mlflow.create_experiment(
            config["experiment_name"],
            artifact_location=(PROJECT_ROOT / "mlartifacts").as_uri(),
        )
    mlflow.set_experiment(config["experiment_name"])

    best_model, best_r2, best_name = None, -999, None

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    for name, params in config["models"].items():
        # Ensure params is dict for logging
        safe_params = params.copy() if params else {}
        safe_params.pop("type", None)

        with mlflow.start_run(run_name=name):
            model = get_model(name, params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            mlflow.log_param("model_name", name)
            mlflow.log_params(safe_params)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            # Newer MLflow defaults to skops, which rejects tree models as "untrusted".
            # These models were trained locally, so cloudpickle (the classic format) is fine.
            mlflow.sklearn.log_model(model, name, serialization_format="cloudpickle")

            print(f"{name}: R2={r2:.4f}, RMSE={rmse:.4f}")

            if r2 > best_r2:
                best_model = model
                best_r2 = r2
                best_name = name

    best_model_path = models_dir / "best_model.pkl"
    joblib.dump(best_model, best_model_path)
    print(f"Best model: {best_name} (R2={best_r2:.3f}) saved to {best_model_path}")


if __name__ == "__main__":
    train()
