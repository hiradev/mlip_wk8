# MLIP_WK8 – Model Deployment

End-to-end house price pipeline (California housing): train several models, track them with MLflow, save the best one with joblib, and serve it with FastAPI and Streamlit.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Run everything from the project root.

```bash
python -m src.train                      # trains 3 models, logs to MLflow, saves models/best_model.pkl
mlflow ui --backend-store-uri sqlite:///mlflow.db   # compare runs at http://127.0.0.1:5000
uvicorn app.api:app --reload             # API at http://127.0.0.1:8000/docs
streamlit run app/streamlit_app.py       # UI
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"MedInc":8.3,"HouseAge":41,"AveRooms":6.9,"AveBedrms":1.0,"Population":322,"AveOccup":2.5,"Latitude":37.88,"Longitude":-122.23}'
```

## Layout

- `config/params.yaml` – experiment name, seed, test size, model hyperparameters
- `src/` – data loading, model factory, training (MLflow), evaluation, utils
- `app/` – FastAPI endpoint and Streamlit UI
- `models/` – saved `best_model.pkl` (git-ignored)
