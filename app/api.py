from fastapi import FastAPI, HTTPException, Request
# from typing import Dict, Any, List
import pandas as pd
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from predict_service import run_prediction
from gradio.routes import mount_gradio_app
from gui import demo
from state_store import set_state

BASE_DIR = Path(__file__).parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all assets (model, data, SHAP values) at startup and keep them in memory for fast API responses."""
    # Startup
    # Load model bundle
    #if not hasattr(app.state, "model"):   # skip if already injected (tests)
    if not hasattr(app.state, "client_data"):
        model_bundle = joblib.load(BASE_DIR / "ml" / "model" / "lgbm_bestmodel_fbeta10_bundle.pkl")
        app.state.model = model_bundle["model"]
        app.state.features = model_bundle["feature_names"]
        app.state.best_threshold = model_bundle["threshold"]
        # Load client test data
        client_data = pd.read_parquet(BASE_DIR / "data" / "prod_data" / "new_test_data_20Features.parquet")
        print("Client data shape: ",client_data.shape)
        app.state.client_data = client_data[app.state.features]
        # Load SHAP values
        app.state.shap_values_all = pd.read_parquet(BASE_DIR / "data" / "prod_data" / "shap_values.parquet")
        app.state.expected_value = joblib.load(BASE_DIR / "data" / "prod_data" / "expected_value.pkl")
        # store reference BEFORE Gradio wraps app
        set_state(app.state)               
        print("All assets loaded")
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(lifespan=lifespan,title="Scoring model for Home Credit Risk", description="Predict loan default probability with SHAP explanations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
#@app.get("/")
#def root():
#    """Informations about the API and how to use it."""
#    return {
#        "message": "Welcome to the HOME CREDIT Credit Scoring app. Use /predict with client id to get the prediction for the client credit application and SHAP explanations.",
#        "status": "running",
#    }


@app.get("/health")
def health(request: Request):
    try:                        # ← Request here too
        model = request.app.state.model if hasattr(request.app.state, "model") else None
        return {"status": "ok", "model_loaded": model is not None}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/{loan_id}")
async def predict(loan_id: int, request: Request):   # ← Request injected by FastAPI
    try:
        result = run_prediction(loan_id, request.app.state)  # ← pass request.app.state
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
# Mount Gradio at "/" 
app = mount_gradio_app(app, demo, path="/")
print(app.state)