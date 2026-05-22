from time import perf_counter
import uuid
from fastapi import FastAPI, HTTPException
# from typing import Dict, Any, List
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all assets (model, data, SHAP values) at startup and keep them in memory for fast API responses."""
    # Startup
    # Load model bundle
    if not hasattr(app.state, "model"):   # skip if already injected (tests)
        model_bundle = joblib.load("../ml/model/lgbm_bestmodel_fbeta10_bundle.pkl")
        app.state.model = model_bundle["model"]
        app.state.features = model_bundle["feature_names"]
        app.state.best_threshold = model_bundle["threshold"]
        # Load client test data
        client_data = pd.read_parquet("../data/prod_data/new_test_data_20features.parquet")
        print("Client data shape: ",client_data.shape)
        app.state.client_data = client_data[app.state.features]
        # Load SHAP values
        app.state.shap_values_all = pd.read_parquet("../data/prod_data/shap_values.parquet")
        app.state.expected_value = joblib.load("../data/prod_data/expected_value.pkl")
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
@app.get("/")
def root():
    """Informations about the API and how to use it."""
    return {
        "message": "Welcome to the HOME CREDIT Credit Scoring app. Use /predict with client id to get the prediction for the client credit application and SHAP explanations.",
        "status": "running",
    }

@app.get("/health")
def health():
    try:
        return {"status": "ok", "model_loaded": app.state.model is not None}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/predict/{loan_id}")
async def predict(loan_id: int):
    request_id  = str(uuid.uuid4())
    t_start     = perf_counter()
    try:
        # Ensure client id exists in test data
        if not (1 <= loan_id <= app.state.client_data.shape[0]):
            msg = f"Client id {loan_id} not in database. Enter 1–{app.state.client_data.shape[0]}."
            raise HTTPException(status_code=400, detail=msg)
        # Load current client data
        client_particulars = app.state.client_data.iloc[[loan_id-1]]

        # Predict decision of client credit application +++++++++++++++++++++++++
        
        # prediction[0][0] is proba of class 0 (no default) and prediction[0][1] is proba of class 1 (default)
        # Inference (timed separately)
        t_infer   = perf_counter() #The time.perf_counter() function returns a high-resolution timer value used to measure how long a piece of code takes to run. It is designed for performance measurement, includes time spent during sleep
        prediction = app.state.model.predict_proba(client_particulars)
        inference_ms = round((perf_counter() - t_infer) * 1000, 2)
        proba      = float(prediction[0][1])
        proba_class = "default"    if proba > app.state.best_threshold else "no default"
        decision    = "Reject loan application" if proba > app.state.best_threshold else "Accept loan application"
        
        # Get shap values for current client +++++++++++++++++++++++++++++++++++++++++++++
        try:
            
            shap_values_client = app.state.shap_values_all.iloc[[loan_id-1]]
            
            total_ms = round((perf_counter() - t_start) * 1000, 2)
            return {
                'request_id': request_id,
                'Client_id': loan_id,
                'Client default probability': proba, 
                'Class': proba_class,
                'Decision': decision,
                'inference_ms': inference_ms,
                'total_ms': total_ms,
                'Client_info': client_particulars.to_json(orient='records'),
                'Expected_Shap_Value' : float(app.state.expected_value),
                'Shap_values_client' : shap_values_client.to_json(orient='records')
            }
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))