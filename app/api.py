from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all assets (model, data, SHAP values) at startup and keep them in memory for fast API responses."""
    # Startup
    # Load model bundle
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

class LoanID(BaseModel):
    id: int

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
        if app.state.model is not None:
            return {"status": "ok", "model_loaded": True}
        else:
            return {"status": "error", "model_loaded": False}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        # return {"status": "error", "details": str(e)}

@app.post("/predict/{loan_id}")
async def predict(loan_id: int):
    try:
        # Ensure client id exists in test data
        if (loan_id-1) >= app.state.client_data.shape[0]:            
            raise HTTPException(status_code=404, detail="Client id not in application database. Enter a whole number between 1 and 48745.")
        if ((loan_id-1) < 0):
            raise HTTPException(status_code=404, detail="Client id not in application database. Enter a whole number between 1 and 48745.")
        # Load current client data
        client_particulars = app.state.client_data.iloc[[loan_id-1]]
        # Predict decision of client credit application
        prediction = app.state.model.predict_proba(client_particulars)
        # prediction[0][0] is proba of class 0 (no default) and prediction[0][1] is proba of class 1 (default)
        proba = prediction[0][1] 
        if proba > app.state.best_threshold:
            proba_class = 'default'
            decision = "Reject loan application"
        else:
            proba_class = 'no default'
            decision = "Accept loan application"
        try:
            shap_values_client = app.state.shap_values_all.iloc[[loan_id-1]]
            return {
                'Client_id': loan_id,
                'Client default probability': proba, 
                'Class': proba_class,
                'Decision': decision,
                'Client_info': client_particulars.to_json(orient='records'),
                'Expected_Shap_Value' : float(app.state.expected_value),
                'Shap_values_client' : shap_values_client.to_json(orient='records')
            }
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))