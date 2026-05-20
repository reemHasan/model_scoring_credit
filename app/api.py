from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd
import joblib
import shap
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Scoring model for Home Credit Risk", description="Predict loan default probability with SHAP explanations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model bundle
model_bundle = joblib.load('../ml/model/lgbm_bestmodel_fbeta10_bundle.pkl')
model = model_bundle['model']
features = model_bundle['feature_names']
best_threshold = model_bundle['threshold']
# Load client test data
client_data = pd.read_parquet('../data/prod_data/new_test_data_20features.parquet')
client_data = client_data[features]
# Load SHAP values for test data and expected value for SHAP explanations
shap_values_all = pd.read_parquet('../data/prod_data/shap_values.parquet')
expected_value = joblib.load('../data/prod_data/expected_value.pkl')


class LoanID(BaseModel):
    id: int

@app.get("/")
def root():
    """Informations about the API and how to use it."""
    return {
        "message": "Welcome to the HOME CREDIT Credit Scoring"+
        "/n app. Use /predict with client id to get the prediction for the client credit application and SHAP explanations.",
        "status": "running",
    }

@app.get("/health")
def health():
    try:
        if model is not None:
            return {"status": "ok", "model_loaded": True}
        else:
            return {"status": "error", "model_loaded": False}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        # return {"status": "error", "details": str(e)}

@app.post("/predict/{loan_id}")
def predict(loan_id: int):
    try:
        # Ensure client id exists in test data
        if (loan_id-1) >= client_data.shape[0]:            
            raise HTTPException(status_code=404, detail="Client id not in application database. Enter a whole number between 1 and 48745.")
        if ((loan_id-1) < 0):
            raise HTTPException(status_code=404, detail="Client id not in application database. Enter a whole number between 1 and 48745.")
        # Load current client data
        client_particulars = client_data.iloc[[loan_id-1]]
        # Predict decision of client credit application
        prediction = model.predict_proba(client_particulars)
        # prediction[0][0] is proba of class 0 (no default) and prediction[0][1] is proba of class 1 (default)
        proba = prediction[0][1] 
        if proba > best_threshold:
            proba_class = 'default'
            decision = "Reject loan application"
        else:
            proba_class = 'no default'
            decision = "Accept loan application"
        try:
            shap_values_client = shap_values_all.iloc[[loan_id-1]]
            return {
                'Client_id': loan_id,
                'Client default probability': proba, 
                'Class': proba_class,
                'Decision': decision,
                'Client_info': client_particulars.to_json(orient='records'),
                #'Expected Shap Value' : expected_value,
                'Shap_values_client' : shap_values_client.to_json(orient='records')
            }
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))