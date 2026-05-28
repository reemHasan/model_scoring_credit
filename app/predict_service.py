import json
from time import perf_counter
import uuid
import pandas as pd

def run_prediction(loan_id: int, app_state) -> dict:
    """
    Core prediction logic shared by FastAPI endpoint and Gradio UI.
    app_state is FastAPI's app.state (passed in explicitly).

    Returns a plain dict with all result fields.
    """
    n_clients = app_state.client_data.shape[0]
    request_id  = str(uuid.uuid4())
    t_start     = perf_counter()
    # Ensure client id exists in test data
    if not (1 <= loan_id <= n_clients):
        raise ValueError(f"Client id not in application database. "
            f"Enter a whole number between 1 and {n_clients}.")
    # Load current client data
    client_particulars = app_state.client_data.iloc[[loan_id-1]]

    # Predict decision of client credit application +++++++++++++++++++++++++ 
    # prediction[0][0] is proba of class 0 (no default) and prediction[0][1] is proba of class 1 (default)
    # Inference (timed separately)
    t_infer   = perf_counter() #The time.perf_counter() function returns a high-resolution timer value used to measure how long a piece of code takes to run. It is designed for performance measurement, includes time spent during sleep
    prediction = app_state.model.predict_proba(client_particulars)
    inference_ms = round((perf_counter() - t_infer) * 1000, 2)
    proba      = float(prediction[0][1])
    proba_class = "default"    if proba > app_state.best_threshold else "no default"
    decision    = "Reject loan application" if proba > app_state.best_threshold else "Accept loan application"
    # Get shap values for current client +++++++++++++++++++++++++++++++++++++++++++++
    shap_values_client = app_state.shap_values_all.iloc[[loan_id-1]]
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
            'Expected_Shap_Value' : float(app_state.expected_value),
            'Shap_values_client' : shap_values_client.to_json(orient='records')
            }