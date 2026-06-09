import json
import time
import uuid
import asyncio
from time import perf_counter
from app.logger import log_prediction
from app.database import store_record   # now both async
from fastapi import HTTPException
from fastapi import BackgroundTasks

def json_to_dict(json_str: str | None) -> dict | None:
    """Parse a JSON string that contains a list with one dict → return the dict."""
    if not json_str:
        return None
    try:
        parsed = json.loads(json_str)
        return parsed[0] if isinstance(parsed, list) else parsed
    except Exception:
        return None

def run_prediction(loan_id: int, app_state) -> dict:
    """
    Core prediction logic shared by FastAPI endpoint and Gradio UI.
    app_state is FastAPI's app.state (passed in explicitly).

    Returns a plain dict with all result fields.
    """
    n_clients = app_state.client_data.shape[0]
    # Ensure client id exists in test data
    if not (1 <= loan_id <= n_clients):
        raise ValueError(f"Client id not in application database. "
            f"Enter a whole number between 1 and {n_clients}.")
    # Load current client data
    client_particulars = app_state.client_data.iloc[[loan_id-1]]
    # Predict decision of client credit application 
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
    return {
        'Client_id': loan_id,
        'Client default '
        'probability': proba, 
        'Class': proba_class,
            'Decision': decision,
            'inference_ms': inference_ms,
            'Client_info': client_particulars.to_json(orient='records'),
            'Expected_Shap_Value' : float(app_state.expected_value),
            'Shap_values_client' : shap_values_client.to_json(orient='records')
            }

async def predict_and_log( loan_id: int, state, model_name, background_tasks: BackgroundTasks| None = None) -> dict:
    print("id log prediction:", id(log_prediction))
    request_id = str(uuid.uuid4()) # generate universally unique identifiers (UUIDs) for each request to track them in logs and DB
    t_start    = time.perf_counter()

    def _schedule(coro_func, **kwargs):
        """
        Schedule a coroutine as background work.
        - FastAPI context  → BackgroundTasks (tied to request lifecycle)
        - Gradio context   → asyncio.create_task (fire and forget)
        """
        if background_tasks is not None:
            # called from FastAPI endpoint — use BackgroundTasks
            background_tasks.add_task(coro_func, **kwargs)
        else:
            # called from Gradio — use asyncio task
            asyncio.create_task(coro_func(**kwargs))

    try:
        result = run_prediction(loan_id, state)
        inference_ms = result.get("inference_ms", 0)
        total_ms = round((time.perf_counter() - t_start) * 1000, 2)
        features    = json_to_dict(result.get("Client_info"))
        shap_values = json_to_dict(result.get("Shap_values_client"))
        # Step 1 — fast JSON log (before response)
        log_prediction(
            request_id=   request_id,
            loan_id=      loan_id,
            proba_default= result["Client default probability"],
            proba_class=  result["Class"],
            decision=     result["Decision"],
            inference_ms= inference_ms,
            total_ms=     total_ms,
            status_code=  200,
            client_features= features,
            shap_values= shap_values,
            model_runtime= model_name,
            error_message= None,
        )
        # slow — scheduled in background regardless of caller
        _schedule(
            store_record,
            request_id=request_id, loan_id=loan_id,
            proba_default=result["Client default probability"],
            proba_class=result["Class"], decision=result["Decision"],
            inference_ms=inference_ms, total_ms=total_ms,
            status_code=200, error_message=None,
            features=features, shap_values=shap_values,
            model_runtime=model_name
        )
    except ValueError as e:
        print("ENTERED VALUEERROR BLOCK")
        total_ms = round((time.perf_counter() - t_start) * 1000, 2)
        log_prediction(
            request_id=request_id, loan_id=loan_id,
            proba_default=None, proba_class=None, decision=None,
            inference_ms=0, total_ms=total_ms,
            status_code=400, error_message=str(e),
            client_features=None, shap_values=None, model_runtime="None" # 400 Bad Request is appropriate for invalid input
        )
        # just in case of error, write into db by ascync method as fastapi raise exception before background tasks finished
        await store_record(request_id=request_id, loan_id=loan_id,
            proba_default=None, proba_class=None, decision=None,
            inference_ms=0, total_ms=total_ms,
            status_code=400, error_message=str(e),
            features=None, shap_values=None, model_runtime="None")
        print("TASK SCHEDULED")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        total_ms = round((time.perf_counter() - t_start) * 1000, 2)
        log_prediction(
            request_id=request_id, loan_id=loan_id,
            proba_default=None, proba_class=None, decision=None,
            inference_ms=0, total_ms=total_ms,
            status_code=500, error_message=str(e),
            client_features=None, shap_values=None, model_runtime="None"
        )
        await store_record(
            request_id=request_id, loan_id=loan_id,
            proba_default=None, proba_class=None, decision=None,
            inference_ms=0, total_ms=total_ms,
            status_code=500, error_message=str(e),
            features=None, shap_values=None, model_runtime="None"
        )
        raise HTTPException(status_code=500, detail=str(e))
    return {**result, "request_id": request_id, "total_ms": total_ms}



    



