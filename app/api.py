import logging
from pathlib import Path
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
# from logger import log_prediction
from app.predict_service import predict_and_log
from app.state_store import set_state
from app.database import init_db
from app.gui import demo
from gradio.routes import mount_gradio_app
# uncomment when profiling
#from app.profiling.profiler  import ProfilerMiddleware
# from fastapi.responses import RedirectResponse
from fastapi import BackgroundTasks
import onnxruntime as rt

BASE_DIR = Path(__file__).parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all assets (model, data, SHAP values) at startup and keep them in memory for fast API responses."""
    # Startup
    # Load model bundle
    #if not hasattr(app.state, "model"):   # skip if already injected (tests)
    if not hasattr(app.state, "client_data"):
        model_bundle = joblib.load(BASE_DIR / "ml/model/lgbm_bestmodel_fbeta10_bundle.pkl")
        app.state.model_name = "onnx"
        #app.state.model = model_bundle["model"]
        app.state.features = model_bundle["feature_names"]
        app.state.best_threshold = model_bundle["threshold"]
        # Load ONNX session — replaces bundle["model"]
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1

        app.state.onnx_session = rt.InferenceSession(
            str(BASE_DIR/"ml/model/lgbm_model_quantized.onnx"),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        # Load client test data
        client_data = pd.read_parquet(BASE_DIR / "data/prod_data/new_test_data_20Features.parquet")
        print("Client data shape: ",client_data.shape)
        app.state.client_data = client_data[app.state.features]
        # Load SHAP values
        app.state.shap_values_all = pd.read_parquet(BASE_DIR / "data/prod_data/shap_values.parquet")
        app.state.expected_value = joblib.load(BASE_DIR / "data/prod_data/expected_value.pkl")
        # Pre-warm — run one dummy prediction so first real request is fast
        dummy_X = app.state.client_data.iloc[[0]].values.astype(np.float32)
        # ONNX warm-up
        session    = app.state.onnx_session
        input_name = session.get_inputs()[0].name
        out_name   = session.get_outputs()[1].name
        session.run([out_name], {input_name: dummy_X})
        # store reference BEFORE Gradio wraps app
        set_state(app.state)               
        print("All assets loaded")
        # init_db is now async — await it directly in lifespan
    try:
        await init_db()
    except Exception as e:
        logging.getLogger("api").warning(f"DB init failed: {e} — continuing without DB")
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(lifespan=lifespan,title="Scoring model for Home Credit Risk",
               description="Predict loan default probability with SHAP explanations",
               version="1.0.0",license_info={"name": "MIT",},)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Enable when profiling
# app.add_middleware(ProfilerMiddleware)


@app.get("/health")
def health(request: Request):
    """Check if the API is running and if the model is loaded."""
    try:                      
        session = request.app.state.onnx_session if hasattr(request.app.state, "onnx_session") else None
        return {"status": "ok", "model_loaded": session is not None, "version": "1.0.0",
                "model name": app.state.model_name,
                "available_endpoints": {
                "gradio interface": "/",
                "predict": "/predict/{loan_id}",
                "docs": "/docs",},
                }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/{loan_id}")
async def predict(loan_id: int, request: Request, background_tasks: BackgroundTasks): # background tasks to be run after returning a response
    
    """Run prediction for a given loan_id and return the result. Logs the request and response details."""
    return await predict_and_log(loan_id, request.app.state,  background_tasks)

# Mount Gradio at "/" 
app = mount_gradio_app(app, demo, path="/")
# print(app.state)