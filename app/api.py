import logging
from pathlib import Path
from contextlib import asynccontextmanager
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
# from logger import log_prediction
from app.predict_service import predict_and_log
from app.state_store import set_state
from app.database import init_db
from app.gui import demo
from gradio.routes import mount_gradio_app
# from fastapi.responses import RedirectResponse


BASE_DIR = Path(__file__).parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all assets (model, data, SHAP values) at startup and keep them in memory for fast API responses."""
    # Startup
    # Load model bundle
    #if not hasattr(app.state, "model"):   # skip if already injected (tests)
    if not hasattr(app.state, "client_data"):
        model_bundle = joblib.load(BASE_DIR / "ml/model/lgbm_bestmodel_fbeta10_bundle.pkl")
        app.state.model = model_bundle["model"]
        app.state.features = model_bundle["feature_names"]
        app.state.best_threshold = model_bundle["threshold"]
        # Load client test data
        client_data = pd.read_parquet(BASE_DIR / "data/prod_data/new_test_data_20Features.parquet")
        print("Client data shape: ",client_data.shape)
        app.state.client_data = client_data[app.state.features]
        # Load SHAP values
        app.state.shap_values_all = pd.read_parquet(BASE_DIR / "data/prod_data/shap_values.parquet")
        app.state.expected_value = joblib.load(BASE_DIR / "data/prod_data/expected_value.pkl")
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
#@app.get("/")
#async def root_redirect():
#    return RedirectResponse(url="/gui")

@app.get("/health")
def health(request: Request):
    """Check if the API is running and if the model is loaded."""
    try:                      
        model = request.app.state.model if hasattr(request.app.state, "model") else None
        return {"status": "ok", "model_loaded": model is not None, "version": "1.0.0",
        "available_endpoints": {
        #    "gui": "/gui",
            "predict": "/predict/{loan_id}",
            "docs": "/docs",},
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/{loan_id}")
async def predict(loan_id: int, request: Request): # background tasks to be run after returning a response
    
    """Run prediction for a given loan_id and return the result. Logs the request and response details."""
    return await predict_and_log(loan_id, request.app.state)


# Mount Gradio at "/" 
app = mount_gradio_app(app, demo, path="/")
# print(app.state)