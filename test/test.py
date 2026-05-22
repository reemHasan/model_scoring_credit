"""
Pytest test suite for the Home Credit Scoring API.
Compatible with the lifespan + app.state pattern (FastAPI modern style).
Run:
    pytest test.py -v
    pytest test.py -v -s        # show print output
    pytest test.py -v -x        # stop on first failure
"""
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from pathlib import Path
import sys
# import app ONCE here — lifespan is bypassed by the fixture
sys.path.append(str(Path("..").resolve()))
from app.api import app

#Fake data config
N_CLIENTS     = 10
N_FEATURES    = 5
FEATURE_NAMES = ["feat_1", "feat_2", "feat_3", "feat_4", "feat_5"]
THRESHOLD     = 0.3

def make_fake_client_data() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame(
        np.random.rand(N_CLIENTS, N_FEATURES),
        columns=FEATURE_NAMES,
    )
def make_fake_shap_values() -> pd.DataFrame:
    np.random.seed(0)
    return pd.DataFrame(
        np.random.rand(N_CLIENTS, N_FEATURES),
        columns=FEATURE_NAMES,
    )
def make_fake_model(proba: float = 0.3) -> MagicMock:
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - proba, proba]])
    return model

# Core fixture — injects state directly, bypasses lifespan 
@pytest.fixture()
def client(request):
    
    proba = getattr(request, "param", {}).get("proba", 0.3)
    app.state.model           = make_fake_model(proba)
    app.state.features        = FEATURE_NAMES
    app.state.best_threshold  = THRESHOLD
    app.state.client_data     = make_fake_client_data()
    app.state.shap_values_all = make_fake_shap_values()
    app.state.expected_value  = 0.12

    # TestClient context manager handles startup/shutdown
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    # Teardown — reset state after each test
    for attr in ("model", "features", "best_threshold",
                 "client_data", "shap_values_all", "expected_value"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)

# 1. Test root endpoint
class TestRootEndpoint:
    def test_status_code_200(self, client):
        assert client.get("/").status_code == 200

    def test_status_is_running(self, client):
        assert client.get("/").json()["status"] == "running"

# 2. Test health endpoint
class TestHealthEndpoint:
    def test_status_code_200(self, client):
        assert client.get("/health").status_code == 200

    def test_model_loaded_true(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

# 3. Test Predict endpoint
class TestPredictEndpoint:

    required_keys = {
        "request_id",
        "Client_id",
        "Client default probability",
        "Class",
        "Decision",
        "inference_ms",
        "total_ms",
        "Client_info",
        "Expected_Shap_Value",
        "Shap_values_client",
    }
    # -------- Test input validation ---------
    def test_status_code_200_first_client(self, client):
        assert client.post("/predict/1").status_code == 200

    def test_status_code_200_last_client(self, client):
        assert client.post(f"/predict/{N_CLIENTS}").status_code == 200

    def test_client_id_in_response(self, client):
        assert client.post("/predict/1").json()["Client_id"] == 1
    
    def test_id_zero_returns_404(self, client):
        assert client.post("/predict/0").status_code == 400 # The 404 errors represent resources not existing, and in error 400, the resource exists, but the input is wrong
  
    def test_id_negative_returns_404(self, client):
        assert client.post("/predict/-5").status_code == 400

    def test_id_above_max_returns_404(self, client):
        assert client.post(f"/predict/{N_CLIENTS + 1}").status_code == 400

    def test_string_id_returns_422(self, client):
        assert client.post("/predict/abc").status_code == 422

    def test_float_id_returns_422(self, client):
        assert client.post("/predict/1.5").status_code == 422

    #--------- Test response structure and content ---------
    def test_probability_present(self, client):
        assert "Client default probability" in client.post("/predict/1").json()

    def test_probability_between_0_and_1(self, client):
        proba = client.post("/predict/1").json()["Client default probability"]
        assert 0.0 <= proba <= 1.0

    def test_class_is_valid_value(self, client):
        assert client.post("/predict/1").json()["Class"] in ("default", "no default")

    def test_decision_is_valid_value(self, client):
        decision = client.post("/predict/1").json()["Decision"]
        assert decision in ("Accept loan application", "Reject loan application")

    def test_client_info_is_valid_json(self, client):
        parsed = json.loads(client.post("/predict/1").json()["Client_info"])
        assert isinstance(parsed, list) and len(parsed) == 1

    def test_shap_values_is_valid_json(self, client):
        parsed = json.loads(client.post("/predict/1").json()["Shap_values_client"])
        assert isinstance(parsed, list) and len(parsed) == 1

    def test_client_info_has_all_features(self, client):
        info = json.loads(client.post("/predict/1").json()["Client_info"])[0]
        for feat in FEATURE_NAMES:
            assert feat in info

    def test_shap_values_has_all_features(self, client):
        shap = json.loads(client.post("/predict/1").json()["Shap_values_client"])[0]
        for feat in FEATURE_NAMES:
            assert feat in shap

    def test_all_required_keys_present(self, client):
        data = client.post("/predict/1").json()
        assert self.required_keys.issubset(data.keys())

    def test_content_type_is_json(self, client):
        response = client.post("/predict/1")
        assert "application/json" in response.headers["content-type"]

    #--------- Test Prediction logic with different proba values (mocked) ---------
    @pytest.mark.parametrize("client", [{"proba": 0.8}], indirect=True)
    def test_high_proba_gives_default_class(self, client):
        """proba=0.8 > threshold=0.3 → reject."""
        data = client.post("/predict/1").json()
        assert data["Class"]    == "default"
        assert data["Decision"] == "Reject loan application"

    @pytest.mark.parametrize("client", [{"proba": 0.2}], indirect=True)
    def test_low_proba_gives_no_default_class(self, client):
        """proba=0.2 < threshold=0.3 → accept."""
        data = client.post("/predict/1").json()
        assert data["Class"]    == "no default"
        assert data["Decision"] == "Accept loan application"

    @pytest.mark.parametrize("client", [{"proba": 0.3}], indirect=True)
    def test_proba_at_threshold_is_accepted(self, client):
        """proba == threshold → not strictly greater → accepted."""
        assert client.post("/predict/1").json()["Class"] == "no default"

    @pytest.mark.parametrize("client", [{"proba": 0.8}], indirect=True)
    def test_class_and_decision_are_consistent(self, client):
        data = client.post("/predict/1").json()
        if data["Class"] == "default":
            assert data["Decision"] == "Reject loan application"
        else:
            assert data["Decision"] == "Accept loan application"


    
