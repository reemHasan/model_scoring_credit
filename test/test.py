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
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from pathlib import Path
import sys
sys.path.append(str(Path("..").resolve()))
from app.api import app
from app.state_store import set_state

# ── Fake data config ──────────────────────────────────────────────────────────
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

# ── Core fixture ──────────────────────────────────────────────────────────────
@pytest.fixture()
def client(request):
    """
    Patches:
      - app.api.init_db              → no real DB connection at startup
      - gradio.routes.mount_gradio_app → Gradio never starts
      - app.predict_service.log_prediction → no file writes
      - app.predict_service.store_record   → no DB calls

    log_prediction and store_record are patched in predict_service — that is where they are imported and called. 
    """
    proba = getattr(request, "param", {}).get("proba", 0.3)
    app.state.model           = make_fake_model(proba)
    app.state.features        = FEATURE_NAMES
    app.state.best_threshold  = THRESHOLD
    app.state.client_data     = make_fake_client_data()
    app.state.shap_values_all = make_fake_shap_values()
    app.state.expected_value  = 0.12
    set_state(app.state)
    # patch init_db in lifespan to avoid real DB connection, and patch mount_gradio_app to prevent Gradio from starting
    with patch("app.api.init_db", new_callable=AsyncMock) as mock_init_db, \
         patch("gradio.routes.mount_gradio_app", side_effect=lambda a, d, **kw: a), \
         patch("app.predict_service.log_prediction") as mock_log, \
         patch("app.predict_service.store_record", new_callable=AsyncMock) as mock_store:
        with TestClient(app, raise_server_exceptions=False) as c:
            c.mock_init_db = mock_init_db
            c.mock_log     = mock_log
            c.mock_store   = mock_store
            #print("fixture patched:", predict_service.log_prediction is mock_log)
            #print("fixture patched:", predict_service.store_record is mock_store)
            #print("fixture log id:", id(mock_log))
            #print("service log id:", id(predict_service.log_prediction))
            yield c
    for attr in ("model", "features", "best_threshold",
                 "client_data", "shap_values_all", "expected_value"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


# ══════════════════════════════════════════════════════════════════════════════
# 1. HEALTH ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════
class TestHealthEndpoint:

    def test_status_code_200(self, client):
        assert client.get("/health").status_code == 200

    def test_model_loaded_true(self, client):
        assert client.get("/health").json()["model_loaded"] is True

    def test_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_model_loaded_false_when_none(self):
        app.state.model           = None
        app.state.features        = FEATURE_NAMES
        app.state.best_threshold  = THRESHOLD
        app.state.client_data     = make_fake_client_data()
        app.state.shap_values_all = make_fake_shap_values()
        app.state.expected_value  = 0.12
        with patch("app.api.init_db", new_callable=AsyncMock), \
            patch("gradio.routes.mount_gradio_app", side_effect=lambda a, d, **kw: a), \
            patch("app.predict_service.log_prediction"), \
            patch("app.predict_service.store_record", new_callable=AsyncMock):
            with TestClient(app) as c:
                data = c.get("/health").json()
        assert data["model_loaded"] is False


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREDICT ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════
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

    # ── Input validation ──────────────────────────────────────────────────────
    def test_status_code_200_first_client(self, client):
        assert client.post("/predict/1").status_code == 200

    def test_status_code_200_last_client(self, client):
        assert client.post(f"/predict/{N_CLIENTS}").status_code == 200

    def test_client_id_in_response(self, client):
        assert client.post("/predict/1").json()["Client_id"] == 1

    def test_id_zero_returns_400(self, client):
        assert client.post("/predict/0").status_code == 400

    def test_id_negative_returns_400(self, client):
        assert client.post("/predict/-5").status_code == 400

    def test_id_above_max_returns_400(self, client):
        assert client.post(f"/predict/{N_CLIENTS + 1}").status_code == 400

    def test_string_id_returns_422(self, client):
        assert client.post("/predict/abc").status_code == 422

    def test_float_id_returns_422(self, client):
        assert client.post("/predict/1.5").status_code == 422

    # ── Response structure ────────────────────────────────────────────────────
    def test_all_required_keys_present(self, client):
        assert self.required_keys.issubset(client.post("/predict/1").json().keys())

    def test_content_type_is_json(self, client):
        assert "application/json" in client.post("/predict/1").headers["content-type"]

    def test_probability_between_0_and_1(self, client):
        proba = client.post("/predict/1").json()["Client default probability"]
        assert 0.0 <= proba <= 1.0

    def test_client_info_is_valid_json(self, client):
        parsed = json.loads(client.post("/predict/1").json()["Client_info"])
        assert isinstance(parsed, list) and len(parsed) == 1

    def test_shap_values_is_valid_json(self, client):
        parsed = json.loads(client.post("/predict/1").json()["Shap_values_client"])
        assert isinstance(parsed, list) and len(parsed) == 1

    def test_request_id_is_string(self, client):
        assert isinstance(client.post("/predict/1").json()["request_id"], str)

    def test_inference_ms_is_positive(self, client):
        assert client.post("/predict/1").json()["inference_ms"] >= 0

    def test_total_ms_is_positive(self, client):
        assert client.post("/predict/1").json()["total_ms"] >= 0

    def test_total_ms_gte_inference_ms(self, client):
        data = client.post("/predict/1").json()
        assert data["total_ms"] >= data["inference_ms"]

    # ── Prediction logic ──────────────────────────────────────────────────────
    @pytest.mark.parametrize("client", [{"proba": 0.8}], indirect=True)
    def test_high_proba_gives_default_class(self, client):
        data = client.post("/predict/1").json()
        assert data["Class"]    == "default"
        assert data["Decision"] == "Reject loan application"

    @pytest.mark.parametrize("client", [{"proba": 0.2}], indirect=True)
    def test_low_proba_gives_no_default_class(self, client):
        data = client.post("/predict/1").json()
        assert data["Class"]    == "no default"
        assert data["Decision"] == "Accept loan application"

    @pytest.mark.parametrize("client", [{"proba": 0.3}], indirect=True)
    def test_proba_at_threshold_is_accepted(self, client):
        assert client.post("/predict/1").json()["Class"] == "no default"

    @pytest.mark.parametrize("client", [{"proba": 0.8}], indirect=True)
    def test_class_and_decision_are_consistent(self, client):
        data = client.post("/predict/1").json()
        if data["Class"] == "default":
            assert data["Decision"] == "Reject loan application"
        else:
            assert data["Decision"] == "Accept loan application"


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOGGING TESTS — verify log_prediction() is called correctly
#    Patched in app.predict_service — that is where it is imported and called
# ══════════════════════════════════════════════════════════════════════════════
class TestLogging:
    
    def test_log_called_once_per_request(self, client):
        client.post("/predict/1")
        client.mock_log.assert_called_once()

    def test_log_status_200_on_success(self, client):
        client.post("/predict/1")
        _, kwargs = client.mock_log.call_args
        assert kwargs["status_code"] == 200

    def test_log_correct_loan_id(self, client):
        client.post("/predict/3")
        _, kwargs = client.mock_log.call_args
        assert kwargs["loan_id"] == 3

    def test_log_request_id_is_string(self, client):
        client.post("/predict/1")
        _, kwargs = client.mock_log.call_args
        assert isinstance(kwargs["request_id"], str)
        assert len(kwargs["request_id"]) > 0

    def test_log_proba_between_0_and_1(self, client):
        client.post("/predict/1")
        _, kwargs = client.mock_log.call_args
        assert 0.0 <= kwargs["proba_default"] <= 1.0

    def test_log_inference_ms_non_negative(self, client):
        client.post("/predict/1")
        _, kwargs = client.mock_log.call_args
        assert kwargs["inference_ms"] >= 0

    def test_log_total_ms_non_negative(self, client):
        client.post("/predict/1")
        _, kwargs = client.mock_log.call_args
        assert kwargs["total_ms"] >= 0

    def test_log_error_message_none_on_success(self, client):
        client.post("/predict/1")
        _, kwargs = client.mock_log.call_args
        assert kwargs["error_message"] is None

    def test_log_status_400_on_invalid_id(self, client):
        client.post("/predict/0")
        _, kwargs = client.mock_log.call_args
        assert kwargs["status_code"] == 400

    def test_log_error_message_set_on_invalid_id(self, client):
        client.post("/predict/0")
        _, kwargs = client.mock_log.call_args
        assert kwargs["error_message"] is not None
        assert len(kwargs["error_message"]) > 0

    def test_log_called_once_on_invalid_id(self, client):
        client.post("/predict/0")
        client.mock_log.assert_called_once()

    def test_multiple_requests_log_multiple_times(self, client):
        client.post("/predict/1")
        client.post("/predict/2")
        assert client.mock_log.call_count == 2


# ══════════════════════════════════════════════════════════════════════════════
# 4. DB STORE TESTS — verify store_record() is called correctly
#    Patched in app.predict_service — that is where it is imported and called
# ══════════════════════════════════════════════════════════════════════════════
class TestDBStore:

    @patch("app.predict_service.log_prediction") 
    @patch("app.predict_service.store_record", new_callable=AsyncMock)
    async  def test_correct_patch_log_store(self, mock_store, mock_log, client):
        print("id log_prediction in test:", id(mock_log))
        print("id store_record in test:", id(mock_store))
        client.post("/predict/1")
        print("log_prediction called:", mock_log.called)
        print("store_record awaited:", mock_store.await_count)
        assert mock_log.called
        assert mock_store.await_count == 1

    @patch("app.api.predict_and_log",new_callable=AsyncMock)
    def test_debug_predict_and_log(self, mock_predict_and_log, client):
        response = client.post("/predict/1")
        print(mock_predict_and_log.await_count)
        #mock_predict_and_log.side_effect = Exception("DEBUG")
        #with pytest.raises(Exception):
        #    await predict_and_log(1, app.state)

    def test_debug_all_mocks(self, client):
        response =client.post("/predict/1")
        print(response.status_code)
        print(response.text)
        print("\n── Mock status ──────────────────────────────")
        print(f"init_db    called: {client.mock_init_db.called}")
        print(f"log        called: {client.mock_log.called}")
        print(f"store      called: {client.mock_store.called}")
        print(f"store call_args  : {client.mock_store.call_args}")
        assert True

    def test_debug_invalid_id(self, client):
        response = client.post("/predict/0")
        print("\nStatus code  :", response.status_code)
        print("mock_store called:", client.mock_store.called)
        print("mock_store call_count:", client.mock_store.call_count)
        print("mock_store call_args:", client.mock_store.call_args)
        assert True

    def test_store_called_once_per_request(self, client):
        client.post("/predict/1")
        client.mock_store.assert_called_once()

    def test_store_correct_loan_id(self, client):
        client.post("/predict/3")
        _, kwargs = client.mock_store.call_args
        assert kwargs["loan_id"] == 3

    def test_store_status_200_on_success(self, client):
        client.post("/predict/1")
        _, kwargs = client.mock_store.call_args
        assert kwargs["status_code"] == 200

    def test_store_request_id_is_string(self, client):
        client.post("/predict/1")
        _, kwargs = client.mock_store.call_args
        assert isinstance(kwargs["request_id"], str)

    def test_store_proba_between_0_and_1(self, client):
        client.post("/predict/1")
        _, kwargs = client.mock_store.call_args
        assert 0.0 <= kwargs["proba_default"] <= 1.0

    def test_store_features_is_dict_with_all_features(self, client):
        client.post("/predict/1")
        _, kwargs = client.mock_store.call_args
        assert isinstance(kwargs["features"], dict)
        for feat in FEATURE_NAMES:
            assert feat in kwargs["features"]

    def test_store_shap_values_is_dict_with_all_features(self, client):
        client.post("/predict/1")
        _, kwargs = client.mock_store.call_args
        assert isinstance(kwargs["shap_values"], dict)
        for feat in FEATURE_NAMES:
            assert feat in kwargs["shap_values"]

    def test_store_error_message_none_on_success(self, client):
        client.post("/predict/1")
        _, kwargs = client.mock_store.call_args
        assert kwargs["error_message"] is None

    def test_store_status_400_on_invalid_id(self, client):
        client.post("/predict/0")
        _, kwargs = client.mock_store.call_args
        assert kwargs["status_code"] == 400

    def test_store_error_message_set_on_invalid_id(self, client):
        client.post("/predict/0")
        _, kwargs = client.mock_store.call_args
        assert kwargs["error_message"] is not None

    def test_store_features_none_on_invalid_id(self, client):
        client.post("/predict/0")
        _, kwargs = client.mock_store.call_args
        assert kwargs["features"] is None

    def test_store_called_once_on_invalid_id(self, client):
        client.post("/predict/0")
        client.mock_store.assert_called_once()

    def test_multiple_requests_store_multiple_times(self, client):
        client.post("/predict/1")
        client.post("/predict/2")
        assert client.mock_store.call_count == 2

    def test_request_id_same_in_log_and_store(self, client):
        """log_prediction and store_record must share the same request_id."""
        client.post("/predict/1")
        _, log_kwargs   = client.mock_log.call_args
        _, store_kwargs = client.mock_store.call_args
        assert log_kwargs["request_id"] == store_kwargs["request_id"]
    
    def test_init_db_called_at_startup(self, client):
        """init_db must be called once during lifespan startup."""
        client.mock_init_db.assert_called_once()

# ══════════════════════════════════════════════════════════════════════════════
# 5. PREDICT SERVICE — unit tests for run_prediction() directly
# ══════════════════════════════════════════════════════════════════════════════
class TestPredictor:

    @pytest.fixture()
    def fake_state(self):
        class FakeState:
            pass
        state                 = FakeState()
        state.model           = make_fake_model(proba=0.4)
        state.features        = FEATURE_NAMES
        state.best_threshold  = THRESHOLD
        state.client_data     = make_fake_client_data()
        state.shap_values_all = make_fake_shap_values()
        state.expected_value  = 0.12
        return state

    def test_valid_id_returns_dict(self, fake_state):
        from app.predict_service import run_prediction
        assert isinstance(run_prediction(1, fake_state), dict)

    def test_result_has_ml_keys(self, fake_state):
        """run_prediction returns ML keys only — no HTTP or logging keys."""
        from app.predict_service import run_prediction
        result = run_prediction(1, fake_state)
        for key in ("Client_id", "Client default probability",
                    "Class", "Decision", "Client_info",
                    "Expected_Shap_Value", "Shap_values_client",
                    "inference_ms"):
            assert key in result, f"Missing key: {key}"

    def test_no_request_id_in_run_prediction(self, fake_state):
        """request_id is set in predict_and_log — not in run_prediction."""
        from app.predict_service import run_prediction
        assert "request_id" not in run_prediction(1, fake_state)

    def test_no_total_ms_in_run_prediction(self, fake_state):
        """total_ms is measured in predict_and_log — not in run_prediction."""
        from app.predict_service import run_prediction
        assert "total_ms" not in run_prediction(1, fake_state)

    def test_invalid_id_raises_value_error(self, fake_state):
        from app.predict_service import run_prediction
        with pytest.raises(ValueError):
            run_prediction(0, fake_state)

    def test_id_above_max_raises_value_error(self, fake_state):
        from app.predict_service import run_prediction
        with pytest.raises(ValueError):
            run_prediction(N_CLIENTS + 1, fake_state)

    def test_client_id_matches_input(self, fake_state):
        from app.predict_service import run_prediction
        assert run_prediction(3, fake_state)["Client_id"] == 3

    def test_inference_ms_non_negative(self, fake_state):
        from app.predict_service import run_prediction
        assert run_prediction(1, fake_state)["inference_ms"] >= 0

