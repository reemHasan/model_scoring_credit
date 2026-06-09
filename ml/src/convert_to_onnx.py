# src/convert_to_onnx.py
import joblib
import numpy as np
from pathlib import Path
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

# Load your trained model bundle
BASE_DIR = Path(__file__).parent.parent
bundle    = joblib.load(BASE_DIR/"model/lgbm_bestmodel_fbeta10_bundle.pkl")
model     = bundle["model"]
features  = bundle["feature_names"]
threshold = bundle["threshold"]

n_features = len(features)

# Convert to ONNX (full precision float32)
initial_type = [("float_input", FloatTensorType([None, n_features]))]
onnx_model   = convert_lightgbm(
    model,
    initial_types=initial_type,
    target_opset=12,
)

# Save
#Path("ml/model/onnx").mkdir(parents=True, exist_ok=True)
with open(BASE_DIR/"model/lgbm_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ ONNX model saved — {n_features} features")