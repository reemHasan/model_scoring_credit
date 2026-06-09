# src/quantize_onnx.py
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
input_path  = BASE_DIR/"model/lgbm_model.onnx"
output_path = BASE_DIR/"model/lgbm_model_quantized.onnx"
quantize_dynamic(
    model_input=input_path,
    model_output=output_path,
    weight_type=QuantType.QUInt8,   # INT8 quantization
)
# Compare file sizes
orig_size  = Path(input_path).stat().st_size  / 1024
quant_size = Path(output_path).stat().st_size / 1024
print(f"Original  : {orig_size:.1f} KB")
print(f"Quantized : {quant_size:.1f} KB")
print(f"Reduction : {(1 - quant_size/orig_size):.1%}")