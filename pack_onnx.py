# pack_onnx.py
import onnx
from pathlib import Path

SRC = Path("models/mrnet_abnormal_sagittal.onnx")
DST = Path("models/mrnet_abnormal_sagittal_packed.onnx")

if __name__ == "__main__":
    if not SRC.exists():
        raise FileNotFoundError(f"Missing: {SRC}")

    # Load with external data (reads the .onnx.data referenced by the graph)
    model = onnx.load_model(str(SRC), load_external_data=True)

    # Save everything embedded into ONE file
    onnx.save_model(model, str(DST), save_as_external_data=False)

    print("Packed ONNX written to:", DST)