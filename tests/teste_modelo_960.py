
import onnxruntime as ort
import numpy as np
import time

model_path = "models/yolo11n_960.onnx"

options = ort.SessionOptions()
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.enable_mem_pattern = False

session = ort.InferenceSession(
    model_path,
    sess_options=options,
    providers=[
        "DmlExecutionProvider",
        "CPUExecutionProvider"
    ]
)

input_info = session.get_inputs()[0]

print("Modelo carregado!")
print("Provedores:", session.get_providers())
print("Entrada:", input_info.shape)

# Teste de inferência
image = np.zeros((1, 3, 960, 960), dtype=np.float32)

try:
    start = time.perf_counter()

    output = session.run(
        None,
        {input_info.name: image}
    )

    elapsed = (time.perf_counter() - start) * 1000

    print("Inferência executada!")
    print("Formato da saída:", output[0].shape)
    print(f"Tempo de inferência: {elapsed:.2f} ms")

except Exception as error:
    print("Erro na inferência:", error)
