import onnxruntime as ort
import cv2
import numpy as np

print("ONNX Runtime:", ort.__version__)
print("OpenCV:", cv2.__version__)
print()

providers = ort.get_available_providers()

print("Provedores disponíveis:")

for provider in providers:
    print(provider)

if "DmlExecutionProvider" in providers:
    print("\nDirectML encontrado!")
else:
    print("\nERRO: DirectML não encontrado!")