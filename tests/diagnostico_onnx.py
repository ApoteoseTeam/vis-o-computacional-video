
import onnxruntime as ort
import sys

print("Python:", sys.executable)
print("ONNX Runtime:", ort.__version__)
print("Localização:", ort.__file__)

print("\nProvedores disponíveis:")
providers = ort.get_available_providers()

for provider in providers:
    print("-", provider)

print("\nDirectML disponível:", "DmlExecutionProvider" in providers)
