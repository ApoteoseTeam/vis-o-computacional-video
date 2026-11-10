import onnxruntime as ort

print("Versão:", ort.__version__)

providers = ort.get_available_providers()

print("Provedores disponíveis:")

for provider in providers:
    print("-", provider)

if "DmlExecutionProvider" in providers:
    print("\nAMD GPU disponível via DirectML!")
else:
    print("\nDirectML não foi encontrado.")