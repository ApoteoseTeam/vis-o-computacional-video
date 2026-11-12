
import cv2
import time

print("1. Iniciando teste...", flush=True)

print("2. Abrindo câmera DSHOW...", flush=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("3. VideoCapture retornou", flush=True)

if not cap.isOpened():
    print("ERRO: câmera não abriu", flush=True)
    raise SystemExit(1)

try:
    print("4. Configurando largura...", flush=True)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    print("5. Configurando altura...", flush=True)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("6. Configurando FPS...", flush=True)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("7. Configuração concluída!", flush=True)

    print(
        "Backend:",
        cap.getBackendName(),
        flush=True
    )

    print(
        "FPS informado:",
        cap.get(cv2.CAP_PROP_FPS),
        flush=True
    )

    print("8. Iniciando captura...", flush=True)

    frames = 0
    inicio = time.perf_counter()

    while time.perf_counter() - inicio < 5:
        success, frame = cap.read()

        if not success:
            print("Falha na leitura!", flush=True)
            break

        frames += 1

    tempo = time.perf_counter() - inicio

    print(
        f"FPS REAL: {frames / tempo:.2f}",
        flush=True
    )

finally:
    cap.release()
    print("Câmera encerrada.", flush=True)
