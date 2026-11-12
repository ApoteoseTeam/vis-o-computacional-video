
import cv2
import time

CAMERA_INDEX = 0

TESTES = [
    ("MJPG", 1280, 720),
    ("YUY2", 1280, 720),
    ("MJPG", 640, 480),
    ("YUY2", 640, 480),
]


def testar(formato, largura, altura):
    print(
        f"\nTestando {formato} - {largura}x{altura}",
        flush=True
    )

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        return

    try:
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*formato)
        )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
        cap.set(cv2.CAP_PROP_FPS, 30)

        codigo = int(cap.get(cv2.CAP_PROP_FOURCC))

        formato_real = "".join(
            chr((codigo >> (8 * i)) & 0xFF)
            for i in range(4)
        )

        print("Formato negociado:", formato_real)
        print(
            "Resolução negociada:",
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "x",
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

        print(
            "FPS informado:",
            cap.get(cv2.CAP_PROP_FPS)
        )

        # Aquecimento
        for _ in range(10):
            cap.read()

        inicio = time.perf_counter()
        frames = 0

        while time.perf_counter() - inicio < 5:
            sucesso, frame = cap.read()

            if not sucesso:
                print("Falha na captura.")
                break

            frames += 1

        duracao = time.perf_counter() - inicio

        print(
            f"FPS REAL: {frames / duracao:.2f}",
            flush=True
        )

    finally:
        cap.release()
        time.sleep(1)


for formato, largura, altura in TESTES:
    testar(formato, largura, altura)
