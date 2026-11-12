
import cv2
import time

RESOLUTIONS = [
    (640, 480),
    (800, 600),
    (960, 540),
    (1024, 576),
    (1280, 720),
]

for width, height in RESOLUTIONS:

    print(f"\nTestando {width}x{height}")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Erro ao abrir câmera")
        continue

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        actual_height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        print(
            f"Resolução real: "
            f"{actual_width}x{actual_height}"
        )

        # Aquecimento
        for _ in range(10):
            success, _ = cap.read()
            if not success:
                break

        start = time.perf_counter()
        frames = 0

        while time.perf_counter() - start < 5:

            success, frame = cap.read()

            if not success:
                break

            frames += 1

        elapsed = time.perf_counter() - start

        print(
            f"FPS real: {frames / elapsed:.2f}"
        )

    finally:
        cap.release()
        time.sleep(1)
