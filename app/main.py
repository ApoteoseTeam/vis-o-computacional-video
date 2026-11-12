
import cv2
import time

from camera import USBCamera
from detector import PersonDetector
from tracker import PersonTracker
from line_counter import LineCounter


# ==================================
# CONFIGURAÇÕES
# ==================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

UPPER_LINE = 0.35
LOWER_LINE = 0.65

METRICS_INTERVAL = 100


def main():

    camera = None

    # ==================================
    # 1. INICIALIZAR YOLO
    # ==================================

    detector = PersonDetector(
        model_path="models/yolo11n_960.onnx",
        confidence=0.25
    )

    # ==================================
    # 2. INICIALIZAR BYTETRACK
    # ==================================

    tracker = PersonTracker()

    # ==================================
    # 3. INICIALIZAR CONTADOR
    # ==================================

    line_counter = LineCounter(
        upper_position=UPPER_LINE,
        lower_position=LOWER_LINE
    )

    # ==================================
    # 4. CONECTAR CÂMERA USB
    # ==================================

    camera = USBCamera(
        camera_index=CAMERA_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS
    )

    camera.open()

    print("\n" + "=" * 45)
    print("REALI - PEOPLE COUNTER")
    print("YOLO11 + ONNX + BYTETRACK")
    print("=" * 45)
    print("IN: Cima -> Baixo")
    print("OUT: Baixo -> Cima")
    print("Pressione Q para sair.")
    print("=" * 45)

    # ==================================
    # 5. MÉTRICAS
    # ==================================

    timings = []

    frame_count = 0
    fps = 0.0

    fps_start = time.perf_counter()
    fps_frames = 0

    try:

        while True:

            # Captura frame da webcam
            success, frame = camera.read()

            if not success:

                print("Falha ao capturar frame da câmera.")
                break

            start_time = time.perf_counter()

            # ==================================
            # 6. DETECÇÃO YOLO
            # ==================================

            detections = detector.detect(frame)

            # ==================================
            # 7. RASTREAMENTO BYTETRACK
            # ==================================

            tracked = tracker.update(detections)

            # ==================================
            # 8. CONTAGEM
            # ==================================

            line_counter.update(
                tracked,
                frame.shape[0]
            )

            # ==================================
            # 9. CONFIGURAÇÃO DAS LINHAS
            # ==================================

            height, width = frame.shape[:2]

            upper_line = int(height * UPPER_LINE)
            lower_line = int(height * LOWER_LINE)

            cv2.line(
                frame,
                (0, upper_line),
                (width, upper_line),
                (0, 255, 0),
                2
            )

            cv2.line(
                frame,
                (0, lower_line),
                (width, lower_line),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "ZONA DE CONTAGEM",
                (10, max(upper_line - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "LIMITE INFERIOR",
                (10, min(lower_line + 25, height - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

            # ==================================
            # 10. DESENHAR PESSOAS
            # ==================================

            if tracked.tracker_id is not None:

                for i in range(len(tracked)):

                    x1, y1, x2, y2 = map(
                        int,
                        tracked.xyxy[i]
                    )

                    track_id = int(
                        tracked.tracker_id[i]
                    )

                    if track_id < 0:
                        continue

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.circle(
                        frame,
                        (center_x, center_y),
                        4,
                        (255, 0, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        f"Pessoa #{track_id}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            # ==================================
            # 11. CALCULAR FPS
            # ==================================

            elapsed = time.perf_counter() - start_time

            fps_frames += 1
            frame_count += 1

            if elapsed > 0:
                fps = 1 / elapsed

            # Média de FPS em uma janela de 1 segundo
            current_time = time.perf_counter()

            if current_time - fps_start >= 1.0:

                fps = fps_frames / (current_time - fps_start)

                fps_frames = 0
                fps_start = current_time

            # ==================================
            # 12. ARMAZENAR MÉTRICAS
            # ==================================

            timings.append(
                detector.last_timings.copy()
            )

            if len(timings) >= METRICS_INTERVAL:

                print("\nMÉDIAS DE PROCESSAMENTO")

                for key in [
                    "preprocess",
                    "inference",
                    "postprocess",
                    "total"
                ]:

                    average = sum(
                        item[key] for item in timings
                    ) / len(timings)

                    print(f"{key}: {average:.2f} ms")

                timings.clear()

            # ==================================
            # 13. DASHBOARD
            # ==================================

            cv2.rectangle(
                frame,
                (10, 10),
                (310, 165),
                (30, 30, 30),
                -1
            )

            cv2.putText(
                frame,
                f"ENTRADAS: {line_counter.entries}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"SAIDAS: {line_counter.exits}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"PRESENTES: {line_counter.people_inside}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # ==================================
            # 14. EXIBIR CÂMERA
            # ==================================

            cv2.imshow(
                "Reali - Camera USB | YOLO11 + ByteTrack",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        # ==================================
        # 15. FINALIZAR
        # ==================================

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        print("\n" + "=" * 45)
        print("RESULTADO DA SESSÃO")
        print("=" * 45)

        print("Frames processados:", frame_count)
        print("Entradas:", line_counter.entries)
        print("Saídas:", line_counter.exits)
        print("Pessoas presentes:", line_counter.people_inside)

        print("=" * 45)


if __name__ == "__main__":
    main()
