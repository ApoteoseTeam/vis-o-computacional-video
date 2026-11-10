
import cv2
import time

from detector import PersonDetector
from tracker import PersonTracker
from line_counter import LineCounter


VIDEO_PATH = "videos/pessoas.mp4"

# Configurações da zona de contagem
UPPER_LINE = 0.35
LOWER_LINE = 0.65


def main():

    # 1. Inicializar detector YOLO
    detector = PersonDetector(
        model_path="models/yolo11n_960.onnx",
        confidence=0.25
    )

    # 2. Inicializar ByteTrack
    tracker = PersonTracker()

    # 3. Inicializar contador
    line_counter = LineCounter(
        upper_position=UPPER_LINE,
        lower_position=LOWER_LINE
    )

    # 4. Abrir vídeo
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Não foi possível abrir: {VIDEO_PATH}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_video = cap.get(cv2.CAP_PROP_FPS)

    print(f"Resolução: {width}x{height}")
    print(f"FPS original: {fps_video}")

    print("=" * 40)
    print("PEOPLE COUNTER - YOLO + BYTETRACK")
    print("=" * 40)
    print("IN: Cima -> Baixo")
    print("OUT: Baixo -> Cima")
    print("Pressione Q para sair.")
    print("=" * 40)

    while True:

        success, frame = cap.read()

        if not success:
            break

        start_time = time.perf_counter()
        
        # ==================================
        # 1. DETECÇÃO YOLO
        # ==================================

        detections = detector.detect(frame)
        if not hasattr(main, "timings"):
            main.timings = []

        main.timings.append(detector.last_timings.copy())

        if len(main.timings) >= 100:

            for key in ["preprocess", "inference", "postprocess", "total"]:

                average = sum(
                    item[key] for item in main.timings
                ) / len(main.timings)

                print(f"{key}: {average:.2f} ms")

            main.timings.clear()

        yolo_time = time.perf_counter()
        # ==================================
        # 2. RASTREAMENTO BYTETRACK
        # ==================================

        tracked = tracker.update(detections)

        tracker_time = time.perf_counter()
        # ==================================
        # 3. CONTAGEM DE ENTRADAS E SAÍDAS
        # ==================================

        line_counter.update(
            tracked,
            frame.shape[0]
        )

        counter_time = time.perf_counter()
        # ==================================
        # 4. CONFIGURAÇÃO DAS LINHAS
        # ==================================

        height, width = frame.shape[:2]

        upper_line = int(height * UPPER_LINE)
        lower_line = int(height * LOWER_LINE)

        # Linha superior
        cv2.line(
            frame,
            (0, upper_line),
            (width, upper_line),
            (0, 255, 0),
            2
        )

        # Linha inferior
        cv2.line(
            frame,
            (0, lower_line),
            (width, lower_line),
            (0, 0, 255),
            2
        )

        # Identificação da zona
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
        # 5. DESENHAR PESSOAS RASTREADAS
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

                # Ignorar IDs inválidos
                if track_id < 0:
                    continue

                # Centro da pessoa
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Caixa delimitadora
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Ponto central
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    4,
                    (255, 0, 0),
                    -1
                )

                # ID da pessoa
                label = f"Pessoa #{track_id}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # ==================================
        # 6. CALCULAR FPS
        # ==================================

        elapsed = time.perf_counter() - start_time

        fps = (
            1 / elapsed
            if elapsed > 0
            else 0
        )

        # ==================================
        # 7. DASHBOARD
        # ==================================

        # Fundo do dashboard
        cv2.rectangle(
            frame,
            (10, 10),
            (310, 165),
            (30, 30, 30),
            -1
        )

        # Entradas
        cv2.putText(
            frame,
            f"ENTRADAS: {line_counter.entries}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # Saídas
        cv2.putText(
            frame,
            f"SAIDAS: {line_counter.exits}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

        # Pessoas presentes
        cv2.putText(
            frame,
            f"PRESENTES: {line_counter.people_inside}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # FPS
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
        # 8. EXIBIR VÍDEO
        # ==================================

        cv2.imshow(
            "People Counter - YOLO11 + ByteTrack",
            frame
        )

        # Sair pressionando Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ==================================
    # 9. FINALIZAR
    # ==================================

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 40)
    print("RESULTADO FINAL")
    print("=" * 40)

    print("Entradas:", line_counter.entries)
    print("Saídas:", line_counter.exits)

    print(
        "Pessoas presentes:",
        line_counter.people_inside
    )

    print(
    f"YOLO: {(yolo_time - start_time) * 1000:.2f} ms | "
    f"ByteTrack: {(tracker_time - yolo_time) * 1000:.2f} ms | "
    f"LineCounter: {(counter_time - tracker_time) * 1000:.2f} ms"
)
    
    print("=" * 40)


if __name__ == "__main__":
    main()
