
import cv2
import time

from collections import deque

from camera import USBCamera
from detector import PersonDetector
from tracker import PersonTracker
from line_counter import LineCounter


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

MODEL_PATH = "models/yolo11n_960.onnx"

UPPER_LINE = 0.35
LOWER_LINE = 0.65

WINDOW_NAME = "Reali Vision - Monitoramento"

# Cores OpenCV (BGR)
GREEN = (90, 210, 100)
RED = (80, 80, 240)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)
DARK = (35, 35, 35)


# ============================================================
# ZONA DE CONTAGEM
# ============================================================

def draw_counting_zone(frame):
    height, width = frame.shape[:2]

    upper_y = int(height * UPPER_LINE)
    lower_y = int(height * LOWER_LINE)

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, upper_y),
        (width, lower_y),
        ORANGE,
        -1
    )

    cv2.addWeighted(
        overlay,
        0.10,
        frame,
        0.90,
        0,
        frame
    )

    cv2.line(
        frame,
        (0, upper_y),
        (width, upper_y),
        ORANGE,
        2
    )

    cv2.line(
        frame,
        (0, lower_y),
        (width, lower_y),
        ORANGE,
        2
    )

    cv2.putText(
        frame,
        "ZONA DE CONTAGEM",
        (10, (upper_y + lower_y) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        ORANGE,
        2
    )


# ============================================================
# DESENHAR DETECÇÕES
# ============================================================

def draw_detections(frame, tracked):
    if tracked.tracker_id is None:
        return

    for i in range(len(tracked)):
        track_id = int(tracked.tracker_id[i])

        if track_id < 0:
            continue

        x1, y1, x2, y2 = map(
            int,
            tracked.xyxy[i]
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            GREEN,
            2
        )

        label = f"ID {track_id}"

        label_y = max(20, y1 - 8)

        cv2.putText(
            frame,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            WHITE,
            2
        )


# ============================================================
# DASHBOARD
# ============================================================

def draw_dashboard(
    frame,
    line_counter,
    camera_fps,
    processing_fps,
    detections_count
):
    height, width = frame.shape[:2]

    panel_height = 115

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, panel_height),
        DARK,
        -1
    )

    cv2.addWeighted(
        overlay,
        0.80,
        frame,
        0.20,
        0,
        frame
    )

    cv2.putText(
        frame,
        "REALI VISION",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        f"ENTRADAS: {line_counter.entries}",
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GREEN,
        2
    )

    cv2.putText(
        frame,
        f"SAIDAS: {line_counter.exits}",
        (210, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        RED,
        2
    )

    cv2.putText(
        frame,
        f"PRESENTES: {line_counter.people_inside}",
        (400, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        f"CAM: {camera_fps:.1f} FPS",
        (15, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        WHITE,
        1
    )

    cv2.putText(
        frame,
        f"IA: {processing_fps:.1f} FPS",
        (180, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        WHITE,
        1
    )

    cv2.putText(
        frame,
        f"DETECCOES: {detections_count}",
        (340, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        WHITE,
        1
    )

    cv2.putText(
        frame,
        "Pressione Q para sair",
        (15, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        WHITE,
        1
    )


# ============================================================
# APLICAÇÃO PRINCIPAL
# ============================================================

def main():

    print("\n======================================")
    print("REALI VISION")
    print("YOLO11 + BYTETRACK + OPENCV")
    print("======================================")

    camera = USBCamera(
        camera_index=CAMERA_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
        backend="DSHOW",
        use_mjpg=False
    )

    detector = PersonDetector(
        model_path=MODEL_PATH,
        confidence=0.25
    )

    tracker = PersonTracker()

    line_counter = LineCounter(
        upper_position=UPPER_LINE,
        lower_position=LOWER_LINE
    )

    processing_times = deque(maxlen=60)

    last_frame_id = -1
    processing_fps = 0.0

    processed_frames = 0
    dropped_frames = 0

    try:
        camera.start()

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_AUTOSIZE
        )

        while camera.is_running():

            camera_data = camera.read_latest()

            if camera_data is None:
                time.sleep(0.001)
                continue

            frame_id, frame, timestamp = camera_data

            if frame_id == last_frame_id:
                # Mantém a janela responsiva.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                time.sleep(0.001)
                continue

            if last_frame_id >= 0:
                dropped_frames += max(
                    0,
                    frame_id - last_frame_id - 1
                )

            last_frame_id = frame_id

            start_time = time.perf_counter()

            # ----------------------------------------
            # 1. YOLO
            # ----------------------------------------

            detections = detector.detect(frame)

            # ----------------------------------------
            # 2. BYTETRACK
            # ----------------------------------------

            # Compatível com o tracker.py atualizado
            # que aceita timestamp.
            tracked = tracker.update(
                detections,
                timestamp=timestamp
            )

            # ----------------------------------------
            # 3. CONTAGEM
            # ----------------------------------------

            line_counter.update(
                tracked,
                frame.shape[0]
            )

            # ----------------------------------------
            # 4. DESENHO
            # ----------------------------------------

            draw_counting_zone(frame)
            draw_detections(frame, tracked)

            # ----------------------------------------
            # 5. MÉTRICAS
            # ----------------------------------------

            elapsed = (
                time.perf_counter() - start_time
            )

            processing_times.append(elapsed)

            if processing_times:
                average_time = (
                    sum(processing_times)
                    / len(processing_times)
                )

                if average_time > 0:
                    processing_fps = (
                        1.0 / average_time
                    )

            camera_metrics = camera.get_metrics()

            # ----------------------------------------
            # 6. DASHBOARD
            # ----------------------------------------

            draw_dashboard(
                frame=frame,
                line_counter=line_counter,
                camera_fps=camera_metrics["fps"],
                processing_fps=processing_fps,
                detections_count=len(detections)
            )

            # ----------------------------------------
            # 7. EXIBIÇÃO
            # ----------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            processed_frames += 1

            if processed_frames % 150 == 0:
                print(
                    f"[REALI] "
                    f"Camera: {camera_metrics['fps']:.1f} | "
                    f"IA: {processing_fps:.1f} | "
                    f"IN: {line_counter.entries} | "
                    f"OUT: {line_counter.exits} | "
                    f"Descartados: {dropped_frames}"
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[REALI] Interrompido pelo usuário.")

    finally:
        camera.stop()
        cv2.destroyAllWindows()

        print("\n========== RESUMO ==========")
        print("Frames processados:", processed_frames)
        print("Frames descartados:", dropped_frames)
        print("Entradas:", line_counter.entries)
        print("Saídas:", line_counter.exits)
        print("Pessoas presentes:", line_counter.people_inside)
        print("============================")


if __name__ == "__main__":
    main()
