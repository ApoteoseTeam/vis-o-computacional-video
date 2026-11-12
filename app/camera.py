
import cv2
import threading
import time


class USBCamera:
    def __init__(
        self,
        camera_index=0,
        width=640,
        height=480,
        fps=30,
        backend="DSHOW",
        use_mjpg=False
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self.use_mjpg = use_mjpg

        self.cap = None
        self.thread = None

        self.running = threading.Event()
        self.lock = threading.Lock()

        self.latest_frame = None
        self.latest_timestamp = None
        self.frame_id = 0

        # Métricas de captura
        self.capture_count = 0
        self.capture_start = 0.0
        self.read_times = []

        self.capture_fps = 0.0
        self.average_read_ms = 0.0

    def start(self):
        if self.running.is_set():
            return self

        backends = {
            "DSHOW": cv2.CAP_DSHOW,
            "MSMF": cv2.CAP_MSMF,
            "AUTO": cv2.CAP_ANY
        }

        selected_backend = backends.get(
            self.backend.upper(),
            cv2.CAP_DSHOW
        )

        self.cap = cv2.VideoCapture(
            self.camera_index,
            selected_backend
        )

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise RuntimeError(
                "Não foi possível abrir a webcam."
            )

        if self.use_mjpg:
            self.cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG")
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width
        )
        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height
        )
        self.cap.set(
            cv2.CAP_PROP_FPS,
            self.fps
        )

        # Alguns drivers ignoram essa configuração.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        fourcc = int(
            self.cap.get(cv2.CAP_PROP_FOURCC)
        )

        codec = "".join(
            chr((fourcc >> (8 * i)) & 0xFF)
            for i in range(4)
        )

        print("\n========== REALI CAMERA ==========")
        print("Backend:", self.cap.getBackendName())
        print("Formato:", codec)
        print(
            "Resolução:",
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "x",
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        print(
            "FPS informado:",
            self.cap.get(cv2.CAP_PROP_FPS)
        )
        print("==================================\n")

        self.latest_frame = None
        self.latest_timestamp = None
        self.frame_id = 0

        self.capture_count = 0
        self.capture_start = time.perf_counter()
        self.read_times.clear()

        self.capture_fps = 0.0
        self.average_read_ms = 0.0

        self.running.set()

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="RealiCameraCapture"
        )

        self.thread.start()

        print("[CAMERA] Thread iniciada.")

        return self

    def _capture_loop(self):
        while self.running.is_set():
            start_read = time.perf_counter()

            success, frame = self.cap.read()

            end_read = time.perf_counter()

            if not success:
                print("[CAMERA] Falha na captura.")
                self.running.clear()
                break

            read_ms = (
                end_read - start_read
            ) * 1000

            # Publica somente o frame mais recente.
            with self.lock:
                self.latest_frame = frame
                self.latest_timestamp = end_read
                self.frame_id += 1

            self.capture_count += 1
            self.read_times.append(read_ms)

            elapsed = (
                time.perf_counter()
                - self.capture_start
            )

            if elapsed >= 5:
                fps_real = (
                    self.capture_count / elapsed
                )

                read_medio = (
                    sum(self.read_times)
                    / len(self.read_times)
                )

                with self.lock:
                    self.capture_fps = fps_real
                    self.average_read_ms = read_medio

                print(
                    f"[CAMERA] FPS real: {fps_real:.1f} | "
                    f"cap.read médio: {read_medio:.1f} ms"
                )

                self.capture_count = 0
                self.capture_start = time.perf_counter()
                self.read_times.clear()

    def read_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None

            return (
                self.frame_id,
                self.latest_frame.copy(),
                self.latest_timestamp
            )

    def get_metrics(self):
        with self.lock:
            return {
                "fps": self.capture_fps,
                "read_ms": self.average_read_ms,
                "frames": self.frame_id
            }

    def is_running(self):
        return self.running.is_set()

    def stop(self):
        self.running.clear()

        if self.thread is not None:
            self.thread.join(timeout=3)

            if self.thread.is_alive():
                print(
                    "[CAMERA] Aviso: thread ainda encerrando."
                )
                # Evita liberar o dispositivo enquanto
                # outra thread ainda executa cap.read().
                return

            self.thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        print("[CAMERA] Webcam encerrada.")
