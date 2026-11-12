
import cv2


class USBCamera:

    def __init__(
        self,
        camera_index=0,
        width=1280,
        height=720,
        fps=30
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self):

        # Tenta abrir utilizando DirectShow
        self.cap = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW
        )

        # Alternativa: Media Foundation
        if not self.cap.isOpened():

            self.cap.release()

            self.cap = cv2.VideoCapture(
                self.camera_index,
                cv2.CAP_MSMF
            )

        if not self.cap.isOpened():

            self.release()

            raise RuntimeError(
                f"Não foi possível acessar a câmera "
                f"USB de índice {self.camera_index}"
            )

        # Configurações da câmera
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

        # Tenta manter apenas frames recentes
        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        actual_width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        actual_height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        actual_fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        print("\nCÂMERA CONECTADA")
        print(f"Resolução: {actual_width}x{actual_height}")
        print(f"FPS solicitado: {self.fps}")
        print(f"FPS informado: {actual_fps}")

        return self

    def read(self):

        if self.cap is None or not self.cap.isOpened():
            return False, None

        return self.cap.read()

    def release(self):

        if self.cap is not None:

            self.cap.release()
            self.cap = None

            print("Câmera desconectada.")
