
import cv2
import numpy as np
import onnxruntime as ort

from time import perf_counter


class PersonDetector:

    def __init__(
        self,
        model_path="models/yolo11n_960.onnx",
        confidence=0.25,
        iou_threshold=0.65
    ):

        self.confidence = confidence
        self.iou_threshold = iou_threshold

        self.last_timings = {}

        # Configura ONNX Runtime
        options = ort.SessionOptions()

        options.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
        )

        options.enable_mem_pattern = False

        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=[
                "DmlExecutionProvider",
                "CPUExecutionProvider"
            ]
        )

        self.input_name = (
            self.session.get_inputs()[0].name
        )

        self.output_name = (
            self.session.get_outputs()[0].name
        )

        input_shape = (
            self.session.get_inputs()[0].shape
        )

        self.input_height = (
            input_shape[2]
            if isinstance(input_shape[2], int)
            else 640
        )

        self.input_width = (
            input_shape[3]
            if isinstance(input_shape[3], int)
            else 640
        )

        print("\n========== YOLO ==========")
        print("Modelo carregado!")
        print(
            "Provedores:",
            self.session.get_providers()
        )

        print(
            "Entrada:",
            self.input_width,
            "x",
            self.input_height
        )

        print("==========================\n")

    def preprocess(self, frame):

        height, width = frame.shape[:2]

        # Mantém a proporção original
        scale = min(
            self.input_width / width,
            self.input_height / height
        )

        new_width = int(
            round(width * scale)
        )

        new_height = int(
            round(height * scale)
        )

        resized = cv2.resize(
            frame,
            (new_width, new_height)
        )

        # Calcula preenchimento
        pad_x = (
            self.input_width - new_width
        ) / 2

        pad_y = (
            self.input_height - new_height
        ) / 2

        left = int(
            round(pad_x - 0.1)
        )

        right = int(
            round(pad_x + 0.1)
        )

        top = int(
            round(pad_y - 0.1)
        )

        bottom = int(
            round(pad_y + 0.1)
        )

        image = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )

        # BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Normalização
        image = image.astype(
            np.float32
        ) / 255.0

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # Adiciona dimensão do batch
        image = np.expand_dims(
            image,
            axis=0
        )

        return image, scale, left, top

    def detect(self, frame):

        t0 = perf_counter()

        image, scale, pad_x, pad_y = (
            self.preprocess(frame)
        )

        t1 = perf_counter()

        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name: image
            }
        )

        t2 = perf_counter()

        predictions = outputs[0][0]

        # Ajusta formato da saída YOLO
        if predictions.shape[0] < predictions.shape[1]:

            predictions = predictions.T

        frame_height, frame_width = (
            frame.shape[:2]
        )

        # Classe 0 = pessoa no modelo COCO
        person_scores = predictions[:, 4]

        mask = (
            person_scores >= self.confidence
        )

        person_predictions = predictions[mask]

        scores = person_scores[mask]

        # Converte coordenadas simultaneamente
        centers = person_predictions[:, :2]

        sizes = person_predictions[:, 2:4]

        x1 = (
            centers[:, 0]
            - sizes[:, 0] / 2
            - pad_x
        ) / scale

        y1 = (
            centers[:, 1]
            - sizes[:, 1] / 2
            - pad_y
        ) / scale

        x2 = (
            centers[:, 0]
            + sizes[:, 0] / 2
            - pad_x
        ) / scale

        y2 = (
            centers[:, 1]
            + sizes[:, 1] / 2
            - pad_y
        ) / scale

        x1 = np.clip(
            x1,
            0,
            frame_width - 1
        ).astype(np.int32)

        y1 = np.clip(
            y1,
            0,
            frame_height - 1
        ).astype(np.int32)

        x2 = np.clip(
            x2,
            0,
            frame_width - 1
        ).astype(np.int32)

        y2 = np.clip(
            y2,
            0,
            frame_height - 1
        ).astype(np.int32)

        valid = (
            (x2 > x1)
            & (y2 > y1)
        )

        boxes = np.column_stack((
            x1[valid],
            y1[valid],
            x2[valid] - x1[valid],
            y2[valid] - y1[valid]
        ))

        scores = scores[valid]

        t3 = perf_counter()

        # Non-Maximum Suppression
        if len(boxes) > 0:

            indexes = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                self.confidence,
                self.iou_threshold
            )

            indexes = np.array(
                indexes
            ).flatten()

        else:

            indexes = np.array(
                [],
                dtype=int
            )

        t4 = perf_counter()

        detections = []

        for index in indexes:

            x, y, width, height = boxes[index]

            detections.append({

                "class_id": 0,

                "confidence": float(
                    scores[index]
                ),

                "box": (
                    int(x),
                    int(y),
                    int(x + width),
                    int(y + height)
                )

            })

        t5 = perf_counter()

        # Métricas de desempenho
        self.last_timings = {

            "preprocess": (
                t1 - t0
            ) * 1000,

            "inference": (
                t2 - t1
            ) * 1000,

            "decode": (
                t3 - t2
            ) * 1000,

            "nms": (
                t4 - t3
            ) * 1000,

            "output": (
                t5 - t4
            ) * 1000,

            "postprocess": (
                t5 - t2
            ) * 1000,

            "total": (
                t5 - t0
            ) * 1000

        }

        return detections
