
import cv2
import numpy as np
import onnxruntime as ort


class PersonDetector:

    def __init__(
        self,
        model_path="models/yolo11n_960.onnx",
        confidence=0.25,
        iou_threshold=0.65
    ):
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # Configura o ONNX Runtime para DirectML
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = False

        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=[
                "DmlExecutionProvider",
                "CPUExecutionProvider"
            ]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape

        self.input_height = (
            input_shape[2] if isinstance(input_shape[2], int) else 640
        )
        self.input_width = (
            input_shape[3] if isinstance(input_shape[3], int) else 640
        )

        print("Modelo carregado!")
        print("Provedores:", self.session.get_providers())
        print("Entrada:", self.input_width, "x", self.input_height)

    def preprocess(self, frame):

        height, width = frame.shape[:2]

        # Mantém a proporção original
        scale = min(
            self.input_width / width,
            self.input_height / height
        )

        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        resized = cv2.resize(
            frame,
            (new_width, new_height)
        )

        # Preenchimento para atingir 640x640
        pad_x = (self.input_width - new_width) / 2
        pad_y = (self.input_height - new_height) / 2

        left = int(round(pad_x - 0.1))
        right = int(round(pad_x + 0.1))
        top = int(round(pad_y - 0.1))
        bottom = int(round(pad_y + 0.1))

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalização e conversão para NCHW
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        return image, scale, left, top

    def detect(self, frame):

        image, scale, pad_x, pad_y = self.preprocess(frame)

        # Inferência
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: image}
        )

        predictions = outputs[0][0]

        # YOLO11 normalmente retorna [84, 8400].
        # Transpõe para [8400, 84].
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes = []
        scores = []

        for prediction in predictions:

            # 4 coordenadas + 80 classes COCO
            class_scores = prediction[4:]

            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            # Classe 0 = pessoa
            if class_id != 0 or confidence < self.confidence:
                continue

            center_x, center_y, box_width, box_height = prediction[:4]

            # Converte XYWH para XYXY
            x1 = (center_x - box_width / 2 - pad_x) / scale
            y1 = (center_y - box_height / 2 - pad_y) / scale
            x2 = (center_x + box_width / 2 - pad_x) / scale
            y2 = (center_y + box_height / 2 - pad_y) / scale

            height, width = frame.shape[:2]

            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y2 = int(np.clip(y2, 0, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ])

            scores.append(confidence)

        # Remove detecções duplicadas
        indexes = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.confidence,
            self.iou_threshold
        )

        detections = []

        for index in np.array(indexes).flatten():

            x, y, width, height = boxes[index]

            detections.append({
                "class_id": 0,
                "confidence": scores[index],
                "box": (x, y, x + width, y + height)
            })

        return detections
