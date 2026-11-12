
import numpy as np
import supervision as sv

from trackers import ByteTrackTracker


class PersonTracker:

    def __init__(self):

        self.tracker = ByteTrackTracker(
            frame_rate=10.0,
            lost_track_buffer=30,
            track_activation_threshold=0.25,
            minimum_consecutive_frames=1
        )

    def update(self, detections, timestamp=None):

        boxes = []
        confidences = []
        class_ids = []

        for detection in detections:

            boxes.append(detection["box"])
            confidences.append(detection["confidence"])
            class_ids.append(detection["class_id"])

        # Converte para o formato Supervision
        sv_detections = sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int)
        )

        # Executa o rastreamento
        tracked = self.tracker.update(sv_detections)

        return tracked
