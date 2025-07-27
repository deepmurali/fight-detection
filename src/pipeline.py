import cv2
import numpy as np

from src.detector import load_yolo_model, run_yolo_inference
from src.strongsort_tracker import load_strongsort

def run_detector_tracker_pipeline(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    model = load_yolo_model()       # YOLOv8 model
    tracker = load_strongsort()     # StrongSORT + ReID (OSNet)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 detection
        detections = run_yolo_inference(model, frame)  # list of dicts

        # Format detections: [[x1, y1, x2, y2, conf, class], ...]
        dets = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["conf"]
            cls = det.get("class", 0)  # default to 0 if class is missing
            dets.append([x1, y1, x2, y2, conf, cls])
        dets = np.array(dets, dtype=np.float32)

        # Run StrongSORT tracking
        outputs = tracker.update(dets=dets, img=frame)

        # Draw results
        for output in outputs:
            x1, y1, x2, y2, track_id, conf, cls, det_ind = map(int, output)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 + StrongSORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
