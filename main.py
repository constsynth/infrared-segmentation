import cv2
from PIL import Image
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import os
import torch
from math import dist


print(torch.cuda.is_available())
font = cv2.FONT_HERSHEY_SIMPLEX

def upload_model(path_to_model: str = None, device: str = 'cuda'):
    model = YOLO(path_to_model).to(device)
    return model

def estimate_speed(point_1, point_2, ppm_rate: int = 8, fps: int = 15) -> int:
    d_pixel = dist(point_1, point_2)
    d_meters = d_pixel/ppm_rate
    time_constant = fps*3.6
    speed = d_meters * time_constant
    return int(speed)

def tracking(rtsp_address: str = None, path_to_model: str = None):
    model = upload_model(path_to_model)
    cap = cv2.VideoCapture(rtsp_address)

    track_history = defaultdict(lambda: [])

    while True:
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True, conf=0.3, tracker='bytetrack.yaml')

            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            annotated_frame = results[0].plot()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((int(x), int(y)))
                if len(track) >= 2:
                    speed = estimate_speed(track[-1], track[-2], ppm_rate=16)
                    cv2.putText(annotated_frame, f'{speed} km/h', track[-1], font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if len(track) > 50:
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            cv2.imshow("infrared image tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tracking(rtsp_address='30 Minutes of Cars Driving By in 2009.mp4', path_to_model='./yolov8s-seg.pt')