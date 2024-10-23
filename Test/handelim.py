import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import mediapipe as mp

model_path = 'Models/gan2.pt'  
model = YOLO(model_path)

cap = cv2.VideoCapture(0)
def bbox_overlap(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)
    
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=0.5)

# class_names = ["toothpaste", "deodorant", "facewash", "goodday"]
class_names = ['Biscuit', 'Chips', 'Deodorant', 'Toothpaste', 'veggreen', 'vegred']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    hands_bbox = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
            hands_bbox.append([x_min, y_min, x_max, y_max])

    results = model(frame)

    detections = results[0].boxes
    if detections is not None:
        bboxes = detections.xyxy.cpu().numpy()  
        scores = detections.conf.cpu().numpy() 
        class_ids = detections.cls.cpu().numpy().astype(int) 

        detections_for_tracking = []
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            
            bbox = list(map(float, bbox[:4]))  
            
            if not any(bbox_overlap(bbox, hand_bbox) for hand_bbox in hands_bbox):
                detections_for_tracking.append((bbox, score, class_id))
        
        tracks = tracker.update_tracks(detections_for_tracking, frame=frame)

        for track in tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                track_id = track.track_id
                ltrb = track.to_ltrb() 
                class_id = track.det_class  

                class_name = class_names[class_id] if class_id < len(class_names) else 'Unknown'

                cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(frame, f'Class: {class_name}', (int(ltrb[0]), int(ltrb[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow('Real-Time Object Detection and Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


