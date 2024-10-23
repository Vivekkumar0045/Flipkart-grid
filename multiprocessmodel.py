import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import os
import mediapipe as mp
import multiprocessing
import time
from fastapi import FastAPI
from mangum import Mangum
import warnings

# Removing Unnecessary Warnings
warnings.filterwarnings("ignore")

print("Loading Models ...")
model_path = 'Models/fgmain.pt'  
mainModel = YOLO(model_path)

capture_model_path = 'Models/Capture.pt'
captureModel = YOLO(capture_model_path)

app = FastAPI()
handler = Mangum(app)

def bbox_overlap(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

@app.get('/')
def endpoint_l():
    return {'output': 'Hello BY Vivek'}

@app.get('/hello')
def endpoint_2():
    return {'output': 'Hello only by api'}

def start_detection(capture_queue, id_name_dict, id_frame_count, processed_ids):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
    class_names = ['PackType-2', 'BottleType-1', 'PackType-1', 'BottleType-1', 'Lock', 'TubeType-1', 'FreshType-2', 'FreshType-1']
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

        results = mainModel.predict(frame, verbose=False,conf=0.1)
        detections_for_tracking = []

        if results:
            bboxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for bbox, score, class_id in zip(bboxes, scores, class_ids):
                if score < 0.5:
                    continue
                if not any(bbox_overlap(bbox, hand_bbox) for hand_bbox in hands_bbox):
                    detections_for_tracking.append((bbox, score, class_id))

        tracks = tracker.update_tracks(detections_for_tracking, frame=frame)

        for track in tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.det_class

                if track_id in id_frame_count:
                    id_frame_count[track_id] += 1
                else:
                    id_frame_count[track_id] = 1

                if id_frame_count[track_id] >= 20 and track_id not in id_name_dict and track_id not in processed_ids:
                    class_name = class_names[class_id] if class_id < len(class_names) else 'Unknown'
                    id_name_dict[track_id] = class_name
                    capture_queue.put((class_name, track_id))
                    processed_ids.add(track_id)  

                cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (255, 0, 0), 2)
                label = f'ID: {track_id}, Class: {class_names[class_id]}'
                cv2.putText(frame, label, (int(ltrb[0]), int(ltrb[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def capture_and_detect_and_crop(capture_queue, cropped_images_dict, threshold=10):
    captured_coordinates_dict = {}
    def is_new_coordinates(detection_name, new_coords):
        """Check if new_coords are different from previous ones based on a threshold."""
        if detection_name not in captured_coordinates_dict:
            return True
        for prev_coords in captured_coordinates_dict[detection_name]:
            if all(abs(new - prev) < threshold for new, prev in zip(new_coords, prev_coords)):
                return False
        return True

    while True:
        if not capture_queue.empty():
            class_name, track_id = capture_queue.get()
            print("Capture Function Called Successfully ...")

            cap_capture = cv2.VideoCapture(2)
            output_folder = 'LocalStorage'

            if not cap_capture.isOpened():
                print("Error: Could not open video for capturing images.")
                return

            while True:
                ret, frame = cap_capture.read()
                if not ret:
                    print("Error: Could not read frame for capturing images.")
                    break

                results = captureModel(frame, conf=0.1, verbose=False)
                if results and len(results[0].boxes) > 0:
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    for i, detection in enumerate(results[0].boxes):
                        x1, y1, x2, y2 = map(int, detection.xyxy[0])
                        detection_name = captureModel.names[int(detection.cls[0])]
                        cropped_image = frame[y1:y2, x1:x2]
                        new_coords = (x1, y1, x2, y2)

                        if is_new_coordinates(detection_name, new_coords):
                            image_filename = f"{detection_name}_{int(time.time()) + i}.jpg"
                            image_path = os.path.join(output_folder, image_filename)
                            cv2.imwrite(image_path, cropped_image)

                            if detection_name not in cropped_images_dict:
                                cropped_images_dict[detection_name] = []
                            cropped_images_dict[detection_name].append(image_filename)

                            if detection_name not in captured_coordinates_dict:
                                captured_coordinates_dict[detection_name] = []
                            captured_coordinates_dict[detection_name].append(new_coords)

                            print(f"Saved new image: {image_filename}")
                        else:
                            print(f"Skipped saving image for {detection_name} with the same coordinates.")

                    print(f"Processed {len(results[0].boxes)} detections.")
                    break
                else:
                    print("No results returned by the model.")
                    break

            cap_capture.release()
            cv2.destroyAllWindows()
            continue
        else:
            time.sleep(0.1)


print("Initializing MultProcessing ...")

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    id_name_dict = manager.dict() 
    id_frame_count = manager.dict() 
    cropped_images_dict = manager.dict()
    capture_queue = multiprocessing.Queue()
    processed_ids = set()  

    p1 = multiprocessing.Process(target=start_detection, args=(capture_queue, id_name_dict, id_frame_count, processed_ids))
    p2 = multiprocessing.Process(target=capture_and_detect_and_crop, args=(capture_queue, cropped_images_dict))

    p1.start()
    p2.start()

    p1.join()
    p2.join()


    print(id_name_dict) 
    print(id_frame_count)

