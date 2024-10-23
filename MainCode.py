# Imports ----------------------------------------------------------------
print("Impprting required Libraries ...")
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import os
import mediapipe as mp
import threading as th
import time
from fastapi import FastAPI
from mangum import Mangum
import uvicorn
import warnings
import keyboard
from datetime import datetime
import queue
capture_queue = queue.Queue()
# Removing  Unnecessary  Warnings
warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# Model Initialisation -------------------------------------------------
print("Loading Models ...")
model_path = 'Models/fgmain.pt'  # Main detection model 
mainModel = YOLO(model_path)
capture_model_path = 'Models/Capture.pt'
captureModel = YOLO(capture_model_path)
app=FastAPI()
handler= Mangum(app)

# stop_flag = 1
capture_flag = False
# flag_lock = th.Lock()


print("Models Loaded Succesfully ..")
# Functions Defined --------------------------------------------------------

def bbox_overlap(bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

@app.get('/')
def endpoint_l():
    output = { 'output':'Hello BY Vivek'}
    return output

@app.get('/hello')
def endpoint_2():
    output = { 'output':'Hello only by api'}
    return output




# Main  ------------------------------------------------------------------------------------------------

id_name_dict = {} 
id_frame_count = {} 
cropped_images_dict = {}


# Mainn Functiomns --------------------------------------------------------------------------------------------------



def productapi():
    print("Starting Server ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)


    # @app.get("/{item_id}")
    # def get_item(item_id):
    #     recommendations = {"Recommendations": recommender(find_closest_movie(f'{item_id}'))}
    #     return recommendations



def start():
    cap_d = cv2.VideoCapture(0)
    if not cap_d.isOpened():
        print("Error: Could not open video.")
        return

    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
    class_names = ['PackType-2', 'BottleType-1', 'PackType-1', 'BottleType-1', 'Lock', 'TubeType-1', 'FreshType-2', 'FreshType-1']
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    while True:
        ret, frame = cap_d.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert to RGB and detect hands
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

        results = mainModel.predict(frame,conf=0.2, verbose=False)
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

                if id_frame_count[track_id] >= 20 and track_id not in id_name_dict:
                    class_name = class_names[class_id] if class_id < len(class_names) else 'Unknown'
                    id_name_dict[track_id] = class_name
                    if len(hands_bbox) == 0:
                        capture_queue.put((class_name, track_id))

                cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (255, 0, 0), 2)
                label = f'ID: {track_id}, Class: {class_names[class_id]}'
                cv2.putText(frame, label, (int(ltrb[0]), int(ltrb[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow('Object Detection', frame)

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_d.release()
    cv2.destroyAllWindows()

def capture_and_detect_and_crop():
    while True:
        if not capture_queue.empty():
            class_name, track_id = capture_queue.get()
            print("Capture Function Called Successfully ...")

            cap_capture = cv2.VideoCapture(1)  
            output_folder = 'LocalStorage'

            if not cap_capture.isOpened():
                print("Error: Could not open video for capturing images.")
                return

            while True:
                ret, frame = cap_capture.read()
                if not ret:
                    print("Error: Could not read frame for capturing images.")
                    break

                results = captureModel(frame, conf=0.05, verbose=False)
                if results and len(results[0].boxes) > 0:
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    for i, detection in enumerate(results[0].boxes):
                        x1, y1, x2, y2 = map(int, detection.xyxy[0])
                        detection_name = captureModel.names[int(detection.cls[0])]
                        cropped_image = frame[y1:y2, x1:x2]
                        image_filename = f"{detection_name}_{int(time.time()) + i}.jpg"
                        image_path = os.path.join(output_folder, image_filename)
                        cv2.imwrite(image_path, cropped_image)

                        if detection_name not in cropped_images_dict:
                            cropped_images_dict[detection_name] = []
                        cropped_images_dict[detection_name].append(image_filename)

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

# MultiThreading  --------------------------------------------
# lock = th.Lock()
t1 = th.Thread(target=start )
# t2 = th.Thread(target=productapi)
# t3 = th.Thread(target=capture_and_detect_and_crop)



t1.start()
# t2.start()
# t3.start()
# time.sleep(30)





t1.join()
# t2.join()
# t3.join()
print(id_name_dict) 
print(id_frame_count)













