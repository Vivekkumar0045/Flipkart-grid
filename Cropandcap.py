print("Impprting required Libraries ...")
import cv2
from ultralytics import YOLO
import numpy as np
import os
import mediapipe as mp
import time

cropped_images_dict = {}
capture_model_path = 'Models/Capture.pt'
captureModel = YOLO(capture_model_path)

def bbox_overlap(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

def capture_and_detect_and_crop():
    while True:
        if True:
            print("Capture Function Called Successfully ...")

            cap_capture = cv2.VideoCapture(0)  
            output_folder = 'LocalStorage'

            if not cap_capture.isOpened():
                print("Error: Could not open video for capturing images.")
                return

            while True:
                ret, frame = cap_capture.read()
                if not ret:
                    print("Error: Could not read frame for capturing images.")
                    break

                results = captureModel(frame, conf=0.25, verbose=False)
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
            
            break
        else:
            time.sleep(0.1)
            break
        

capture_and_detect_and_crop()
print(cropped_images_dict)

