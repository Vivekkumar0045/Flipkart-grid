import cv2
from ultralytics import YOLO
import time
import os
import threading

cropped_images_dict = {}
flag_lock = threading.Lock()
def capture_and_detect_and_crop():
    global capture_flag
    global flag_lock

    print("Capture Function Called Successfully ...")

    model = YOLO('Models/Capture.pt') 
    cap = cv2.VideoCapture(1)
    output_folder = 'LocalStorage'

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break  

        print("Performing detection...")  
        try:
            results = model(frame, conf=0.05, verbose=False)
        except Exception as e:
            print(f"Error in detection: {e}")
            break 

        if results and len(results) > 0:
            print("Detection completed successfully.")
            annotated_frame = results[0].plot() 

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            unique_number = int(time.time())  

            for i, detection in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, detection.xyxy[0])  
                class_index = int(detection.cls[0])  
                detection_name = model.names[class_index] 
                cropped_image = frame[y1:y2, x1:x2]
                image_filename = f"{detection_name}_{unique_number + i}.jpg"
                image_path = os.path.join(output_folder, image_filename)
                cv2.imwrite(image_path, cropped_image)

                if detection_name not in cropped_images_dict:
                    cropped_images_dict[detection_name] = []
                cropped_images_dict[detection_name].append(image_filename)

            print(f"Processed {len(results[0].boxes)} detections.")
            break  
        else:
            print("No results returned by the model. Exiting loop.")
            break  

        
        if not capture_flag:
            print("Capture Flag is now False. Stopping capture.")
            break 
        
        time.sleep(0.1)
        break

    print("Captured and Cropped. Capture_flag set to False again.")
    
    capture_flag = False
    
    cap.release()
    cv2.destroyAllWindows()

# cp = capture_and_detect_and_crop()
# print(cp)
capture_and_detect_and_crop()