import cv2
import os
import time

save_folder = "ToothPaste/Max_Fresh" 
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

cap = cv2.VideoCapture(0)

total_images = 100
time_limit = 10  

capture_interval = time_limit / total_images
n=340
start_time = time.time()
for i in range(total_images):
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break
       
    file_name = f"image_{n+1}.jpg"
    file_path = os.path.join(save_folder, file_name)
    cv2.imwrite(file_path, frame)
    n=n+1
    cv2.imshow('Captured Image', frame)

    elapsed_time = time.time() - start_time
    if elapsed_time >= time_limit:
        break 

    time.sleep(capture_interval)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Captured {i+1} images in {save_folder}")
print(n)
