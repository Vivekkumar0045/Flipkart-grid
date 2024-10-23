from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

model_path = 'Models/Apfr.pt'
model = YOLO(model_path)
minconf = 0.007

image_path = 'Data\Veggie\IMG20241001000856.jpg'
results = model(image_path, conf=minconf)[0]

results.save(filename='r.jpg')
ar =0
circle_area = 1
segment_areas = []
segment_dimensions = []
segment_circle_areas = []
if results.masks is not None:
    masks = results.masks.data.numpy()  
    for i, mask in enumerate(masks):
        area = np.sum(mask)  
        segment_areas.append(area)
        ar=area

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)  
            
            segment_dimensions.append((w, h)) 
            
            diameter = max(w, h)  
            circle_area = np.pi * (diameter / 2) ** 2  
            segment_circle_areas.append(circle_area)

else:
    print("No segments found.")


Defect = circle_area-ar
perdef = (Defect/circle_area)*100
print(f"Percentage Defect = {perdef}%")

img = cv2.imread('r.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')  
plt.show()



