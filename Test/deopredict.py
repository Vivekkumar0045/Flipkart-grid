import torch
from ultralytics import YOLO
from PIL import Image
import os

print("Loading the Model ..")
deo_model = YOLO('Models/deo.pt') 
print("Successfully Loaded the Model ..")

deo_types = ['Set Wet','Yardley'] 

def predict_deo_image(deo_image_path):
    
    deo_image = Image.open(deo_image_path)
    
    deo_results = deo_model.predict(deo_image, task='classify')

    top_deo_class_index = deo_results[0].probs.top1 
    top_deo_confidence = deo_results[0].probs.top1conf 

    predicted_deo_name = deo_types[top_deo_class_index]

    return predicted_deo_name, top_deo_confidence

os.system('cls' if os.name == 'nt' else 'clear')

while True:
    deo_image_path = input("Enter the path to the deo image (or 'end' to stop): ")
    
    if deo_image_path.lower() == 'end':
        print("Prediction process stopped.")
        break
    
    predicted_deo_name, deo_confidence = predict_deo_image(deo_image_path)
    
    print(f'Predicted deo type: {predicted_deo_name}, Confidence: {deo_confidence:.4f}') 


# Images/2c.jpg