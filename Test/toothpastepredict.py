import torch
from ultralytics import YOLO
from PIL import Image
import os

print("Loading the Model ..")
toothpaste_model = YOLO('Models/toothpaste.pt') 

toothpaste_types = ['Close Up','Max Fresh'] 

def predict_toothpaste_image(toothpaste_image_path):
    
    toothpaste_image = Image.open(toothpaste_image_path)
    
    toothpaste_results = toothpaste_model.predict(toothpaste_image, task='classify')

    top_toothpaste_class_index = toothpaste_results[0].probs.top1  
    top_toothpaste_confidence = toothpaste_results[0].probs.top1conf  

    predicted_toothpaste_name = toothpaste_types[top_toothpaste_class_index]

    return predicted_toothpaste_name, top_toothpaste_confidence

os.system('cls' if os.name == 'nt' else 'clear')

while True:
    toothpaste_image_path = input("Enter the path to the toothpaste image (or 'end' to stop): ")
    
    if toothpaste_image_path.lower() == 'end':
        print("Prediction process stopped.")
        break
    
    predicted_toothpaste_name, toothpaste_confidence = predict_toothpaste_image(toothpaste_image_path)
    
    print(f'Predicted toothpaste type: {predicted_toothpaste_name}, Confidence: {toothpaste_confidence:.4f}')


# Images/4a.jpg