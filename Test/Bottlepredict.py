import torch
from ultralytics import YOLO
from PIL import Image
import os

print("Loading the Model ..")
bottle_model = YOLO('Models/bottle.pt')  
print("Successfully Loaded the Model ..")

bottle_types = ['Maaza','Slice']  

def predict_bottle_image(bottle_image_path):
    
    bottle_image = Image.open(bottle_image_path)
    
    bottle_results = bottle_model.predict(bottle_image, task='classify')

    top_bottle_class_index = bottle_results[0].probs.top1 
    top_bottle_confidence = bottle_results[0].probs.top1conf  
    predicted_bottle_name = bottle_types[top_bottle_class_index]

    return predicted_bottle_name, top_bottle_confidence

os.system('cls' if os.name == 'nt' else 'clear')

while True:
    bottle_image_path = input("Enter the path to the bottle image (or 'end' to stop): ")
    
    if bottle_image_path.lower() == 'end':
        print("Prediction process stopped.")
        break
    
    predicted_bottle_name, bottle_confidence = predict_bottle_image(bottle_image_path)
    
    print(f'Predicted bottle type: {predicted_bottle_name}, Confidence: {bottle_confidence:.4f}')

# Images/3a.jpg
