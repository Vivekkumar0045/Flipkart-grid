import torch
from ultralytics import YOLO
from PIL import Image
import os

print("Loading the Model ..")
biscuit_model = YOLO('Models/biscuit.pt')  
print("Successfully Loaded the Model ..")

biscuit_types = ['Good Day','Sobisco Biscuit']  

def predict_biscuit_image(biscuit_image_path):
    
    biscuit_image = Image.open(biscuit_image_path)
    
    biscuit_results = biscuit_model.predict(biscuit_image, task='classify')

    top_biscuit_class_index = biscuit_results[0].probs.top1  
    top_biscuit_confidence = biscuit_results[0].probs.top1conf  
    predicted_biscuit_name = biscuit_types[top_biscuit_class_index]

    return predicted_biscuit_name, top_biscuit_confidence

os.system('cls' if os.name == 'nt' else 'clear')

while True:
    biscuit_image_path = input("Enter the path to the biscuit image (or 'end' to stop): ")
    
    if biscuit_image_path.lower() == 'end':
        print("Prediction process stopped.")
        break
    
    predicted_biscuit_name, biscuit_confidence = predict_biscuit_image(biscuit_image_path)
    
    print(f'Predicted biscuit type: {predicted_biscuit_name}, Confidence: {biscuit_confidence:.4f}') 


# Images/5a.jpg