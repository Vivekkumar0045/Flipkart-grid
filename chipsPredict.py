import torch
from ultralytics import YOLO
from PIL import Image
import os

print("Loading the Model ..")
chips_model = YOLO('Models/chips2.pt')  
print("Successfully Loadded the Model ..")
chip_types = ['Green_Lays', 'Kurkure', 'Red_Lays']  


def predict_chip_image(chip_image_path):
    
    chip_image = Image.open(chip_image_path)
    
    chip_results = chips_model.predict(chip_image, task='classify')

    top_chip_class_index = chip_results[0].probs.top1  
    top_chip_confidence = chip_results[0].probs.top1conf  

    predicted_chip_name = chip_types[top_chip_class_index]

    return predicted_chip_name, top_chip_confidence

chip_image_path = 'LocalStorage\Chips_1729440833.jpg' 
predicted_chip_name, chip_confidence = predict_chip_image(chip_image_path)

print(f'Predicted chip type: {predicted_chip_name}, Confidence: {chip_confidence:.4f}')

# os.system('cls' if os.name == 'nt' else 'clear')

# while True:
#     chip_image_path = input("Enter the path to the chip image (or 'end' to stop): ")
    
#     if chip_image_path.lower() == 'end':
#         print("Prediction process stopped.")
#         break
    
#     predicted_chip_name, chip_confidence = predict_chip_image(chip_image_path)
    
#     print(f'Predicted chip type: {predicted_chip_name}, Confidence: {chip_confidence:.4f}')
