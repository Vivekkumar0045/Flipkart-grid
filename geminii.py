import google.generativeai as genai
from IPython.display import Markdown
import os
import cv2
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from nltk.corpus import stopwords
import pickle
import nltk
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
df = pd.read_csv('Data/product_new.csv')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, float):  
        return ""
    text = str(text).lower() 
    text = re.sub(r'\s+', ' ', text)  #  extra spaces
    text = re.sub(r'[^\w\s]', '', text)  #  special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  #  stopwords
    return text


with open('Models/vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

with open('Models/cosine_similarity.pkl', 'rb') as f:
    loaded_cosine_sim = pickle.load(f)


GOOGLE_API_KEY = 'AIzaSyAmfhblnwg1JxT5hOfJqv7sAASHlnO1cvE'

genai.configure(api_key=GOOGLE_API_KEY)


def delete_image(filename):
    try:
        os.remove(filename)
        print(f"Image {filename} deleted successfully.")
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
    except Exception as e:
        print(f"Error: {e}")

def capture_image(number):
    filename = f'captured_frame_{number}.png'
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
    else:
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")
    
    cap.release()

capture_image(1)
image_path = 'captured_frame_1.png'
# image_path = 'Images/dif.jpg'

sample_file = genai.upload_file(path=image_path, display_name="screenshot")

# print(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

# Prompt 
response = model.generate_content([sample_file, "Extract text from the given image . only provide me the text dont write any other word "])

# print(response.text)
delete_image(image_path)

def make_single_line(text):
   
    return ' '.join(text.splitlines()).replace("  ", " ")


# ttxt = '''' ''''''
single_line = make_single_line(response.text)
# print(single_line)

new_data = [single_line]

new_vec = loaded_vectorizer.transform(new_data)
df['combined'] = df['product name'].fillna('') + ' ' + df['product description'].fillna('')

df['cleaned'] = df['combined'].apply(preprocess_text)
df_vectorized = loaded_vectorizer.transform(df['cleaned']) 

similarities = cosine_similarity(new_vec, df_vectorized)
similarity_scores = similarities[0]

sorted_indices = np.argsort(similarity_scores)[::-1] 
unique_products = set()
count = 0

response2= model.generate_content(response.text + "Extract any date from the given text and modify it in  yyyy/mm/dd format and dont write anything else ")

print("Top 5 similar products:")
for idx in sorted_indices:
    product_name = df['product name'].iloc[idx]

    if product_name not in unique_products:
        print(f"Product: {product_name} | Similarity Score: {similarity_scores[idx]}")
        unique_products.add(product_name) 
        count += 1
        
    if count == 5:
        break


print(response2.text)
