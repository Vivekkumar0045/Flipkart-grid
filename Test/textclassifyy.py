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
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    text = ' '.join([word for word in text.split() if word not in stop_words]) 
    return text


with open('Models/vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

with open('Models/cosine_similarity.pkl', 'rb') as f:
    loaded_cosine_sim = pickle.load(f)


new_vec = loaded_vectorizer.transform(new_data)
df['combined'] = df['product name'].fillna('') + ' ' + df['product description'].fillna('')

df['cleaned'] = df['combined'].apply(preprocess_text)
df_vectorized = loaded_vectorizer.transform(df['cleaned']) 

similarities = cosine_similarity(new_vec, df_vectorized)
similarity_scores = similarities[0]

sorted_indices = np.argsort(similarity_scores)[::-1] 
unique_products = set()
count = 0

new_data = []



print("Top 5 similar products:")
for idx in sorted_indices:
    product_name = df['product name'].iloc[idx]

    if product_name not in unique_products:
        print(f"Product: {product_name} | Similarity Score: {similarity_scores[idx]}")
        unique_products.add(product_name) 
        count += 1
        
    if count == 5:
        break