from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Union
from pydantic import BaseModel
from datetime import date

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class Product(BaseModel):
    productname: str
    quantity: int
    category: str 
    freshness: Union[int, None] 
    expiry: Union[date, None]  

@app.get("/products", response_model=List[Product])
def get_products():
    products = [
        Product(productname="Cucumber", quantity=3, category="FreshType-2", freshness=90, expiry=None),
        Product(productname="Apple", quantity=5, category="FreshType-1", freshness=65, expiry=None),
        Product(productname="Lays-Spanish Tomato", quantity=2, category="PackType-1", freshness=None, expiry=date(2024, 12, 31)),
        Product(productname="Maaza", quantity=1, category="BottleType-1", freshness=None, expiry=date(2024, 11, 25))
    ]
    return products


