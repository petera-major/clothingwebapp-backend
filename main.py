from fastapi import FastAPI, Form
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from fastapi import File, UploadFile
from typing import List
import requests
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_product_title(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        title_tag = soup.find("title")
        return title_tag.text.strip() if title_tag else "fashion item"
    except Exception as e:
        return f"fashion item (error scraping: {str(e)})"

@app.post("/describe-upload/")
async def describe_upload(file: UploadFile = File(...)):
    contents = await file.read()

@app.post("/generate-outfit/")
async def generate_outfit(request: Request, prompt: str = Form(...)):
    try:

        form = await request.form()
        urls = form.getlist("urls")  

        items_from_links = [extract_product_title(url) for url in urls]
        combined_items = ", ".join(items_from_links)

        full_prompt = f"{prompt}, including: {combined_items}"

        formatted_prompt = (
            f"A full-body mannequin photo of a trendy, realistic outfit worn by a stylish young adult. "
            f"Include the following items: {full_prompt}. "
            f"The background should be clean or neutral. High-quality fashion photography style."
        )

        response = openai.Image.create(
            prompt=formatted_prompt,
            n=1,
            size="512x512"
        )
        image_url = response["data"][0]["url"]
        return {"image_url": image_url}
    
    except Exception as e:
        return {"error": str(e)}