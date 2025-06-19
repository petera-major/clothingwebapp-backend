from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from fastapi import File, UploadFile
from typing import List
from openai import OpenAI
import requests
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://envisionyurtaste.netlify.app"],  
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
async def generate_outfit(
    prompt: str = Form(...),
    tags: List[str] = Form(...),
    files: List[UploadFile] = File(...)
):
    clothing_descriptions = []

    for i in range(len(files)):
        label = tags[i]
        filename = files[i].filename
        clothing_descriptions.append(f"{label}: {filename}")

    outfit_prompt = (
        f"{prompt}\n"
        f"Build an outfit using these uploaded items:\n"
        f"{', '.join(clothing_descriptions)}\n"
        f"Show a full-body mannequin wearing the completed look. Neutral background."
    )

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=outfit_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return {"image_url": response.data[0].url}
    except Exception as e:
        return {"error": str(e)}
