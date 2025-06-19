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
    urls: List[str] = Form(...)
):
    labeled_items = []

    for raw in urls:
        if ": " in raw:
            label, url = raw.split(": ", 1)
        else:
            label, url = "Item", raw  

        title = extract_product_title(url)
        labeled_items.append(f"{label}: {title}")

    combined_items = "\n".join(labeled_items)

    full_prompt = (
        f"{prompt}\n"
        f"Create an outfit that includes the following items:\n"
        f"{combined_items}\n"
        f"Show the full-body outfit on a mannequin. Clean, high-quality background."
    )

    response = client.images.generate(
        model="dall-e-3",
        prompt=full_prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )

    image_url = response.data[0].url
    return {"image_url": image_url}

