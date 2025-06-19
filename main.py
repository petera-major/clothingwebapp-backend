from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from fastapi import File, UploadFile
from typing import List
import openai
import requests
import base64
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://envisionyurtaste.netlify.app"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def describe_image(file: UploadFile):
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this clothing item simply."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            ],
            max_tokens=100
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"Vision API failed: {e}")
        return "an item of clothing"

@app.post("/generate-outfit/")
async def generate_outfit(
    prompt: str = Form(...),
    files: List[UploadFile] = File(...),
    tags: List[str] = Form(...)
):
    try:
        labeled_items = []

        for file, tag in zip(files, tags):
            desc = await describe_image(file)
            labeled_items.append(f"{tag}: {desc}")

        combined_items = "\n".join(labeled_items)
        full_prompt = (
            f"{prompt}\n"
            f"Create an outfit using these items:\n"
            f"{combined_items}\n"
            f"Show it styled on a model. High-quality fashion photo."
        )

        image_resp = openai.Image.create(
            prompt=full_prompt,
            n=1,
            size="1024x1024"
        )

        image_url = image_resp["data"][0]["url"]
        return {"image_url": image_url}

    except Exception as e:
        print(f"Outfit generation failed: {e}")
        return {"error": "Something went wrong while generating the outfit."}