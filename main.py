from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from fastapi import File, UploadFile
from typing import List
from openai import OpenAI
import requests
import base64
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI setup
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
        response = client.chat.completions.create(
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
        return response.choices[0].message.content
    except Exception as e:
        print("Vision API failed:", e)
        return "a fashion item"

@app.get("/ping-openai")
def ping():
    try:
        client.models.list()
        return {"status": "✅ OpenAI reachable"}
    except Exception as e:
        return {"status": "❌ Not reachable", "error": str(e)}


@app.post("/generate-outfit/")
async def generate_outfit(
    prompt: str = Form(...),
    files: List[UploadFile] = File(...),
    tags: List[str] = Form(...)
):
    try:
        labeled_items = []

        for file, tag in zip(files, tags):
            description = await describe_image(file)
            labeled_items.append(f"{tag}: {description}")

        full_prompt = (
            f"{prompt}\nCreate an outfit using the following:\n" +
            "\n".join(labeled_items) +
            "\nShow it on a full-body mannequin, clean background, fashion editorial lighting."
        )

        response = client.images.generate(
            model="dall-e-2",  
            prompt=full_prompt,
            n=1,
            size="1024x1024"
        )

        image_url = response.data[0].url
        return {"image_url": image_url}

    except Exception as e:
        print("Outfit generation failed:", e)
        return {"error": "Something went wrong while generating the outfit."}