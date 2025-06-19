from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import base64
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
        return "a clothing item"

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

        final_prompt = (
            f"{prompt}\n"
            f"Create an outfit using:\n" + "\n".join(labeled_items) +
            "\nShow the outfit styled on a mannequin in high-quality lighting."
        )

        img_response = client.images.generate(
            model="dall-e-3", 
            prompt=final_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return {"image_url": img_response.data[0].url}
    except Exception as e:
        print("Outfit generation failed:", e)
        return {"error": "Something went wrong while generating the outfit."}
