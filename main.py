from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List
from openai import OpenAI
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

@app.post("/generate-outfit/")
async def generate_outfit(
    prompt: str = Form(...),
    tags: List[str] = Form(...)
):
    try:
        full_prompt = (
            f"{prompt}\n"
            f"Use the following clothing items:\n"
            f"{chr(10).join(tags)}\n"
            f"Create a stylish outfit shown on a mannequin with a clean sleek white background."
        )

        image_response = client.images.generate(
            model="dall-e-2",
            prompt=full_prompt,
            n=1,
            size="1024x1024"
        )
        image_url = image_response.data[0].url

        rec_prompt = (
            f"Based on this style: {prompt}\n"
            f"and these items: {', '.join(tags)}\n"
            f"List 5 real fashion shopping links (Zara, H&M, ASOS, etc.) that match the look."
        )

        rec_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": rec_prompt}
            ],
            max_tokens=300
        )

        import re
        links = re.findall(r'https?://\S+', rec_response.choices[0].message.content)

        return {
            "image_url": image_url,
            "recommendations": links[:5]  
        }

    except Exception as e:
        print("Outfit generation failed:", e)
        return {"error": "Something went wrong while generating the outfit."}
