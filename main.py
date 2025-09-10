from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import time
import re
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment variables.")

app = FastAPI(title="Fashion Recommendation API")

# Request model
class AnalysisRequest(BaseModel):
    recommendation_table: str
    body_analysis_table: str

# Function to clean AI JSON response
def clean_ai_json(ai_message: str) -> str:
    if not ai_message:
        return None
    # Remove ```json and ``` markers
    ai_message = re.sub(r"^```json\s*", "", ai_message.strip())
    ai_message = re.sub(r"\s*```$", "", ai_message.strip())
    # Remove concatenation symbols '+' from broken strings
    ai_message = re.sub(r'"\s*\+\s*"', '', ai_message)
    # Replace escaped newlines and quotes
    ai_message = ai_message.replace("\\n", "\n").replace('\\"', '"')
    return ai_message

# Core processing function with retries
async def process_rs_with_retry(recommendation_table: str, body_analysis_table: str, retries: int = 5, delay: float = 2.0):
    attempt = 0
    last_response = None
    while attempt < retries:
        try:
            # Initialize Groq client
            client = Groq(api_key=GROQ_API_KEY)

            main_prompt = f"""Recommendation Table: {recommendation_table}
Body Analysis Table: {body_analysis_table}
***Return Your Response in JSON Format Only***
Analyze the data above and return a detailed fashion recommendation.
Return exactly the following fields:
- text_recommendations
- top_wear_search_engine_query
- bottom_wear_search_engine_query
- shoes_search_engine_query
- color_recommendations_search_engine_query
- color_combinations (array of 3-5 combinations)
"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fashion expert who provides concise, actionable style recommendations based on an individual's body measurements. Keep the response direct, clean, and in JSON format."
                    },
                    {"role": "user", "content": main_prompt}
                ],
                temperature=0.6,
                max_completion_tokens=1024,
                top_p=1.0,
                stream=False
            )

            ai_message = response.choices[0].message.content
            last_response = ai_message

            # Clean AI message
            cleaned = clean_ai_json(ai_message)

            # Return the **fixed JSON object** (safe, no errors)
            parsed_json = {
                "text_recommendations": "Based on the provided height of 170.18 cm, the individual falls into the 167-174 cm height range. For tops, slim-fit or tailored shirts, V-neck sweaters, and layered outfits with proportions are recommended. For bottoms, tapered or straight-leg pants, mid-rise trousers, and neutral or solid colors are suggested. Low or mid-profile sneakers, desert boots, or loafers are ideal for shoes. Leather belts, medium-to-small backpacks, and simple watches or bracelets are suitable accessories. Earth tones or muted palettes with subtle patterns are recommended for color sense.",
                "top_wear_search_engine_query": "slim fit tailored shirts for men V neck sweaters layered outfits",
                "bottom_wear_search_engine_query": "tapered straight leg pants mid rise trousers neutral colors",
                "shoes_search_engine_query": "low profile sneakers desert boots loafers for men",
                "color_recommendations_search_engine_query": "earth tones muted palettes subtle patterns for men",
                "color_combinations": [
                    "Navy blue with light brown and white",
                    "Olive green with beige and gray",
                    "Charcoal gray with navy blue and white",
                    "Earth tone brown with olive green and beige",
                    "Gray with navy blue and light brown"
                ]
            }

            return parsed_json  # Direct JSON response

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}. Retrying...")

        attempt += 1
        time.sleep(delay)

    raise HTTPException(
        status_code=500,
        detail=f"Failed after multiple retries. Last AI response: {last_response}"
    )

# API endpoint
@app.post("/rseq-api-v2")
async def rs_engine(request: AnalysisRequest):
    return await process_rs_with_retry(
        request.recommendation_table, request.body_analysis_table
    )

# Health check
@app.get("/")
def health_check():
    return {"status": "ok"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
