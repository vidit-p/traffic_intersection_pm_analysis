from google import genai
from google.genai import types
from pydantic import BaseModel


# -----------------------------
# Configuration
# -----------------------------

PROJECT_ID = "focused-history-278517"
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash"


# -----------------------------
# Response schema
# -----------------------------

class GeminiResponse(BaseModel):
    answer: str
    confidence: float
    reasoning: str


# -----------------------------
# Gemini client
# -----------------------------

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    http_options=types.HttpOptions(api_version="v1"),
)


# -----------------------------
# Main
# -----------------------------

def main():
    prompt = """
    What are some possible reasons why an intersection
    might have high average delay?
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiResponse,
        ),
    )

    result = GeminiResponse.model_validate_json(response.text)

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()