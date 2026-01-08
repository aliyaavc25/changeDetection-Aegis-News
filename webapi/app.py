from fastapi import FastAPI
import httpx

from pydantic import BaseModel,Field,ConfigDict
from typing import Optional
import json

class AppriseNotification(BaseModel):
    version: str
    title: str
    message: str # This contains your ScrapeData as a JSON string
    type: str

class ScrapeData(BaseModel):
    #url: str
    #prompt: str = "Summarize the main content of this page" # Default value
    # This captures the {{diff}} token from changedetection
    url: str = Field(alias="base_url")
    changes: Optional[str] = None 
    # This captures the full current text
    current_snapshot: Optional[str] = None
    prompt: str = "Extract the news headlines and details"
    # This allows the model to be initialized using either 'url' or 'base_url'
    model_config = ConfigDict(populate_by_name=True)



app = FastAPI()

from fastapi.exceptions import RequestValidationError
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation Error: {exc.errors()}") # This will show exactly which field failed
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@app.post("/trigger-scrape")
async def trigger(notification: AppriseNotification):
    # This calls your separate scraper service
    #url = request_data.url
    #prompt = request_data.prompt
    # 1. Parse the string inside 'message' into our ScrapeData model
    raw_data = json.loads(notification.message, strict=False)
    # --- NEW PRINT LOG ---
    print("\n--- INCOMING PAYLOAD FROM CHANGEDETECTION ---")
    print(json.dumps(raw_data, indent=2)) # Formats the payload for readability
    print("---------------------------------------------\n")
    request_data = ScrapeData(**raw_data)

    payload = {
        "url": request_data.url,
        "diff": request_data.changes, 
        "full_text": request_data.current_snapshot,
        "prompt": request_data.prompt
    }
    #print(f"Scraping: {url}")
    custom_timeout = httpx.Timeout(300.0, connect=10.0)
    #return {"message": "Success", "url": url}
    async with httpx.AsyncClient(timeout=custom_timeout) as client:
        try:
            response = await client.post(
                "http://localhost:8001/scrape", 
                json=payload
            )
            return response.json()
        except httpx.TimeoutException:
            return {"error": "The scraper took too long to respond. Try a simpler prompt."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)