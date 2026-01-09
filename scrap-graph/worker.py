import sys
import os
import json
import traceback
import asyncio
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_aws import ChatBedrockConverse



# --- STEP 1: LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
API_KEY = os.getenv("API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION]):
    print("❌ Missing AWS credentials in environment variables!")
    sys.exit(1)


# --------------------
# STEP 2: COMPAT PATCH
# --------------------
try:
    import langchain_core
    import langchain_core.output_parsers as op

    try:
        from langchain_core.output_parsers.structured import (
            ResponseSchema,
            StructuredOutputParser,
        )
    except ImportError:
        class ResponseSchema:
            def __init__(self, name, description, type="string"):
                self.name = name
                self.description = description
                self.type = type

        class StructuredOutputParser:
            @classmethod
            def from_response_schemas(cls, schemas):
                return cls()

    op.ResponseSchema = ResponseSchema
    op.StructuredOutputParser = StructuredOutputParser
    sys.modules["langchain_core.output_parsers.ResponseSchema"] = ResponseSchema

    print("✅ Compatibility patch applied")

except ImportError:
    print("❌ langchain-core not installed")
    sys.exit(1)

# --------------------
# STEP 3: IMPORT SCRAPER
# --------------------
from scrapegraphai.graphs import SmartScraperGraph

# --------------------
# STEP 4: BEDROCK
# --------------------
bedrock_llm = ChatBedrockConverse(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1",
    temperature=0,
    max_tokens=8192,
)

GRAPH_CONFIG = {
    "llm": {
        "model_instance": bedrock_llm,
        "model_tokens": 8192,
    },
    "headless": False,
    "stealth": True,
    "browser_args": ["--no-cache"],
    "verbose": True,
}

# --------------------
# RESULTS STORAGE
# --------------------
RESULTS_DIR = "results"

def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def site_to_filename(site_name: str) -> str:
    safe_name = site_name.lower().replace(" ", "_")
    return os.path.join(RESULTS_DIR, f"{safe_name}.json")

def load_existing_articles(site_name: str) -> List[Dict[str, Any]]:
    path = site_to_filename(site_name)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_articles(site_name: str, articles: List[Dict[str, Any]]):
    path = site_to_filename(site_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)

def merge_new_articles(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing_urls = {a.get("source_url") for a in existing if isinstance(a, dict)}
    new_unique = [a for a in new if isinstance(a, dict) and a.get("source_url") not in existing_urls]
    return existing + new_unique


# --- STEP 3: FASTAPI SETUP ---
app = FastAPI(title="Aegis Scraper Worker")


# --- STEP 4: DEFINE REQUEST AND RESPONSE SCHEMAS ---
class ScrapeRequest(BaseModel):
    url: str
    prompt: Optional[str] = None
    diff: Optional[str] = None
    full_text: Optional[str] = None

class NewsArticle(BaseModel):
    headline: str
    url: str
    details: Optional[str] = None

class NewsData(BaseModel):
    news: List[NewsArticle]

class ArticleContent(BaseModel):
    headline: str
    summary: str


# --- STEP 5: NEWS EXTRACTOR CLASS ---

class NewsExtractor:
    def __init__(self, config_path: str = "config.yaml"):
        self.global_config = self._load_yaml_config(config_path)

    def _load_yaml_config(self, config_path: str):
        print("🔄 Loading configuration from", config_path)
        return {
            "api_key": API_KEY,  # Use the API key from .env
            "default_model": DEFAULT_MODEL  # Default model from .env
        }

    def _get_source_metadata(self, url: str):
        print(f"🔍 Extracting metadata for {url}")
        metadata_mapping = {
            "kompas.com": {
                "prompt": "Extract the headline and body text from this Kompas article.",
                "model": "openai/gpt-4"
            },
        }
        for domain, metadata in metadata_mapping.items():
            if domain in url.lower():
                print(f"✅ Matched domain: {domain}")
                return metadata
        return {"prompt": "Summarize this news page into a JSON object with title and content.", "model": self.global_config["default_model"]}

    def process_url(self, url: str) -> Dict[str, Any]:
        metadata = self._get_source_metadata(url)
        print(f"✅ Metadata processed: {metadata}")
        return metadata


# --- STEP 6: SCRAPE SITE FUNCTION ---
async def scrape_site(site_name: str, site_url: str):
    try:
        news_extractor = NewsExtractor(config_path="config.yaml")
        metadata = news_extractor.process_url(site_url)
        prompt = metadata["prompt"]
        model = metadata["model"]

        main_graph = SmartScraperGraph(
            prompt=prompt,
            source=site_url,
            config={**GRAPH_CONFIG, "depth": 1},
            schema=NewsData,
        )
        result = await asyncio.to_thread(main_graph.run)
        urls = [item["url"] for item in result["news"]][:2]

        all_articles = []
        for url in urls:
            try:
                article_graph = SmartScraperGraph(
                    prompt=prompt,
                    source=url,
                    config=GRAPH_CONFIG,
                    schema=ArticleContent,
                )
                article = await asyncio.to_thread(article_graph.run)
                article["source_url"] = url
                all_articles.append(article)
            except Exception as e:
                print(f"❌ Failed to scrape article {url}: {e}")

        ensure_results_dir()
        existing_articles = load_existing_articles(site_name)
        updated_articles = merge_new_articles(existing_articles, all_articles)
        save_articles(site_name, updated_articles)
        print(f"💾 {site_name}: {len(updated_articles) - len(existing_articles)} new articles saved")

        return updated_articles

    except Exception as e:
        print(f"❌ Failed to scrape site {site_name}: {e}")
        traceback.print_exc()
        return []


# --- STEP 7: API ENDPOINT ---
@app.post("/scrape")
async def run_scrape(request: ScrapeRequest):
    try:
        print(f"🔍 Starting scrape for {request.url}")
        site_name = request.url.split("//")[-1].split("/")[0]
        articles = await scrape_site(site_name, request.url)
        print("\n🎉 Final Scraped Articles:")
        return {"status": "success", "data": articles}

    except Exception as e:
        print("❌ FATAL ERROR in main scraping process")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- ENTRY POINT ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)