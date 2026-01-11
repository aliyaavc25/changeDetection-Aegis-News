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
from langchain_core.messages import HumanMessage

from pymongo import MongoClient
from datetime import datetime, timezone

# --------------------
# STEP 1: LOAD ENV
# --------------------
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "newsDB"

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION]):
    print("❌ Missing AWS credentials in environment variables!")
    sys.exit(1)

mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[DB_NAME]
scraped_news_collection = mongo_db["scraped_news"]

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


from scrapegraphai.graphs import SmartScraperGraph
# --------------------
# STEP 3: BEDROCK
# --------------------
bedrock_llm = ChatBedrockConverse(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1",
    temperature=0,
    max_tokens=2048,
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
# CATEGORIES
# --------------------
CATEGORIES = [
    "Energy", "Environment", "Economy", "Politic", "Health", "Humanitarian",
    "Geopolitical", "Cyber", "Space", "Maritime Security", "International Law",
    "Science", "Technology", "Terrorism", "Transnational Crime", "Military", "Security"
]

# --------------------
# FASTAPI
# --------------------
app = FastAPI(title="Aegis Scraper Worker")

class ScrapeRequest(BaseModel):
    url: str
    prompt: Optional[str] = None
    diff: Optional[str] = None
    full_text: Optional[str] = None

# --------------------
# HELPERS
# --------------------
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_existing_articles(site_name: str) -> List[Dict[str, Any]]:
    cursor = scraped_news_collection.find(
        {"source": site_name},
        {"_id": 0}
    )
    return list(cursor)

def save_articles(site_name: str, articles: List[Dict[str, Any]], region: str):
    valid_articles = [
        {
            "title": a.get("title"),
            "details": a.get("details"),
            "date": a.get("date"),
            "category": a.get("category", []),
            "source": site_name,
            "url": a.get("source_url")
        }
        for a in articles
        if a.get("title") or a.get("details")
    ]

    final_output = {
        "region": region,
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "count": len(valid_articles),
        "news": valid_articles
    }

    # Save to MongoDB
    scraped_news_collection.update_one(
        {"source": site_name},
        {"$set": final_output},
        upsert=True
    )

    # ✅ Save locally like test.py
    local_path = os.path.join(RESULTS_DIR, f"{site_name.lower()}.json")
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print(f"💾 Saved news locally to {local_path}")

def merge_new_articles(existing, new):
    existing_valid = [a for a in existing if isinstance(a, dict)]
    existing_urls = {a.get("url") for a in existing_valid if a.get("url")}
    return existing_valid + [
        a for a in new
        if isinstance(a, dict) and a.get("source_url") not in existing_urls
    ]

# --------------------
# NEWS EXTRACTOR
# --------------------
class NewsExtractor:
    def __init__(self, db):
        self.collection = db["news_site"]

    def process_url(self, url: str) -> Dict[str, Any]:
        for site in self.collection.find({}, {"_id": 0}):
            if site["domain"] in url.lower():
                return site
        return {
            "prompt": "Summarize this news page into a JSON object with title and content.",
            "region": "Unknown"
        }

# --------------------
# SCHEMAS
# --------------------
class NewsLink(BaseModel):
    url: str
    title: Optional[str] = None

class NewsData(BaseModel):
    news: List[NewsLink]

class ArticleContent(BaseModel):
    title: str
    details: str
    date: Optional[str] = None
    category: List[str] = []

# --------------------
# CATEGORY CLASSIFIER
# --------------------
async def classify_category(title: str, snippet: str) -> List[str]:
    prompt_text = f"""
    Return JSON only:
    {{ "category": [] }}

    Rules:
    - Choose ONLY from: {CATEGORIES}
    - Pick 1–5 categories max
    - If none apply, return empty array

    Title: {title}
    Snippet: {snippet}
    """
    try:
        response = await bedrock_llm.ainvoke([HumanMessage(content=prompt_text)])
        content = response.content
        if "```" in content:
            content = content.split("```")[1].strip()
        data = json.loads(content)
        return [c for c in data.get("category", []) if c in CATEGORIES]
    except Exception:
        return []

# --------------------
# SCRAPE SITE
# --------------------
async def scrape_site(site_name: str, site_url: str):
    try:
        extractor = NewsExtractor(mongo_db)
        metadata = extractor.process_url(site_url)

        list_prompt = """
        Extract a list of news articles.
        Return JSON:
        {
          "news": [
            { "url": "...", "title": "..." }
          ]
        }
        """

        main_graph = SmartScraperGraph(
            prompt=list_prompt,
            source=site_url,
            config={**GRAPH_CONFIG, "depth": 1},
            schema=NewsData,
        )

        result = await asyncio.to_thread(main_graph.run)
        if hasattr(result, "model_dump"):
            result = result.model_dump()

        links = result.get("news", [])[:2]
        all_articles = []

        for link in links:
            try:
                url = link.get("url")
                # ✅ DEDUPLICATION: Skip if URL already exists in MongoDB
                if scraped_news_collection.find_one({"url": url}):
                    print(f"⏭️ Skipping already scraped: {url}")
                    continue
                title = link.get("title")

                categories = await classify_category(title or "", title or "")
                if not categories:
                    continue

                article_graph = SmartScraperGraph(
                    prompt=metadata["prompt"],
                    source=url,
                    config=GRAPH_CONFIG,
                    schema=ArticleContent,
                )

                article = await asyncio.to_thread(article_graph.run)
                if hasattr(article, "model_dump"):
                    article = article.model_dump()

                article["title"] = article.get("title") or title
                article["category"] = categories
                article["source_url"] = url
                all_articles.append(article)

            except Exception:
                traceback.print_exc()

        existing = load_existing_articles(site_name)
        merged = merge_new_articles(existing, all_articles)

        final_news = [
            {
                "date": a.get("date"),
                "details": a.get("details"),
                "source": site_name,
                "title": a.get("title"),
                "category": a.get("category", []),
                "url": a.get("source_url")
            }
            for a in merged
            if a.get("title") or a.get("details")
        ]

        # Save to both MongoDB and local JSON
        save_articles(site_name, final_news, metadata.get("region", "Unknown"))
        return final_news

    except Exception:
        traceback.print_exc()
        return []

# --------------------
# API ENDPOINT
# --------------------
@app.post("/scrape")
async def run_scrape(request: ScrapeRequest):
    try:
        print(f"🔍 Starting scrape for {request.url}")
        site_name = request.url.split("//")[-1].split("/")[0]
        articles = await scrape_site(site_name, request.url)
        return {"status": "success", "data": articles}

    except Exception as e:
        print("❌ FATAL ERROR in main scraping process")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --------------------
# ENTRY POINT
# --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
