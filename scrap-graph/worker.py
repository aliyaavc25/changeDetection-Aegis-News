# worker.py
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

import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timezone
from urllib.parse import urljoin

# --------------------
# STEP 1: LOAD ENV
# --------------------
load_dotenv()

#AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
#AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
#AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")



#if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION]):
 #   print("❌ Missing AWS credentials in environment variables!")
   # sys.exit(1)

AWS_REGION = "ap-southeast-5"

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

NEWS_SITES_TABLE = dynamodb.Table("aegis_news_sites")
REGIONAL_NEWS_TABLE = dynamodb.Table("aegis_regional_daily_news")


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
    model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="ap-southeast-5",
    temperature=0,
    max_tokens=2048,
)

GRAPH_CONFIG = {
    "llm": {
        "model_instance": bedrock_llm,
        "model_tokens": 8192,
    },
    "headless": True,
    "stealth": True,
    "browser_args": [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
    ],
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



def normalize_date(date_str: str) -> str:
    """
    Converts a datetime string to YYYY-MM-DD format
    """
    if not date_str:
        return datetime.utcnow().strftime("%Y-%m-%d")
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d")

def load_articles_by_region_date(region: str, date_str: str) -> List[Dict[str, Any]]:
    response = REGIONAL_NEWS_TABLE.query(
        KeyConditionExpression=Key("region").eq(region),
        ScanIndexForward=False
    )

    for item in response.get("Items", []):
        if item.get("date") == date_str:
            return item.get("news", [])

    return []


def save_articles(site_name: str, articles: List[Dict[str, Any]], region: str):
    date_key = normalize_date(datetime.utcnow().isoformat())
    generated_at = datetime.utcnow().isoformat() + "Z"

    existing_news = load_articles_by_region_date(region, date_key)
    existing_urls = {a.get("url") for a in existing_news if a.get("url")}

    new_articles = [
        a for a in articles
        if a.get("url") not in existing_urls
    ]

    final_news = existing_news + new_articles

    item = {
        "region": region,
        "generatedAt": generated_at,
        "date": date_key,
        "count": len(final_news),
        "news": final_news
    }

    REGIONAL_NEWS_TABLE.put_item(Item=item)

    # local backup (unchanged)
    local_path = os.path.join(RESULTS_DIR, f"{site_name.lower()}.json")
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=4)


def merge_new_articles(existing, new):
    existing_valid = [a for a in existing if isinstance(a, dict)]
    existing_urls = {a.get("url") for a in existing_valid if a.get("url")}
    
    new_valid = []
    for a in new:
        if not isinstance(a, dict):
            continue
        if a.get("source_url") in existing_urls:
            continue
        new_valid.append(a)
        
    return existing_valid + new_valid

# --------------------
# NEWS EXTRACTOR
# --------------------
class NewsExtractor:
    def process_url(self, url: str) -> Dict[str, Any]:
        domain = url.split("//")[-1].split("/")[0].lower()

        try:
            response = NEWS_SITES_TABLE.get_item(
                Key={"domain": domain}
            )
            return response["Item"]
        except KeyError:
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
# CATEGORY CLASSIFIER (same as Code 1)
# --------------------
async def classify_category(title: str, snippet: str) -> List[str]:
    prompt_text = f"""
    You are a news classifier.
    Given a news headline and short description, return a JSON object:
    {{"category": []}}

    Rules:
    - Choose ONLY from this list: {CATEGORIES}
    - Pick 1–5 categories MAX
    - If none apply, return an EMPTY ARRAY
    - Return ONLY valid JSON

    Title: {title}
    Snippet: {snippet}
    """
    try:
        response = await bedrock_llm.ainvoke([HumanMessage(content=prompt_text)])
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)
        categories = [c for c in data.get("category", []) if c in CATEGORIES]
        return categories
    except Exception:
        return []

# --------------------
# SCRAPE SITE
# --------------------
async def scrape_site(site_name: str, site_url: str):
    try:
        extractor = NewsExtractor()

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

        for idx, link in enumerate(links, start=1):
            try:
                url = link.get("url")

                # ✅ URL NORMALIZATION (same as Code 1)
                if url and not url.startswith("http"):
                    url = urljoin(site_url, url)

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
                "url":  a.get("source_url") or a.get("url")
            }
            for a in merged
            if a.get("title") or a.get("details")
        ]

        # Save exactly like Code 1
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
