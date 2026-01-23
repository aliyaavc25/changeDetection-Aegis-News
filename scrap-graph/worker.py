# worker.py
import sys
import os
import json
import traceback
import asyncio
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

from datetime import datetime, timezone, timedelta
import time


# --------------------
# STEP 1: LOAD ENV
# --------------------
load_dotenv()

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

MY_TZ = timezone(timedelta(hours=8))




def normalize_date(date_str: str) -> str:
    if not date_str:
        return datetime.now(MY_TZ).strftime("%Y-%m-%d")
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.astimezone(MY_TZ).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now(MY_TZ).strftime("%Y-%m-%d")




def save_articles(site_name: str, articles: List[Dict[str, Any]], region: str):
    date_key = normalize_date(datetime.now(MY_TZ).isoformat())

    
    print(f"\n🔹 [DEBUG] save_articles called for site: {site_name}")
    print(f"    Region: {region}, Date Key: {date_key}")
    print(f"    Total articles scraped: {len(articles)}")
    
    # 1️⃣ Load existing item
    try:
        response = dynamodb_with_retry(
            lambda: REGIONAL_NEWS_TABLE.get_item(
                Key={"region": region, "generatedAt": date_key}
            )
        )
    except Exception as e:
        print("❌ Failed to fetch from DynamoDB after retries:", e)
        response = {}


    # 🔒 FULL-REGION DEDUPLICATION
    existing_urls = dynamodb_with_retry(lambda: get_existing_urls_for_region(region))
    new_articles = [a for a in articles if a.get("url") not in existing_urls]

    print(f"    New articles to insert: {len(new_articles)}")
    if new_articles:
        for a in new_articles:
            print(f"        ✅ {a.get('title')} ({a.get('url')})")
    else:
        print("    ⚠️ No new articles to insert (all duplicates or missing URLs)")
        return

    # 3️⃣ Update ONLY if new news exists
    try:
        dynamodb_with_retry(
            lambda: REGIONAL_NEWS_TABLE.update_item(
                Key={"region": region, "generatedAt": date_key},
                UpdateExpression="""
                    SET #news = list_append(if_not_exists(#news, :empty), :new),
                        #count = if_not_exists(#count, :zero) + :inc,
                        #date = :date
                """,
                ExpressionAttributeNames={
                    "#news": "news",
                    "#count": "count",
                    "#date": "date",
                },
                ExpressionAttributeValues={
                    ":new": new_articles,
                    ":empty": [],
                    ":inc": len(new_articles),
                    ":zero": 0,
                    ":date": date_key,
                },
            )
        )

        print(f"    ✅ Successfully updated DynamoDB with {len(new_articles)} new articles")

    except Exception as e:
        print("❌ Failed to update DynamoDB after retries:", e)
        print("⚠️ Writing local JSON backup to disk...")

        save_local_backup(
            region=region,
            date_key=date_key,
            articles=new_articles,
        )

def dynamodb_with_retry(fn, retries=3, delay=1):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            print(f"⚠️ DynamoDB attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(delay * attempt)  # simple backoff
    raise last_exc


def save_local_backup(region: str, date_key: str, articles: list):
    try:
        filename = f"backup_{region}_{date_key}.json"
        path = os.path.join(RESULTS_DIR, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "region": region,
                    "date": date_key,
                    "count": len(articles),
                    "articles": articles,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"💾 Local backup saved: {path}")

    except Exception as e:
        print("❌ Failed to write local backup:", e)

def normalize_url(base: str, url: str) -> str:
    if not url:
        return ""
    return url if url.startswith("http") else urljoin(base, url)

# --------------------
# NEWS EXTRACTOR
# --------------------
class NewsExtractor:
    def process_url(self, url: str) -> Dict[str, Any]:
        #domain = url.split("//")[-1].split("/")[0].lower()
        parsed = urlparse(url)

        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]

        # take first path segment only (e.g. /en)
        path_parts = [p for p in parsed.path.split("/") if p]
        lang = path_parts[0] if path_parts else ""

        domain = (f"{host}/{lang}" if lang else host).rstrip("/")


        # Normalize domain key
        if domain.startswith("www."):
            domain = domain.replace("www.", "", 1)

        try:
            response = NEWS_SITES_TABLE.get_item(Key={"domain": domain})
            item = response["Item"]

            metadata = {}
            for k, v in item.items():
                if isinstance(v, dict) and "S" in v:
                    metadata[k] = v["S"]
                else:
                    metadata[k] = v

            # ✅ Extract site name from DynamoDB
            metadata["site_name"] = item.get("name") or item.get("site_name") or domain

            return metadata

        except KeyError:
            return {
                "prompt": "Summarize this news page into a JSON object with title and content.",
                "region": "Unknown",
                "site_name": domain  # fallback
            }
def get_existing_urls_for_region(region: str) -> set:
    urls = set()
    try:
        response = REGIONAL_NEWS_TABLE.query(
            KeyConditionExpression=Key("region").eq(region)
        )
        items = response.get("Items", [])
        for item in items:
            for article in item.get("news", []):
                if article.get("url"):
                    urls.add(article["url"])
    except Exception as e:
        print("⚠️ Failed to fetch existing URLs:", e)
    return urls
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

async def translate_article_if_needed(
    title: str,
    details: str
) -> Dict[str, str]:
    """
    Detect language per field and translate non-English parts to English.
    Handles mixed-language (Indonesian / Chinese / English).
    Single Bedrock call.
    """
    prompt = f"""
    You are a professional news translator.

    Tasks:
    1. Detect the language of EACH field separately.
    2. If a field contains any non-English content, translate ONLY that content to English.
    3. Preserve existing English text exactly as-is.
    4. Do NOT summarize, shorten, or rewrite.
    5. Preserve names, numbers, dates, and journalistic tone.

    Return ONLY valid JSON in this exact format:
    {{
      "title": "...",
      "details": "..."
    }}

    Title:
    {title}

    Details:
    {details}
    """

    try:
        response = await bedrock_llm.ainvoke(
            [HumanMessage(content=prompt)]
        )

        content = response.content

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)

        return {
            "title": data.get("title", title),
            "details": data.get("details", details),
        }

    except Exception as e:
        print("⚠️ Translation failed, using original text:", e)
        return {
            "title": title,
            "details": details,
        }


# --------------------
# SCRAPE SITE
# --------------------
async def scrape_site(site_name: str, site_url: str):
    try:
        extractor = NewsExtractor()

        metadata = extractor.process_url(site_url)

        # ================================
        # 🔒 PRELOAD EXISTING URLS (EARLY DEDUP)
        # ================================
        region = metadata.get("region", "Unknown")

        existing_urls = dynamodb_with_retry(
            lambda: get_existing_urls_for_region(region)
        )

        print(f"🔎 Found {len(existing_urls)} existing URLs for this region")


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

        MAX_ARTICLES_PER_SITE = 3
        links = result.get("news", [])[:MAX_ARTICLES_PER_SITE]
        all_articles = []

        for idx, link in enumerate(links, start=1):
            try:
                url = normalize_url(site_url, link.get("url"))

                # 🚫 HARD STOP — already stored
                if url in existing_urls:
                    print(f"    ⏭️  SKIPPED (already stored): {url}")
                    
                    continue

                title = link.get("title")

                # 1️⃣ SCRAPE ARTICLE FIRST
                article_graph = SmartScraperGraph(
                    prompt=metadata["prompt"],
                    source=url,
                    config=GRAPH_CONFIG,
                    schema=ArticleContent,
                )

                article = await asyncio.to_thread(article_graph.run)
                if hasattr(article, "model_dump"):
                    article = article.model_dump()

                # 2️⃣ CLASSIFY USING REAL CONTENT
                categories = await classify_category(
                    article.get("title", title or ""),
                    article.get("details", "")
                )

                if not categories:
                    print(f"    ⏭️  SKIPPED AFTER SCRAPE (no category): {url}")
                    continue

            
               

                article["title"] = article.get("title") or title
                article["category"] = categories
                article["source_url"] = url

                # 🌍 CONDITIONAL TRANSLATION
                if metadata.get("translate_to_en") is True:
                    translated = await translate_article_if_needed(
                        article.get("title", ""),
                        article.get("details", "")
                    )
                    print("🌍 TRANSLATED TITLE:", translated["title"])
                    print("🌍 TRANSLATED DETAILS:", translated["details"][:120], "...")
                    article["title"] = translated["title"]
                    article["details"] = translated["details"]

                all_articles.append(article)


            except Exception:
                traceback.print_exc()



        site_name_from_db = metadata.get("site_name", site_name)

        final_news = [
            {
                "date": a.get("date"),
                "details": a.get("details"),
                "source": site_name_from_db,  # ✅ now uses the actual site name
                "title": a.get("title"),
                "category": a.get("category", []),
                "url": a.get("source_url")
            }
            for a in all_articles
            if a.get("title") or a.get("details")
        ]



        save_articles(
            site_name=metadata.get("site_name", site_name),
            articles=final_news,
            region=metadata.get("region", "Unknown"),
        )

        return final_news

    except Exception:
        traceback.print_exc()
        return []

# --------------------
# API ENDPOINT
# --------------------
async def scrape_background(request: ScrapeRequest):
    try:
        

        parsed = urlparse(request.url)
        site_name = parsed.netloc
        await scrape_site(site_name, request.url)
    except Exception:
        traceback.print_exc()


@app.post("/scrape")
async def run_scrape(request: ScrapeRequest):
    print(f"📥 Scrape job accepted for {request.url}")
    asyncio.create_task(scrape_background(request))
    return {"status": "accepted"}


# --------------------
# ENTRY POINT
# --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
