# worker.py
import sys
import os
import json
import asyncio
import logging
import traceback
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage


# ------------------------------------------------------------------
# LANGCHAIN COMPAT PATCH 
# ------------------------------------------------------------------
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

except ImportError:
    print("langchain-core not installed")
    sys.exit(1)

from scrapegraphai.graphs import SmartScraperGraph
# ------------------------------------------------------------------
# CONFIG & UTILS
# ------------------------------------------------------------------
from config import (
    NEWS_SITES_TABLE,
    REGIONAL_NEWS_TABLE,
    SCRAPER_LLM,
    SCRAPER_FALLBACK_LLM,
    HELPER_LLM,
    TRANSLATOR_LLM,
    GRAPH_CONFIG,
    CATEGORIES,
    MY_TZ,
    MAX_CONCURRENT_SCRAPES,
    MAX_CONCURRENT_LLM,
    MAX_CONCURRENT_ARTICLE,
    MAX_ARTICLES_PER_SITE,
    PROMPT_LIST_ARTICLES,
    PROMPT_EXTRACT_ARTICLE,
    PROMPT_STRUCTURE_ARTICLE,
    PROMPT_CLASSIFY_CATEGORY,
    PROMPT_TRANSLATE_ARTICLE,
    QUEUE_MAXSIZE,
    WORKER_COUNT,
)

from utils import (
    normalize_url,
    normalize_domain,
    normalize_date,
    extract_json,
    dynamodb_with_retry,
    get_existing_urls_for_region,
    save_local_backup,
    llm_with_retry,
)

# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# SEMAPHORES
# ------------------------------------------------------------------
scrape_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)
llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)
article_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ARTICLE)


# ------------------------------------------------------------------
# FASTAPI
# ------------------------------------------------------------------
app = FastAPI(title="Aegis Scraper Worker")


class ScrapeRequest(BaseModel):
    url: str
    prompt: Optional[str] = None
    diff: Optional[str] = None
    full_text: Optional[str] = None


# ------------------------------------------------------------------
# QUEUE
# ------------------------------------------------------------------
scrape_queue: asyncio.Queue[ScrapeRequest] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
# sentinel to signal workers to stop
STOP_SIGNAL = ScrapeRequest(url="__STOP__")



async def scrape_worker(worker_id: int):
    logger.info("Scrape worker %d started", worker_id)
    while True:
        request: ScrapeRequest = await scrape_queue.get()
        if request.url == "__STOP__":
            scrape_queue.task_done()
            logger.info("Worker %d stopping", worker_id)
            break
        try:
            parsed = urlparse(request.url)
            await scrape_site(parsed.netloc, request.url)
        except Exception:
            logger.exception("Worker %d failed to scrape: %s", worker_id, request.url)
        finally:
            scrape_queue.task_done()




# ------------------------------------------------------------------
# SCHEMAS
# ------------------------------------------------------------------
class NewsLink(BaseModel):
    url: str
    title: Optional[str] = None


class NewsData(BaseModel):
    news: List[NewsLink]


class RawArticle(BaseModel):
    title: Optional[str] = None
    text: str
    date: Optional[str] = None


class ArticleContent(BaseModel):
    title: str
    details: str
    date: Optional[str] = None
    category: List[str] = Field(default_factory=list)


# ------------------------------------------------------------------
# SAFE LLM CALL
# ------------------------------------------------------------------
async def llm_call_safe(fn, min_interval: int = 2):
    async with llm_semaphore:
        return await llm_with_retry(fn, min_interval=min_interval)


# ------------------------------------------------------------------
# GRAPH RUNNER
# ------------------------------------------------------------------
async def run_graph_safe(graph):
    async with scrape_semaphore:
        return await asyncio.to_thread(graph.run)


# ------------------------------------------------------------------
# NEWS SITE METADATA
# ------------------------------------------------------------------
class NewsExtractor:
    def process_url(self, url: str) -> Dict[str, Any]:
        parsed = urlparse(url)
        host = normalize_domain(url)

        path_parts = [p for p in parsed.path.split("/") if p]
        first_segment = path_parts[0] if path_parts else ""
        domain_with_path = f"{host}/{first_segment}" if first_segment else host

        for key in (domain_with_path, host):
            try:
                item = NEWS_SITES_TABLE.get_item(Key={"domain": key})["Item"]
                meta = dict(item)

                meta["site_name"] = (
                    item.get("name")
                    or item.get("site_name")
                    or key
                )

                translate_val = item.get("translate_to_en", False)
                if isinstance(translate_val, dict):
                    meta["translate_to_en"] = translate_val.get("BOOL", False)
                else:
                    meta["translate_to_en"] = bool(translate_val)

                return meta

            except Exception:
                continue

        return {
            "prompt": "Summarize this news page into JSON",
            "region": "Unknown",
            "site_name": host,
            "translate_to_en": False,
        }


# ------------------------------------------------------------------
# CATEGORY CLASSIFIER
# ------------------------------------------------------------------
async def classify_category(title: str, snippet: str) -> List[str]:
    prompt = PROMPT_CLASSIFY_CATEGORY.format(
        title=title,
        snippet=snippet,
        categories=CATEGORIES,
    )
    try:
        response = await llm_call_safe(
            lambda: HELPER_LLM.ainvoke([HumanMessage(content=prompt)])
        )
        data = extract_json(response.content)
        return [c for c in data.get("category", []) if c in CATEGORIES]
    except Exception:
        return []


# ------------------------------------------------------------------
# TRANSLATION
# ------------------------------------------------------------------
async def translate_article_if_needed(title: str, details: str) -> Dict[str, str]:
    prompt = PROMPT_TRANSLATE_ARTICLE.format(
        title=title,
        details=details,
    )
    try:
        response = await llm_call_safe(
            lambda: HELPER_LLM.ainvoke([HumanMessage(content=prompt)])
        )
        data = extract_json(response.content)
        return {
            "title": data.get("title", title),
            "details": data.get("details", details),
        }
    except Exception:
        logger.exception("Translation failed")
        return {"title": title, "details": details}


# ------------------------------------------------------------------
# SAVE ARTICLES
# ------------------------------------------------------------------
def save_articles(site_name: str, articles: List[Dict[str, Any]], region: str, existing_urls):
    for a in articles:
        a["date"] = normalize_date(a.get("date"))

    date_key = normalize_date(articles[0]["date"]) if articles else normalize_date(None)

    

    new_articles = [
        a for a in articles if a.get("url") not in existing_urls
    ]

    if not new_articles:
        logger.info("No new articles to insert")
        return

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
        logger.info("Inserted %d new articles", len(new_articles))

    except Exception:
        logger.exception("DynamoDB update failed")
        save_local_backup(region, date_key, new_articles)

# ------------------------------------------------------------------
# ARCTICLE PROCESS
# ------------------------------------------------------------------
async def process_article_link(
    link,
    site_url: str,
    site_name: str,
    metadata: dict,
    existing_urls: set,
):
    async with article_semaphore:
        try:
            url = normalize_url(site_url, link.get("url"))
            if not url or url in existing_urls:
                return None

            raw_graph = SmartScraperGraph(
                prompt=PROMPT_EXTRACT_ARTICLE,
                source=url,
                config={
                    **GRAPH_CONFIG,
                    "llm": {"model_instance": HELPER_LLM, "model_tokens": 2048},
                },
                schema=RawArticle,
            )

            raw = await run_graph_safe(raw_graph)
            raw = raw.model_dump() if hasattr(raw, "model_dump") else raw

            if len((raw.get("text") or "").strip()) < 150:
                return None

            sonnet_prompt = PROMPT_STRUCTURE_ARTICLE.format(
                article_text=raw["text"][:12000]
            )

            try:
                sonnet_resp = await llm_call_safe(
                    lambda: SCRAPER_LLM.ainvoke(
                        [HumanMessage(content=sonnet_prompt)]
                    )
                )
            except Exception:
                sonnet_resp = await llm_call_safe(
                    lambda: SCRAPER_FALLBACK_LLM.ainvoke(
                        [HumanMessage(content=sonnet_prompt)]
                    )
                )

            article = extract_json(sonnet_resp.content)

            categories = await classify_category(
                article.get("title", ""),
                article.get("details", ""),
            )
            if not categories:
                return None

            article["category"] = categories
            article["source"] = metadata.get("site_name", site_name)
            article["url"] = url

            if metadata.get("translate_to_en"):
                translated = await translate_article_if_needed(
                    article["title"],
                    article["details"],
                )
                article.update(translated)

            return article

        except Exception:
            logger.exception("Failed processing article")
            return None

# ------------------------------------------------------------------
# SCRAPE SITE
# ------------------------------------------------------------------
async def scrape_site(site_name: str, site_url: str):
    extractor = NewsExtractor()
    metadata = extractor.process_url(site_url)

    region = metadata.get("region", "Unknown")
    existing_urls = dynamodb_with_retry(
        lambda: get_existing_urls_for_region(region)
    )

    main_graph = SmartScraperGraph(
        prompt=PROMPT_LIST_ARTICLES,
        source=site_url,
        config={
            **GRAPH_CONFIG,
            "llm": {"model_instance": HELPER_LLM, "model_tokens": 4096},
            "depth": 1,
        },
        schema=NewsData,
    )

    result = await run_graph_safe(main_graph)
    result = result.model_dump() if hasattr(result, "model_dump") else result

    links = result.get("news", [])[:MAX_ARTICLES_PER_SITE]

    tasks = [
        process_article_link(
            link=link,
            site_url=site_url,
            site_name=site_name,
            metadata=metadata,
            existing_urls=existing_urls,
        )
        for link in links
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles = [
        r for r in results
        if isinstance(r, dict)
    ]


    save_articles(
        site_name=metadata.get("site_name", site_name),
        articles=articles,
        region=region,
        existing_urls=existing_urls,
    )


    return articles


# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------
async def scrape_background(request: ScrapeRequest):
    parsed = urlparse(request.url)
    await scrape_site(parsed.netloc, request.url)


@app.post("/scrape")
async def run_scrape(request: ScrapeRequest):
    try:
        scrape_queue.put_nowait(request)
        logger.info("Scrape enqueued: %s", request.url)
        return {"status": "queued"}
    except asyncio.QueueFull:
        logger.warning("Queue full, cannot enqueue: %s", request.url)
        return {"status": "queue_full"}, 429


@app.on_event("startup")
async def start_workers():
    for i in range(WORKER_COUNT):
        asyncio.create_task(scrape_worker(i))
    logger.info("All %d scrape workers started", WORKER_COUNT)


@app.on_event("shutdown")
async def stop_workers():
    logger.info("Stopping workers...")
    for _ in range(WORKER_COUNT):
        await scrape_queue.put(STOP_SIGNAL)
    await scrape_queue.join()
    logger.info("All workers stopped")


# ------------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
