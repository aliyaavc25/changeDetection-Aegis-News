# utils.py
import os
import re
import json
import time
import asyncio
import logging
from typing import Callable, Any, Set
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

from boto3.dynamodb.conditions import Key

from config import (
    MY_TZ,
    RESULTS_DIR,
    REGIONAL_NEWS_TABLE,
)

logger = logging.getLogger(__name__)

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------------
# DATE / TIME
# -------------------------------------------------------------------
def normalize_date(date_str: str | None) -> str:
    """
    Convert ISO or relative date strings into YYYY-MM-DD (MY_TZ).
    Supports:
    - ISO 8601
    - yesterday
    - X hours ago
    - X days ago
    """
    now = datetime.now(MY_TZ)

    if not date_str:
        return now.strftime("%Y-%m-%d")

    date_str = date_str.lower().strip()

    # ISO first
    try:
        dt = datetime.fromisoformat(date_str.replace("z", "+00:00"))
        return dt.astimezone(MY_TZ).strftime("%Y-%m-%d")
    except Exception:
        pass

    if "yesterday" in date_str:
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")

    match = re.match(r"(\d+)\s*hour", date_str)
    if match:
        return (now - timedelta(hours=int(match.group(1)))).strftime("%Y-%m-%d")

    match = re.match(r"(\d+)\s*day", date_str)
    if match:
        return (now - timedelta(days=int(match.group(1)))).strftime("%Y-%m-%d")

    return now.strftime("%Y-%m-%d")


# -------------------------------------------------------------------
# URL HELPERS
# -------------------------------------------------------------------
def normalize_url(base: str, url: str | None) -> str:
    if not url:
        return ""
    return url if url.startswith("http") else urljoin(base, url)


def normalize_domain(url: str) -> str:
    """
    Return host-only domain without www or path.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


# -------------------------------------------------------------------
# JSON / LLM HELPERS
# -------------------------------------------------------------------
def extract_json(content: str) -> dict:
    """
    Extract and parse the first JSON object or array from LLM output.
    """
    if not content:
        raise ValueError("Empty LLM response")

    match = re.search(r"(\{.*?\}|\[.*?\])", content, re.S)
    if not match:
        raise ValueError("No JSON found in response")

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decode error: {e}")


# -------------------------------------------------------------------
# DYNAMODB HELPERS
# -------------------------------------------------------------------
def dynamodb_with_retry(
    fn: Callable[[], Any],
    retries: int = 3,
    delay: int = 1,
) -> Any:
    """
    Retry wrapper for DynamoDB operations.
    """
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            logger.warning(
                "DynamoDB attempt %d failed: %s",
                attempt,
                e,
            )
            if attempt < retries:
                time.sleep(delay * attempt)

    raise last_exc


def get_existing_urls_for_region(region: str) -> Set[str]:
    """
    Fetch ALL existing article URLs for a region (full-region dedup).
    """
    urls: Set[str] = set()

    try:
        response = REGIONAL_NEWS_TABLE.query(
            KeyConditionExpression=Key("region").eq(region)
        )

        for item in response.get("Items", []):
            for article in item.get("news", []):
                url = article.get("url")
                if url:
                    urls.add(url)

    except Exception as e:
        logger.warning(
            "Failed to fetch existing URLs for region '%s': %s",
            region,
            e,
        )

    return urls


# -------------------------------------------------------------------
# BACKUP
# -------------------------------------------------------------------
def save_local_backup(
    region: str,
    date_key: str,
    articles: list,
) -> None:
    """
    Write failed DynamoDB inserts to local disk as JSON.
    """
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

        logger.info("Local backup saved: %s", path)

    except Exception:
        logger.exception("Failed to write local backup")


# -------------------------------------------------------------------
# ASYNC LLM RETRY
# -------------------------------------------------------------------
async def llm_with_retry(
    fn: Callable[[], Any],
    retries: int = 3,
    delay: int = 1,
    min_interval: int = 2,
) -> Any:
    """
    Retry wrapper for async LLM calls with backoff and rate spacing.
    """
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            result = await fn()
            await asyncio.sleep(min_interval)
            return result

        except Exception as e:
            last_exc = e

            if "Too many tokens per day" in str(e):
                logger.warning("Daily token quota exhausted, aborting retries")
                raise

            logger.warning(
                "LLM attempt %d failed: %s",
                attempt,
                e,
            )

            if attempt < retries:
                await asyncio.sleep(delay * attempt)

    raise last_exc
