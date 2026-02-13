# config.py
import os
from dotenv import load_dotenv
from datetime import timezone, timedelta
import boto3
from langchain_aws import ChatBedrockConverse

# --------------------
# ENV
# --------------------
load_dotenv()

# --------------------
# GENERAL SETTINGS
# --------------------
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-5")

RESULTS_DIR = os.getenv("RESULTS_DIR", "results")

TIMEZONE_OFFSET = int(os.getenv("TIMEZONE_OFFSET", 8))
MY_TZ = timezone(timedelta(hours=TIMEZONE_OFFSET))

MAX_CONCURRENT_SCRAPES = int(os.getenv("MAX_CONCURRENT_SCRAPES", 3))
MAX_CONCURRENT_LLM = int(os.getenv("MAX_CONCURRENT_LLM", 2))
MAX_CONCURRENT_ARTICLE = int(os.getenv("MAX_CONCURRENT_ARTICLE", 3))

MAX_ARTICLES_PER_SITE = int(os.getenv("MAX_ARTICLES_PER_SITE", 5))

QUEUE_MAXSIZE = int(os.getenv("QUEUE_MAXSIZE", 500))  # max number of queued scrape requests
WORKER_COUNT = int(os.getenv("WORKER_COUNT", 2)) 

# --------------------
# DYNAMODB
# --------------------
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

NEWS_SITES_TABLE_NAME = os.getenv(
    "NEWS_SITES_TABLE", "aegis_news_sites"
)
REGIONAL_NEWS_TABLE_NAME = os.getenv(
    "REGIONAL_NEWS_TABLE", "aegis_regional_daily_news"
)

NEWS_SITES_TABLE = dynamodb.Table(NEWS_SITES_TABLE_NAME)
REGIONAL_NEWS_TABLE = dynamodb.Table(REGIONAL_NEWS_TABLE_NAME)

# --------------------
# LLM INSTANCES (BEDROCK)
# --------------------

# Article structuring & summarization
SCRAPER_LLM = ChatBedrockConverse(
    model_id=os.getenv(
        "SCRAPER_LLM_MODEL",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ),
    region_name=AWS_REGION,
    temperature=float(os.getenv("SCRAPER_LLM_TEMPERATURE", 0)),
    max_tokens=int(os.getenv("SCRAPER_LLM_MAX_TOKENS", 4096)),
)

# Fallback if main Sonnet fails
SCRAPER_FALLBACK_LLM = ChatBedrockConverse(
    model_id=os.getenv(
        "SCRAPER_FALLBACK_LLM_MODEL",
        "apac.anthropic.claude-sonnet-4-20250514-v1:0",
    ),
    region_name=AWS_REGION,
    temperature=float(os.getenv("SCRAPER_FALLBACK_LLM_TEMPERATURE", 0)),
    max_tokens=int(os.getenv("SCRAPER_FALLBACK_LLM_MAX_TOKENS", 4096)),
)

# Homepage scraping, raw article extraction, classification
HELPER_LLM = ChatBedrockConverse(
    model_id=os.getenv(
        "HELPER_LLM_MODEL",
        "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    ),
    region_name=AWS_REGION,
    temperature=float(os.getenv("HELPER_LLM_TEMPERATURE", 0)),
    max_tokens=int(os.getenv("HELPER_LLM_MAX_TOKENS", 1024)),
)

# Translation (Amazon Nova)
TRANSLATOR_LLM = ChatBedrockConverse(
    model_id=os.getenv(
        "TRANSLATOR_LLM_MODEL",
        "apac.amazon.nova-micro-v1:0",
    ),
    region_name=AWS_REGION,
    temperature=float(os.getenv("TRANSLATOR_LLM_TEMPERATURE", 0)),
    max_tokens=int(os.getenv("TRANSLATOR_LLM_MAX_TOKENS", 1024)),
)

# --------------------
# SCRAPEGRAPH CONFIG
# --------------------
GRAPH_CONFIG = {
    "headless": True,
    "stealth": True,
    "browser_args": [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
    ],
    "verbose": True,
    "minify_html": True,
}

# --------------------
# NEWS CATEGORIES
# --------------------
CATEGORIES = [
    "Energy",
    "Environment",
    "Economy",
    "Politic",
    "Health",
    "Humanitarian",
    "Geopolitical",
    "Cyber",
    "Space",
    "Maritime Security",
    "International Law",
    "Science",
    "Technology",
    "Terrorism",
    "Transnational Crime",
    "Military",
    "Security",
]

# --------------------
# PROMPTS
# --------------------

# Homepage → article links
PROMPT_LIST_ARTICLES = """
Extract a list of news articles.

Rules:
- Replace all special quotes like “ ” „ with standard double quotes "
- Ensure JSON is valid
Return JSON:
{
  "news": [
    { "url": "...", "title": "..." }
  ]
}
"""

# Article page → raw content (NO summarization)
PROMPT_EXTRACT_ARTICLE = """
Extract ONLY the main news article content.

Rules:
- Remove navigation, ads, footers, related links
- Do NOT summarize or rewrite
- Preserve paragraph breaks
- Return empty text if article body not found
- Replace all special quotes like “ ” „ with standard double quotes "
- Ensure JSON is valid

Return JSON:
{
  "title": "...",
  "text": "...",
  "date": "..."
}
"""

# Raw text → structured article
PROMPT_STRUCTURE_ARTICLE = """
Using ONLY the article text below, return structured JSON.

Rules:
- Do not invent facts
- Do not summarize excessively
- Professional neutral tone

Return JSON:
{{
  "title": "...",
  "details": "...",
  "date": "..."
}}

ARTICLE TEXT:
{article_text}
"""


# Category classification
PROMPT_CLASSIFY_CATEGORY = """
You are a news classifier.
Given a news headline and short description, return a JSON object:
{{"category": []}}

Rules:
- Choose ONLY from this list: {categories}
- Pick 1–5 categories MAX
- If none apply, return an EMPTY ARRAY
- Return ONLY valid JSON

Title: {title}
Snippet: {snippet}
"""

# Translation
PROMPT_TRANSLATE_ARTICLE = """
Translate the following news components into English.

### STRICT RULES:
1. TITLE: Provide a literal, verbatim translation of the headline.
   Do NOT change the word order if it makes sense in English.
   Do NOT add context.
2. DETAILS: Translate the content accurately into professional English.
3. LANGUAGE: If a field is already in English, return it exactly as-is.
4. FORMAT: Return ONLY a valid JSON object. No preamble.

### INPUT:
Title: {title}
Details: {details}

### OUTPUT FORMAT:
{{
  "title": "...",
  "details": "..."
}}
"""
