import sys
import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import traceback
from typing import List, Optional, Dict, Any
import asyncio
from langchain_aws import ChatBedrockConverse
from scrapegraphai.graphs import SmartScraperGraph

# --- STEP 1: LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# Load API keys and other configurations from the environment
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
API_KEY = os.getenv("API_KEY")  # If you have any other keys
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")  # Default model if not specified

# Ensure that environment variables are loaded correctly
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION]):
    print("❌ Missing AWS credentials in environment variables!")
    sys.exit(1)

# --- STEP 2: BEDROCK CONFIGURATION ---
bedrock_llm = ChatBedrockConverse(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name=AWS_DEFAULT_REGION,
    temperature=0,
    max_tokens=8192
)

GRAPH_CONFIG = {
    "llm": {
        "model_instance": bedrock_llm,
        "model_tokens": 8192
    },
    "headless": False,
    "stealth": True,
    "browser_args": ["--no-cache"],
    "verbose": True
}

# --- STEP 3: FASTAPI SETUP ---
app = FastAPI(title="Aegis Scraper Worker")

# --- STEP 4: DEFINE REQUEST AND RESPONSE SCHEMAS ---

class ScrapeRequest(BaseModel):
    url: str
    prompt: str
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
            "tempo.co": {
                "prompt": "Extract the headline and body text from this Tempo article.",
                "model": "openai/gpt-4"
            },
            "en.antaranews.com": {
                "prompt": "Extract the headline and body text from this Antara News article.",
                "model": "openai/gpt-4"
            },
            "thejakartapost.com": {
                "prompt": "Extract the headline and body text from this Jakarta Post article.",
                "model": "openai/gpt-4"
            },

            "nst.com.my": {
                "prompt": "Extract the headline and body text from this New Straits Times article.",
                "model": "openai/gpt-4"
            },
            "freemalaysiatoday.com": {
                "prompt": "Extract the headline and body text from this Free Malaysia Today article.",
                "model": "openai/gpt-4"
            },
            "bernama.com": {
                "prompt": "Extract the headline and body text from this Bernama article.",
                "model": "openai/gpt-4"
            },
            "theedgemarkets.com": {
                "prompt": "Extract the headline and body text from this Edge Markets article.",
                "model": "openai/gpt-4"
            },

            "straitstimes.com": {
                "prompt": "Extract the headline and body text from this Straits Times article.",
                "model": "openai/gpt-4"
            },
            "todayonline.com": {
                "prompt": "Extract the headline and body text from this Today Online article.",
                "model": "openai/gpt-4"
            },
            "mothership.sg": {
                "prompt": "Extract the headline and body text from this Mothership article.",
                "model": "openai/gpt-4"
            },
            "businesstimes.com.sg": {
                "prompt": "Extract the headline and body text from this Business Times article.",
                "model": "openai/gpt-4"
            },

            "bangkokpost.com": {
                "prompt": "Extract the headline and body text from this Bangkok Post article.",
                "model": "openai/gpt-4"
            },
            "thaipbsworld.com": {
                "prompt": "Extract the headline and body text from this Thai PBS World article.",
                "model": "openai/gpt-4"
            },
            "prachatai.com": {
                "prompt": "Extract the headline and body text from this Prachatai English article.",
                "model": "openai/gpt-4"
            },

            "inquirer.net": {
                "prompt": "Extract the headline and body text from this Philippine Daily Inquirer article.",
                "model": "openai/gpt-4"
            },
            "news.abs-cbn.com": {
                "prompt": "Extract the headline and body text from this ABS-CBN News article.",
                "model": "openai/gpt-4"
            },
            "gmanetwork.com": {
                "prompt": "Extract the headline and body text from this GMA News article.",
                "model": "openai/gpt-4"
            },
            "rappler.com": {
                "prompt": "Extract the headline and body text from this Rappler article.",
                "model": "openai/gpt-4"
            },

            "vietnamnews.vn": {
                "prompt": "Extract the headline and body text from this Vietnam News article.",
                "model": "openai/gpt-4"
            },
            "english.vov.vn": {
                "prompt": "Extract the headline and body text from this VOV World article.",
                "model": "openai/gpt-4"
            },
            "tuoitrenews.vn": {
                "prompt": "Extract the headline and body text from this Tuoi Tre News article.",
                "model": "openai/gpt-4"
            },

            "phnompenhpost.com": {
                "prompt": "Extract the headline and body text from this Phnom Penh Post article.",
                "model": "openai/gpt-4"
            },
            "cambodiachinapress.com": {
                "prompt": "Extract the headline and body text from this Cambodia China Times article.",
                "model": "openai/gpt-4"
            },

            "myanmar-now.org": {
                "prompt": "Extract the headline and body text from this Myanmar Now article.",
                "model": "openai/gpt-4"
            },
            "frontiermyanmar.net": {
                "prompt": "Extract the headline and body text from this Frontier Myanmar article.",
                "model": "openai/gpt-4"
            },
            "irrawaddy.com": {
                "prompt": "Extract the headline and body text from this Irrawaddy article.",
                "model": "openai/gpt-4"
            },

            "laotiantimes.com": {
                "prompt": "Extract the headline and body text from this Laotian Times article.",
                "model": "openai/gpt-4"
            },
            "vientianetimes.org.la": {
                "prompt": "Extract the headline and body text from this Vientiane Times article.",
                "model": "openai/gpt-4"
            },

            "borneobulletin.com.bn": {
                "prompt": "Extract the headline and body text from this Borneo Bulletin article.",
                "model": "openai/gpt-4"
            },
            "brudirect.com": {
                "prompt": "Extract the headline and body text from this Brunei Darussalam News article.",
                "model": "openai/gpt-4"
            },

            "tatoli.tl": {
                "prompt": "Extract the headline and body text from this Tatoli News Agency article.",
                "model": "openai/gpt-4"
            },
            "timorpost.com": {
                "prompt": "Extract the headline and body text from this Timor Post article.",
                "model": "openai/gpt-4"
            },

            "tass.com": {
                "prompt": "Extract the headline and body text from this TASS article.",
                "model": "openai/gpt-4"
            },
            "themoscowtimes.com": {
                "prompt": "Extract the headline and body text from this Moscow Times article.",
                "model": "openai/gpt-4"
            },
            "rt.com": {
                "prompt": "Extract the headline and body text from this RT article.",
                "model": "openai/gpt-4"
            },
            "interfax.ru": {
                "prompt": "Extract the headline and body text from this Interfax article.",
                "model": "openai/gpt-4"
            },
            "kommersant.ru": {
                "prompt": "Extract the headline and body text from this Kommersant article.",
                "model": "openai/gpt-4"
            },

            "kyivpost.com": {
                "prompt": "Extract the headline and body text from this Kyiv Post article.",
                "model": "openai/gpt-4"
            },
            "kyivindependent.com": {
                "prompt": "Extract the headline and body text from this Kyiv Independent article.",
                "model": "openai/gpt-4"
            },
            "ukrinform.net": {
                "prompt": "Extract the headline and body text from this Ukrinform article.",
                "model": "openai/gpt-4"
            },
            "unian.info": {
                "prompt": "Extract the headline and body text from this UNIAN article.",
                "model": "openai/gpt-4"
            },
            "pravda.com.ua": {
                "prompt": "Extract the headline and body text from this Ukrainska Pravda article.",
                "model": "openai/gpt-4"
            },

            "inform.kz": {
                "prompt": "Extract the headline and body text from this Inform.kz article.",
                "model": "openai/gpt-4"
            },
            "kazinform.kz": {
                "prompt": "Extract the headline and body text from this Kazinform article.",
                "model": "openai/gpt-4"
            },
            "astanatimes.com": {
                "prompt": "Extract the headline and body text from this Astana Times article.",
                "model": "openai/gpt-4"
            },
            "loopcentralasia.com": {
                "prompt": "Extract the headline and body text from this Loop Central Asia article.",
                "model": "openai/gpt-4"
            },
            "azattyq.org": {
                "prompt": "Extract the headline and body text from this Azattyq article.",
                "model": "openai/gpt-4"
            },

            "eng.belta.by": {
                "prompt": "Extract the headline and body text from this BelTA article.",
                "model": "openai/gpt-4"
            },
            "rbc.ru": {
                "prompt": "Extract the headline and body text from this RBC article.",
                "model": "openai/gpt-4"
            },
            "charter97.org": {
                "prompt": "Extract the headline and body text from this Charter97 article.",
                "model": "openai/gpt-4"
            },
            "spring96.org": {
                "prompt": "Extract the headline and body text from this Viasna / Spring96 article.",
                "model": "openai/gpt-4"
            },
            "belaruspartisan.org": {
                "prompt": "Extract the headline and body text from this Belarus Partisan article.",
                "model": "openai/gpt-4"
            },

            "agenda.ge": {
                "prompt": "Extract the headline and body text from this Agenda.ge article.",
                "model": "openai/gpt-4"
            },
            "georgiatoday.ge": {
                "prompt": "Extract the headline and body text from this Georgia Today article.",
                "model": "openai/gpt-4"
            },
            "interpressnews.ge": {
                "prompt": "Extract the headline and body text from this InterPressNews article.",
                "model": "openai/gpt-4"
            },
            "civil.ge": {
                "prompt": "Extract the headline and body text from this Civil.ge article.",
                "model": "openai/gpt-4"
            },
            "messenger.com.ge": {
                "prompt": "Extract the headline and body text from this Messenger article.",
                "model": "openai/gpt-4"
            },

            "armenpress.am": {
                "prompt": "Extract the headline and body text from this Armenpress article.",
                "model": "openai/gpt-4"
            },
            "arminfo.info": {
                "prompt": "Extract the headline and body text from this Arminfo article.",
                "model": "openai/gpt-4"
            },
            "panorama.am": {
                "prompt": "Extract the headline and body text from this Panorama article.",
                "model": "openai/gpt-4"
            },
            "mediamax.am": {
                "prompt": "Extract the headline and body text from this Mediamax article.",
                "model": "openai/gpt-4"
            },
            "tert.am": {
                "prompt": "Extract the headline and body text from this Tert article.",
                "model": "openai/gpt-4"
            },

            "azernews.az": {
                "prompt": "Extract the headline and body text from this AzerNews article.",
                "model": "openai/gpt-4"
            },
            "azerbaijanli.com": {
                "prompt": "Extract the headline and body text from this Azerbaijanli article.",
                "model": "openai/gpt-4"
            },
            "1news.az": {
                "prompt": "Extract the headline and body text from this 1News article.",
                "model": "openai/gpt-4"
            },
            "trend.az": {
                "prompt": "Extract the headline and body text from this Trend News article.",
                "model": "openai/gpt-4"
            },
            "report.az": {
                "prompt": "Extract the headline and body text from this Report.az article.",
                "model": "openai/gpt-4"
            },

            "gazeta.uz": {
                "prompt": "Extract the headline and body text from this Gazeta.uz article.",
                "model": "openai/gpt-4"
            },
            "kun.uz": {
                "prompt": "Extract the headline and body text from this Kun.uz article.",
                "model": "openai/gpt-4"
            },
            "mail.ru": {
                "prompt": "Extract the headline and body text from this Mail.ru article.",
                "model": "openai/gpt-4"
            },

            "24.kg": {
                "prompt": "Extract the headline and body text from this 24.kg article.",
                "model": "openai/gpt-4"
            },
            "knews.kg": {
                "prompt": "Extract the headline and body text from this KNews article.",
                "model": "openai/gpt-4"
            },
            "akipress.org": {
                "prompt": "Extract the headline and body text from this AKIpress article.",
                "model": "openai/gpt-4"
            },
            "kg.akipress.org": {
                "prompt": "Extract the headline and body text from this AKIpress KG article.",
                "model": "openai/gpt-4"
            },
            "vb.kg": {
                "prompt": "Extract the headline and body text from this Vecherniy Bishkek article.",
                "model": "openai/gpt-4"
            },

            "turkmenistan.gov.tm": {
                "prompt": "Extract the headline and body text from this Government of Turkmenistan article.",
                "model": "openai/gpt-4"
            },
            "hronikatm.com": {
                "prompt": "Extract the headline and body text from this Hronika TM article.",
                "model": "openai/gpt-4"
            },
            "trend.az": {
                "prompt": "Extract the headline and body text from this Trend (TM) article.",
                "model": "openai/gpt-4"
            },
            "turkmen.news": {
                "prompt": "Extract the headline and body text from this Turkmen News article.",
                "model": "openai/gpt-4"
            },
            "centralasia.news": {
                "prompt": "Extract the headline and body text from this Central Asia News article.",
                "model": "openai/gpt-4"
            }
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
        # Initialize NewsExtractor
        print(f"🧩 Initializing NewsExtractor for {site_name}")
        news_extractor = NewsExtractor(config_path="config.yaml")
        
        # Get metadata (prompt and model) using NewsExtractor
        metadata = news_extractor.process_url(site_url)
        prompt = metadata["prompt"]
        model = metadata["model"]
        
        print(f"\n🔎 Scraping {site_name} ({site_url}) with prompt: {prompt}")

        # Run the scraper for the site
        main_graph = SmartScraperGraph(
            prompt=prompt,
            source=site_url,
            config={**GRAPH_CONFIG, "depth": 1},
            schema=NewsData,
        )

        print(f"🔄 Running scraper graph for {site_name}")
        result = await asyncio.to_thread(main_graph.run)

        print(f"\n📄 {site_name} RESULT:")
        print(json.dumps(result, indent=4))

        urls = [item["url"] for item in result["news"]]
        urls = urls[:1]  # Example: just get 1 article for now

        print(f"\n✅ Found {len(urls)} articles\n")

        # Scrape individual articles
        all_articles = []

        for i, url in enumerate(urls, 1):
            print(f"➡️  Article {i}: {url}")

            try:
                article_graph = SmartScraperGraph(
                    prompt=(metadata["prompt"]),
                    source=url,
                    config=GRAPH_CONFIG,
                    schema=ArticleContent,
                )

                print(f"🔄 Running article graph for {url}")
                article = await asyncio.to_thread(article_graph.run)

                article["source_url"] = url
                all_articles.append(article)

                print(f"✅ Headline: {article.get('headline')}\n")

            except Exception as e:
                print(f"❌ Failed to scrape article: {e}")
                traceback.print_exc()

        # Return all articles for debugging
        return all_articles

    except Exception as e:
        print(f"❌ Failed to scrape site {site_name}: {e}")
        traceback.print_exc()


# --- STEP 7: API ENDPOINT ---

@app.post("/scrape")
async def run_scrape(request: ScrapeRequest):
    try:
        all_scraped_articles = []

        # Scrape the site passed in the request
        print(f"🔍 Starting scrape for {request.url}")
        articles = await scrape_site("Custom Site", request.url)
        all_scraped_articles.extend(articles)

        print("\n🎉 Final Scraped Articles:")
        return {"status": "success", "data": all_scraped_articles}

    except Exception as e:
        print("❌ FATAL ERROR in main scraping process")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- ENTRY POINT ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Runs the FastAPI app on port 8001
