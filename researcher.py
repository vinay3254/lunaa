"""
researcher.py
=============
Fully autonomous trading research agent module.

Handles:
  - News aggregation from multiple RSS feeds + CryptoPanic API
  - Sentiment scoring (keyword-based)
  - Fear & Greed index (CNN)
  - FRED macro-economic data (CPI, PCE, Fed Funds, Unemployment, GDP, M2)
  - US Treasury yield curve
  - Economic calendar (ForexFactory scrape with hardcoded fallback)

All network calls are wrapped in try/except blocks so a failure in one
source never crashes the overall pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import requests
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT: int = 15          # seconds for every HTTP request
_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

RSS_FEEDS: dict[str, str] = {
    "Reuters":        "https://feeds.reuters.com/reuters/businessNews",
    "CNBC":           "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "Yahoo Finance":  "https://finance.yahoo.com/news/rssindex",
    "Investing.com":  "https://www.investing.com/rss/news.rss",
}

# Sentiment keyword lists
_BULLISH_WORDS: list[str] = [
    "surge", "rally", "gain", "rise", "bull", "positive", "growth",
    "record", "beat", "strong", "upgrade", "outperform", "buy",
    "high", "breakout",
]
_BEARISH_WORDS: list[str] = [
    "crash", "fall", "drop", "decline", "bear", "negative", "loss",
    "miss", "weak", "downgrade", "sell", "low", "fear", "risk",
    "collapse", "plunge", "recession",
]

# FRED series to fetch
_FRED_SERIES: dict[str, str] = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi":            "CPIAUCSL",
    "pce":            "PCEPI",
    "unemployment":   "UNRATE",
    "gdp":            "GDP",
    "m2":             "M2SL",
}

# CNN Fear & Greed endpoint
_CNN_FEAR_GREED_URL: str = (
    "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
)

# ForexFactory calendar
_FOREXFACTORY_CALENDAR_URL: str = "https://www.forexfactory.com/calendar"

# US Treasury yield curve XML template (YYYYMM filled at runtime)
_TREASURY_XML_URL_TEMPLATE: str = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/pages/xml?data=daily_treasury_yield_curve"
    "&field_tdr_date_value={yyyymm}"
)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _safe_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> requests.Response | None:
    """Perform a GET request and return the Response or None on failure."""
    try:
        resp = requests.get(
            url,
            headers=headers or _DEFAULT_HEADERS,
            params=params,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching %s", url)
    except requests.exceptions.ConnectionError as exc:
        logger.warning("Connection error fetching %s: %s", url, exc)
    except requests.exceptions.HTTPError as exc:
        logger.warning("HTTP %s error fetching %s: %s", exc.response.status_code, url, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error fetching %s: %s", url, exc)
    return None


def _article_id(title: str, link: str) -> str:
    """Stable deduplication key for a news article."""
    raw = (title.strip().lower() + link.strip().lower()).encode()
    return hashlib.md5(raw).hexdigest()  # noqa: S324


def _parse_date(date_str: str | None) -> datetime | None:
    """
    Try multiple common date formats and return an aware UTC datetime,
    or None if parsing fails.
    """
    if not date_str:
        return None

    # feedparser already handles RFC-2822; accept time.struct_time too
    if isinstance(date_str, time.struct_time):
        try:
            return datetime(*date_str[:6], tzinfo=timezone.utc)
        except Exception:  # noqa: BLE001
            return None

    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# RSS Feed parsing
# ---------------------------------------------------------------------------


def fetch_rss_feed(url: str, source_name: str) -> list[dict]:
    """
    Fetch and parse a single RSS feed.

    Parameters
    ----------
    url : str
        Full URL of the RSS/Atom feed.
    source_name : str
        Human-readable label (e.g. "Reuters").

    Returns
    -------
    list[dict]
        Each dict has keys: title, link, published, source, summary.
        Returns an empty list on any failure.
    """
    logger.info("Fetching RSS feed: %s (%s)", source_name, url)
    articles: list[dict] = []

    try:
        # feedparser handles network retrieval internally; supply a custom
        # agent to avoid 403 responses from some endpoints.
        parsed = feedparser.parse(
            url,
            agent=_USER_AGENT,
            request_headers={"Accept": "application/rss+xml, application/xml, text/xml"},
        )

        if parsed.bozo and not parsed.entries:
            # bozo=True means the feed is malformed; still try entries
            logger.warning(
                "Feed %s has bozo flag: %s", source_name, parsed.bozo_exception
            )

        for entry in parsed.entries:
            title: str = entry.get("title", "").strip()
            link: str = entry.get("link", "").strip()
            summary: str = entry.get("summary", entry.get("description", "")).strip()

            # Strip HTML tags from summary
            summary = re.sub(r"<[^>]+>", " ", summary).strip()
            summary = re.sub(r"\s+", " ", summary)

            # Resolve publish date
            pub_struct = entry.get("published_parsed") or entry.get("updated_parsed")
            if pub_struct:
                published_dt = datetime(*pub_struct[:6], tzinfo=timezone.utc)
            else:
                raw_pub = entry.get("published") or entry.get("updated") or ""
                published_dt = _parse_date(raw_pub) or _now_utc()

            articles.append(
                {
                    "id":        _article_id(title, link),
                    "title":     title,
                    "link":      link,
                    "published": published_dt.isoformat(),
                    "source":    source_name,
                    "summary":   summary[:500],  # cap summary length
                }
            )

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse RSS feed %s: %s", source_name, exc)

    logger.info("  → %d articles from %s", len(articles), source_name)
    return articles


def fetch_all_news() -> list[dict]:
    """
    Aggregate news from every configured RSS feed.

    Deduplicates by (title, link) MD5 hash and sorts newest-first.

    Returns
    -------
    list[dict]
        Unified, deduplicated, recency-sorted article list.
    """
    logger.info("Aggregating news from %d RSS feeds…", len(RSS_FEEDS))
    seen_ids: set[str] = set()
    all_articles: list[dict] = []

    for source_name, url in RSS_FEEDS.items():
        articles = fetch_rss_feed(url, source_name)
        for article in articles:
            aid = article["id"]
            if aid not in seen_ids:
                seen_ids.add(aid)
                all_articles.append(article)

    # Sort newest first (ISO strings sort lexicographically when zero-padded)
    all_articles.sort(key=lambda a: a.get("published", ""), reverse=True)
    logger.info("Total unique articles after deduplication: %d", len(all_articles))
    return all_articles


# ---------------------------------------------------------------------------
# CryptoPanic news
# ---------------------------------------------------------------------------


def fetch_cryptopanic_news(api_key: str | None) -> list[dict]:
    """
    Fetch crypto-specific news from the CryptoPanic API.

    Parameters
    ----------
    api_key : str | None
        CryptoPanic API key.  If None or empty the function returns [].

    Returns
    -------
    list[dict]
        Articles with keys: title, link, published, source, summary, currencies.
    """
    if not api_key:
        logger.info("CryptoPanic API key not set – skipping crypto news.")
        return []

    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": api_key,
        "kind":       "news",
        "public":     "true",
        "filter":     "hot",
    }

    logger.info("Fetching CryptoPanic news…")
    resp = _safe_get(url, params=params)
    if resp is None:
        return []

    articles: list[dict] = []
    try:
        data = resp.json()
        for post in data.get("results", []):
            title = post.get("title", "").strip()
            link = post.get("url", post.get("slug", "")).strip()
            pub_raw = post.get("published_at", post.get("created_at", ""))
            published_dt = _parse_date(pub_raw) or _now_utc()

            currencies = [
                c.get("code", "") for c in post.get("currencies", [])
            ]
            source_title = (
                post.get("source", {}).get("title", "CryptoPanic")
                if isinstance(post.get("source"), dict)
                else "CryptoPanic"
            )

            articles.append(
                {
                    "id":         _article_id(title, link),
                    "title":      title,
                    "link":       link,
                    "published":  published_dt.isoformat(),
                    "source":     source_title,
                    "summary":    "",
                    "currencies": currencies,
                }
            )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error("Failed to parse CryptoPanic response: %s", exc)

    logger.info("CryptoPanic: %d articles fetched", len(articles))
    return articles


# ---------------------------------------------------------------------------
# NLP Sentiment Scoring Pipeline (FinBERT + VADER Fallback)
# ---------------------------------------------------------------------------

FINBERT_PIPE = None
VADER_ANALYZER = None
NLP_MODEL_LOADED = False

def init_nlp_sentiment():
    """Initialize FinBERT or VADER model for high-fidelity news sentiment analysis."""
    global FINBERT_PIPE, VADER_ANALYZER, NLP_MODEL_LOADED
    # 1. Try Loading FinBERT via transformers
    try:
        from transformers import pipeline
        import torch
        logger.info("Initializing FinBERT model (ProsusAI/finbert)...")
        # pipeline automatically handles CPU/GPU device selection
        device = 0 if torch.cuda.is_available() else -1
        FINBERT_PIPE = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
        NLP_MODEL_LOADED = True
        logger.info("FinBERT successfully loaded and ready for inference.")
    except Exception as e:
        logger.warning("HuggingFace transformers or ProsusAI/finbert not loaded/available: %s. Falling back to VADER.", e)
        
    # 2. Setup VADER as fallback
    if not NLP_MODEL_LOADED:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            VADER_ANALYZER = SentimentIntensityAnalyzer()
            logger.info("VADER Sentiment Analyzer loaded successfully.")
        except Exception as e:
            logger.error("VADER Sentiment Analyzer not available: %s. Falling back to rule-based keyword match.", e)


def get_headline_nlp_sentiment(headline: str) -> float:
    """Analyze single headline sentiment using FinBERT, VADER, or keyword matching."""
    global FINBERT_PIPE, VADER_ANALYZER, NLP_MODEL_LOADED
    
    if not NLP_MODEL_LOADED and VADER_ANALYZER is None:
        init_nlp_sentiment()
        
    if NLP_MODEL_LOADED and FINBERT_PIPE is not None:
        try:
            res = FINBERT_PIPE(headline[:512])[0]
            label = res["label"].lower()
            confidence = float(res["score"])
            if label == "positive":
                return confidence
            elif label == "negative":
                return -confidence
            else:
                return 0.0
        except Exception as exc:
            logger.warning("FinBERT inference failed for headline '%s': %s. Falling back to VADER.", headline, exc)
            
    # Fallback to VADER
    if VADER_ANALYZER is not None:
        try:
            res = VADER_ANALYZER.polarity_scores(headline)
            return float(res.get("compound", 0.0))
        except Exception as exc:
            logger.error("VADER inference failed for headline '%s': %s. Falling back to Keyword Matching.", headline, exc)
            
    # Fallback to Keyword Matching
    return scale_sentiment([headline])


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------


def scale_sentiment(headline_list: list[str]) -> float:
    """
    Compute an aggregate sentiment score for a list of headlines.

    Algorithm
    ---------
    For each headline:
      - Tokenise to lowercase words.
      - Count bullish and bearish keyword hits.
    Aggregate score = (total_bullish - total_bearish) / max(total_words / 10, 1)
    Clipped to [-1.0, +1.0].

    Parameters
    ----------
    headline_list : list[str]
        Raw headline strings.

    Returns
    -------
    float
        Sentiment in [-1.0, +1.0].  0.0 for empty input.
    """
    if not headline_list:
        return 0.0

    total_bullish = 0
    total_bearish = 0
    total_words = 0

    for headline in headline_list:
        if not headline:
            continue
        # Tokenise: lowercase alpha tokens only
        tokens = re.findall(r"[a-z]+", headline.lower())
        total_words += len(tokens)
        for token in tokens:
            if token in _BULLISH_WORDS:
                total_bullish += 1
            if token in _BEARISH_WORDS:
                total_bearish += 1

    if total_words == 0:
        return 0.0

    denominator = max(total_words / 10.0, 1.0)
    raw_score = (total_bullish - total_bearish) / denominator
    clipped = max(-1.0, min(1.0, raw_score))
    return round(clipped, 4)


SOURCE_CREDIBILITY = {
    # Tier 1 — highest credibility, weight 1.0
    "reuters.com": 1.0,
    "ft.com": 1.0,
    "wsj.com": 1.0,
    "bloomberg.com": 1.0,
    "apnews.com": 1.0,
    "federalreserve.gov": 1.0,
    "bis.org": 1.0,

    # Tier 2 — good credibility, weight 0.7
    "cnbc.com": 0.7,
    "marketwatch.com": 0.7,
    "economist.com": 0.7,
    "investing.com": 0.7,
    "seekingalpha.com": 0.7,
    "thehindu.com": 0.7,
    "livemint.com": 0.7,
    "economictimes.com": 0.7,

    # Tier 3 — moderate credibility, weight 0.4
    "yahoo.finance.com": 0.4,
    "cryptopanic.com": 0.4,
    "coindesk.com": 0.4,
    "cointelegraph.com": 0.4,
    "benzinga.com": 0.4,

    # Tier 4 — low credibility, weight 0.2
    "reddit.com": 0.2,
    "twitter.com": 0.2,
    "unknown": 0.1
}

def get_credibility_weight(source: str, link: str) -> float:
    source_lower = source.lower()
    link_lower = link.lower()
    
    if "reuters" in source_lower or "reuters.com" in link_lower:
        return 1.0
    if "ft.com" in link_lower or "financial times" in source_lower or "ft.com" in source_lower:
        return 1.0
    if "wsj" in source_lower or "wsj.com" in link_lower:
        return 1.0
    if "bloomberg" in source_lower or "bloomberg.com" in link_lower:
        return 1.0
    if "apnews" in source_lower or "apnews.com" in link_lower:
        return 1.0
    if "federalreserve" in source_lower or "federalreserve.gov" in link_lower:
        return 1.0
    if "bis.org" in link_lower or "bis.org" in source_lower:
        return 1.0
        
    if "cnbc" in source_lower or "cnbc.com" in link_lower:
        return 0.7
    if "marketwatch" in source_lower or "marketwatch.com" in link_lower:
        return 0.7
    if "economist" in source_lower or "economist.com" in link_lower:
        return 0.7
    if "investing.com" in link_lower or "investing" in source_lower:
        return 0.7
    if "seekingalpha" in source_lower or "seekingalpha.com" in link_lower:
        return 0.7
    if "thehindu" in source_lower or "thehindu.com" in link_lower:
        return 0.7
    if "livemint" in source_lower or "livemint.com" in link_lower:
        return 0.7
    if "economictimes" in source_lower or "economictimes.com" in link_lower or "economic times" in source_lower:
        return 0.7
        
    if "yahoo" in source_lower or "yahoo.finance.com" in link_lower or "yahoo.com" in link_lower:
        return 0.4
    if "cryptopanic" in source_lower or "cryptopanic.com" in link_lower:
        return 0.4
    if "coindesk" in source_lower or "coindesk.com" in link_lower:
        return 0.4
    if "cointelegraph" in source_lower or "cointelegraph.com" in link_lower:
        return 0.4
    if "benzinga" in source_lower or "benzinga.com" in link_lower:
        return 0.4
        
    if "reddit" in source_lower or "reddit.com" in link_lower:
        return 0.2
    if "twitter" in source_lower or "twitter.com" in link_lower or "x.com" in link_lower:
        return 0.2
        
    return 0.1

def calculate_asset_sentiment(
    asset_name: str,
    ticker: str,
    news_list: list[dict],
    return_dict: bool = False,
    asset_class: str = "stock",
) -> float | dict:
    """
    Filter ``news_list`` to articles relevant to *asset_name* / *ticker*,
    then return a credibility-weighted sentiment score in [-1.0, +1.0] using NLP.
    Also fetches and integrates Reddit PRAW sentiment if configured.
    """
    name_lower = asset_name.lower()
    ticker_lower = ticker.lower()
    
    # Collect all news items mentioning that asset in last 24h
    relevant_articles = []
    now = datetime.now(timezone.utc)
    
    for article in news_list:
        title = article.get("title", "")
        summary = article.get("summary", "")
        combined = (title + " " + summary).lower()
        
        # Check relevance
        if name_lower in combined or ticker_lower in combined:
            # Check publish date is within last 24h
            pub_str = article.get("published", "")
            try:
                pub_dt = datetime.fromisoformat(pub_str)
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                age_hours = (now - pub_dt).total_seconds() / 3600.0
            except Exception:
                age_hours = 0.0 # fallback if unparseable
                
            if age_hours <= 24.0:
                relevant_articles.append(article)

    # If no articles in last 24h, fall back to all news items in the feed
    if not relevant_articles:
        for article in news_list:
            title = article.get("title", "")
            summary = article.get("summary", "")
            combined = (title + " " + summary).lower()
            if name_lower in combined or ticker_lower in combined:
                relevant_articles.append(article)

    # --- Reddit PRAW Integration ---
    reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
    reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if reddit_client_id and reddit_client_secret:
        try:
            import praw
            reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=os.getenv("REDDIT_USER_AGENT", "TradingResearchAgent/1.0"),
                requestor_kwargs={"timeout": 10}
            )
            
            subreddits = ["stocks", "investing"]
            if asset_class == "crypto":
                subreddits = ["CryptoCurrency", "Bitcoin"]
            elif asset_class == "stock":
                subreddits = ["wallstreetbets", "stocks", "investing"]
                
            query = f"{ticker} OR {asset_name}"
            logger.info("Searching Reddit subreddits %s for '%s'...", subreddits, query)
            
            for sub_name in subreddits:
                subreddit = reddit.subreddit(sub_name)
                for post in subreddit.search(query, time_filter="day", limit=10):
                    title = post.title
                    selftext = post.selftext or ""
                    ups = post.score
                    # Append as relevant article with special flag
                    relevant_articles.append({
                        "title": title,
                        "summary": selftext[:300],
                        "source": f"r/{sub_name}",
                        "link": f"https://reddit.com{post.permalink}",
                        "published": datetime.fromtimestamp(post.created_utc, timezone.utc).isoformat(),
                        "is_reddit": True,
                        "upvotes": ups
                    })
        except Exception as exc:
            logger.warning("Reddit PRAW fetch failed for %s: %s", ticker, exc)

    logger.info(
        "Asset sentiment for %s (%s): %d relevant articles (including social)",
        asset_name,
        ticker,
        len(relevant_articles),
    )

    if not relevant_articles:
        if return_dict:
            return {"score": 0.0, "breakdown": "Sentiment: 0.0 (No news)", "unverified": False}
        return 0.0

    weighted_sum = 0.0
    weight_sum = 0.0
    breakdown_parts = []
    all_sources_low_credibility = True

    for article in relevant_articles:
        title = article.get("title", "")
        raw_sentiment = get_headline_nlp_sentiment(title)  # Upgraded high-fidelity NLP
        source = article.get("source", "unknown")
        link = article.get("link", "")
        
        weight = get_credibility_weight(source, link)
        if article.get("is_reddit"):
            # Scale Reddit weight by upvotes using log-scale multiplier
            upvotes = max(1, article.get("upvotes", 1))
            upvote_mult = float(np.log1p(upvotes))
            weight = weight * upvote_mult
            
        if weight > 0.4:
            all_sources_low_credibility = False
            
        weighted_sum += raw_sentiment * weight
        weight_sum += weight
        
        sign = "+" if raw_sentiment >= 0 else ""
        if article.get("is_reddit"):
            breakdown_parts.append(f"{source} (ups={article.get('upvotes')}) {sign}{raw_sentiment:.1f} × {weight:.1f}")
        else:
            breakdown_parts.append(f"{source} {sign}{raw_sentiment:.1f} × {weight:.1f}")

    final_sentiment = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    final_sentiment = round(final_sentiment, 4)

    sign = "+" if final_sentiment >= 0 else ""
    breakdown_str = f"Sentiment: {sign}{final_sentiment:.1f} ({', '.join(breakdown_parts[:3])})"
    
    if all_sources_low_credibility:
        breakdown_str += " [UNVERIFIED SENTIMENT]"

    if return_dict:
        return {
            "score": final_sentiment,
            "breakdown": breakdown_str,
            "unverified": all_sources_low_credibility
        }
    return final_sentiment



# ---------------------------------------------------------------------------
# CNN Fear & Greed Index
# ---------------------------------------------------------------------------


def fetch_fear_greed_stock() -> dict:
    """
    Fetch the CNN Fear & Greed index for the stock market.

    Returns
    -------
    dict
        Keys: score (float), rating (str), previous_close (float),
               one_week_ago (float), one_month_ago (float),
               one_year_ago (float), timestamp (str).
        Returns an empty dict on failure.
    """
    logger.info("Fetching CNN Fear & Greed index…")
    resp = _safe_get(
        _CNN_FEAR_GREED_URL,
        headers={
            **_DEFAULT_HEADERS,
            "Referer": "https://edition.cnn.com/markets/fear-and-greed",
        },
    )
    if resp is None:
        return {}

    try:
        data = resp.json()
        fg = data.get("fear_and_greed", {})

        score = fg.get("score")
        rating = fg.get("rating", "")

        # Historical snapshots live under fear_and_greed_historical
        historical = data.get("fear_and_greed_historical", {})
        hist_data = historical.get("data", [])

        def _hist_score(idx: int) -> float | None:
            try:
                return float(hist_data[idx].get("y"))
            except (IndexError, TypeError, ValueError):
                return None

        result = {
            "score":          float(score) if score is not None else None,
            "rating":         rating,
            "previous_close": _hist_score(-2),   # second-to-last point
            "one_week_ago":   fg.get("previous_1_week"),
            "one_month_ago":  fg.get("previous_1_month"),
            "one_year_ago":   fg.get("previous_1_year"),
            "timestamp":      _now_utc().isoformat(),
        }
        logger.info("Fear & Greed: score=%.1f (%s)", result["score"] or 0, rating)
        return result

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error("Failed to parse CNN Fear & Greed response: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# FRED macro-economic data
# ---------------------------------------------------------------------------


def _fetch_fred_series(series_id: str, api_key: str, limit: int = 5) -> list[dict]:
    """
    Retrieve the latest *limit* observations for a FRED series.

    Returns
    -------
    list[dict]
        Each dict: {date: str, value: float | None}
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":    series_id,
        "api_key":      api_key,
        "file_type":    "json",
        "sort_order":   "desc",
        "limit":        limit,
        "observation_start": "2020-01-01",
    }
    resp = _safe_get(url, params=params)
    if resp is None:
        return []

    try:
        data = resp.json()
        observations = []
        for obs in data.get("observations", []):
            raw_val = obs.get("value", ".")
            try:
                value: float | None = float(raw_val)
            except (ValueError, TypeError):
                value = None
            observations.append({"date": obs.get("date"), "value": value})
        return observations
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error("FRED parse error for %s: %s", series_id, exc)
        return []


def fetch_fred_data(api_key: str | None) -> dict | None:
    """
    Fetch macro-economic data from the FRED API.

    Retrieves: Fed Funds Rate, CPI, PCE, Unemployment Rate, GDP, M2.

    Parameters
    ----------
    api_key : str | None
        FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html).
        If None or empty, returns None immediately.

    Returns
    -------
    dict | None
        Keys match ``_FRED_SERIES`` (friendly names).  Each value is a dict:
          {latest: float|None, latest_date: str|None, history: list[dict]}.
        Returns None if no API key is provided.
    """
    if not api_key:
        logger.info("FRED API key not set – skipping FRED data.")
        return None

    logger.info("Fetching FRED macro data (%d series)…", len(_FRED_SERIES))
    result: dict = {}

    for friendly_name, series_id in _FRED_SERIES.items():
        logger.debug("  FRED: %s (%s)", friendly_name, series_id)
        observations = _fetch_fred_series(series_id, api_key, limit=12)

        if observations:
            latest_obs = observations[0]  # sorted desc already
            latest_value = latest_obs.get("value")
            latest_date = latest_obs.get("date")
        else:
            latest_value = None
            latest_date = None

        result[friendly_name] = {
            "series_id":   series_id,
            "latest":      latest_value,
            "latest_date": latest_date,
            "history":     observations,
        }

    logger.info("FRED data fetched for: %s", list(result.keys()))
    return result


# ---------------------------------------------------------------------------
# US Treasury yield curve
# ---------------------------------------------------------------------------

# Mapping from Treasury XML tag name → maturity label
_TREASURY_MATURITY_MAP: dict[str, str] = {
    "BC_1MONTH":  "1M",
    "BC_2MONTH":  "2M",
    "BC_3MONTH":  "3M",
    "BC_6MONTH":  "6M",
    "BC_1YEAR":   "1Y",
    "BC_2YEAR":   "2Y",
    "BC_3YEAR":   "3Y",
    "BC_5YEAR":   "5Y",
    "BC_7YEAR":   "7Y",
    "BC_10YEAR":  "10Y",
    "BC_20YEAR":  "20Y",
    "BC_30YEAR":  "30Y",
}


def fetch_treasury_yields() -> dict:
    """
    Fetch the US Treasury yield curve for the current month.

    Parses the official Treasury XML feed — no API key required.

    Returns
    -------
    dict
        Keys: yields (dict of maturity→rate), date (str of latest entry),
              curve_shape ("normal" | "flat" | "inverted" | "unknown"),
              spread_2y_10y (float | None).
        Returns an empty dict on failure.
    """
    now = _now_utc()
    yyyymm = now.strftime("%Y%m")
    url = _TREASURY_XML_URL_TEMPLATE.format(yyyymm=yyyymm)

    logger.info("Fetching Treasury yield curve for %s…", yyyymm)

    # Treasury may not have published current month yet; try previous month too
    for attempt_yyyymm in [yyyymm, (now - timedelta(days=32)).strftime("%Y%m")]:
        attempt_url = _TREASURY_XML_URL_TEMPLATE.format(yyyymm=attempt_yyyymm)
        resp = _safe_get(attempt_url, headers={"User-Agent": _USER_AGENT})
        if resp is not None:
            break
    else:
        logger.error("Could not fetch Treasury yield data.")
        return {}

    try:
        root = ET.fromstring(resp.content)
        # Namespace used by Treasury XML
        ns = {"ns": "http://www.w3.org/2005/Atom"}
        entries = root.findall(".//ns:entry", ns)

        if not entries:
            # Try without namespace
            entries = root.findall(".//entry")

        if not entries:
            logger.warning("No entries found in Treasury XML response.")
            return {}

        # Use the last entry (most recent date)
        last_entry = entries[-1]

        def _find_text(tag: str) -> str | None:
            # Try with and without namespace prefix
            for prefix in ["d:", "m:", ""]:
                elem = last_entry.find(f".//{prefix}{tag}")
                if elem is not None and elem.text:
                    return elem.text.strip()
            # Fallback: search by local-name substring
            for child in last_entry.iter():
                local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if local == tag and child.text:
                    return child.text.strip()
            return None

        # Extract date
        entry_date: str = _find_text("NEW_DATE") or _find_text("Id") or ""
        if "T" in entry_date:
            entry_date = entry_date.split("T")[0]

        # Extract yields
        yields: dict[str, float | None] = {}
        for xml_tag, label in _TREASURY_MATURITY_MAP.items():
            raw = _find_text(xml_tag)
            try:
                yields[label] = float(raw) if raw else None
            except (ValueError, TypeError):
                yields[label] = None

        # Determine yield curve shape
        y2 = yields.get("2Y")
        y10 = yields.get("10Y")

        if y2 is not None and y10 is not None:
            spread = round(y10 - y2, 4)
            if spread > 0.10:
                curve_shape = "normal"
            elif spread < -0.10:
                curve_shape = "inverted"
            else:
                curve_shape = "flat"
        else:
            spread = None
            curve_shape = "unknown"

        result = {
            "yields":        yields,
            "date":          entry_date,
            "curve_shape":   curve_shape,
            "spread_2y_10y": spread,
        }
        logger.info(
            "Treasury yields: 2Y=%.2f%%, 10Y=%.2f%%, curve=%s",
            y2 or 0, y10 or 0, curve_shape,
        )
        return result

    except ET.ParseError as exc:
        logger.error("XML parse error for Treasury yields: %s", exc)
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error parsing Treasury yields: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Economic calendar (ForexFactory)
# ---------------------------------------------------------------------------

# Impact colour class → human-readable label
_IMPACT_MAP: dict[str, str] = {
    "high":   "High",
    "medium": "Medium",
    "low":    "Low",
    "none":   "None",
    "holiday":"Holiday",
}

_SAMPLE_CALENDAR: list[dict] = [
    {
        "date":       "2026-05-30",
        "time":       "08:30",
        "currency":   "USD",
        "event":      "Initial Jobless Claims",
        "impact":     "Medium",
        "forecast":   "220K",
        "previous":   "215K",
        "actual":     None,
    },
    {
        "date":       "2026-05-30",
        "time":       "10:00",
        "currency":   "USD",
        "event":      "Pending Home Sales m/m",
        "impact":     "Medium",
        "forecast":   "1.1%",
        "previous":   "-4.3%",
        "actual":     None,
    },
    {
        "date":       "2026-06-01",
        "time":       "All Day",
        "currency":   "USD",
        "event":      "ISM Manufacturing PMI",
        "impact":     "High",
        "forecast":   "49.8",
        "previous":   "48.7",
        "actual":     None,
    },
    {
        "date":       "2026-06-04",
        "time":       "08:30",
        "currency":   "USD",
        "event":      "Average Hourly Earnings m/m",
        "impact":     "High",
        "forecast":   "0.3%",
        "previous":   "0.3%",
        "actual":     None,
    },
    {
        "date":       "2026-06-04",
        "time":       "08:30",
        "currency":   "USD",
        "event":      "Non-Farm Employment Change",
        "impact":     "High",
        "forecast":   "185K",
        "previous":   "177K",
        "actual":     None,
    },
    {
        "date":       "2026-06-04",
        "time":       "08:30",
        "currency":   "USD",
        "event":      "Unemployment Rate",
        "impact":     "High",
        "forecast":   "4.2%",
        "previous":   "4.2%",
        "actual":     None,
    },
    {
        "date":       "2026-06-05",
        "time":       "08:30",
        "currency":   "USD",
        "event":      "Trade Balance",
        "impact":     "Medium",
        "forecast":   "-64.8B",
        "previous":   "-71.4B",
        "actual":     None,
    },
]


def _parse_forexfactory_html(html: str) -> list[dict]:
    """
    Parse raw ForexFactory calendar HTML and return structured events.

    ForexFactory renders data inside a <table class="calendar__table">.
    Each row is either a date row or an event row.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_=re.compile(r"calendar"))
    if table is None:
        # Try alternative container
        table = soup.find("table")

    if table is None:
        logger.warning("ForexFactory: could not locate calendar table.")
        return []

    events: list[dict] = []
    current_date: str = ""
    now = _now_utc()
    cutoff = now + timedelta(days=7)

    rows = table.find_all("tr")
    for row in rows:
        classes = row.get("class", [])

        # ---- Date row -------------------------------------------------------
        date_cell = row.find("td", class_=re.compile(r"date"))
        if date_cell and date_cell.get_text(strip=True):
            raw_date = date_cell.get_text(strip=True)
            # ForexFactory uses e.g. "Mon May 29" — add current year
            try:
                parsed = datetime.strptime(
                    f"{raw_date} {now.year}", "%a %b %d %Y"
                ).replace(tzinfo=timezone.utc)
                # Handle year boundary
                if parsed < now - timedelta(days=30):
                    parsed = parsed.replace(year=now.year + 1)
                current_date = parsed.strftime("%Y-%m-%d")
            except ValueError:
                current_date = raw_date
            continue

        # ---- Event row -------------------------------------------------------
        # Skip rows without impact markers
        impact_span = row.find(
            "td", class_=re.compile(r"impact")
        ) or row.find("span", class_=re.compile(r"impact"))

        time_cell = row.find("td", class_=re.compile(r"time"))
        currency_cell = row.find("td", class_=re.compile(r"currency"))
        event_cell = row.find("td", class_=re.compile(r"event"))
        forecast_cell = row.find("td", class_=re.compile(r"forecast"))
        previous_cell = row.find("td", class_=re.compile(r"previous"))
        actual_cell = row.find("td", class_=re.compile(r"actual"))

        if not event_cell:
            continue

        event_name = event_cell.get_text(strip=True)
        if not event_name:
            continue

        # Parse impact
        impact_str = "None"
        if impact_span:
            span_inner = impact_span.find("span")
            if span_inner:
                span_classes = " ".join(span_inner.get("class", []))
            else:
                span_classes = " ".join(impact_span.get("class", []))
            if "red" in span_classes or "high" in span_classes:
                impact_str = "High"
            elif "orange" in span_classes or "medium" in span_classes:
                impact_str = "Medium"
            elif "yellow" in span_classes or "low" in span_classes:
                impact_str = "Low"
            elif "gray" in span_classes or "holiday" in span_classes:
                impact_str = "Holiday"

        event_time = time_cell.get_text(strip=True) if time_cell else ""
        currency = currency_cell.get_text(strip=True) if currency_cell else ""
        forecast = forecast_cell.get_text(strip=True) if forecast_cell else ""
        previous = previous_cell.get_text(strip=True) if previous_cell else ""
        actual = actual_cell.get_text(strip=True) if actual_cell else None

        if not current_date:
            continue

        # Filter to next 7 days
        try:
            event_dt = datetime.strptime(current_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            if event_dt > cutoff:
                continue
        except ValueError:
            pass

        events.append(
            {
                "date":     current_date,
                "time":     event_time,
                "currency": currency,
                "event":    event_name,
                "impact":   impact_str,
                "forecast": forecast,
                "previous": previous,
                "actual":   actual if actual else None,
            }
        )

    return events


def fetch_economic_calendar() -> list[dict]:
    """
    Scrape the ForexFactory economic calendar for the next 7 days.

    Falls back to a hardcoded sample calendar if scraping fails or
    returns fewer than 2 events (likely blocked/bot-detected).

    Returns
    -------
    list[dict]
        Each dict: date, time, currency, event, impact, forecast, previous, actual.
    """
    logger.info("Fetching economic calendar from ForexFactory…")

    headers = {
        **_DEFAULT_HEADERS,
        "Referer":     "https://www.google.com/",
        "Cache-Control": "no-cache",
    }

    resp = _safe_get(_FOREXFACTORY_CALENDAR_URL, headers=headers, timeout=20)

    if resp is not None:
        try:
            events = _parse_forexfactory_html(resp.text)
            if len(events) >= 2:
                logger.info("Economic calendar: %d events parsed.", len(events))
                return events
            else:
                logger.warning(
                    "ForexFactory returned only %d events – using fallback.", len(events)
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to parse ForexFactory HTML: %s", exc)
    else:
        logger.warning("Could not reach ForexFactory – using fallback calendar.")

    logger.info("Using hardcoded sample economic calendar.")
    return _SAMPLE_CALENDAR


# ---------------------------------------------------------------------------
# Master research aggregator
# ---------------------------------------------------------------------------


def fetch_all_research(config: dict) -> dict:
    """
    Execute every research module and aggregate results.

    Parameters
    ----------
    config : dict
        Expected keys (all optional, degrade gracefully if missing):
          - fred_api_key   : str | None   – FRED API key
          - cryptopanic_api_key : str | None – CryptoPanic API key
          - assets         : list[dict]   – each: {name: str, ticker: str}
          - include_calendar : bool       – fetch economic calendar (default True)
          - include_fear_greed : bool     – fetch CNN F&G (default True)
          - include_treasury : bool       – fetch Treasury yields (default True)

    Returns
    -------
    dict
        {
            "timestamp":         str (ISO-8601 UTC),
            "news":              list[dict],
            "cryptopanic_news":  list[dict],
            "asset_sentiment":   dict[str, float],
            "overall_sentiment": float,
            "fear_greed":        dict,
            "fred":              dict | None,
            "treasury_yields":   dict,
            "economic_calendar": list[dict],
        }
    """
    start_time = _now_utc()
    logger.info("=" * 60)
    logger.info("Starting full research cycle at %s", start_time.isoformat())
    logger.info("=" * 60)

    # -- Config extraction with safe defaults ---------------------------------
    fred_api_key: str | None = (
        config.get("fred_api_key") or os.getenv("FRED_API_KEY") or None
    )
    cryptopanic_api_key: str | None = (
        config.get("cryptopanic_api_key") or os.getenv("CRYPTOPANIC_API_KEY") or None
    )
    assets: list[dict] = config.get("assets", [])
    include_calendar: bool = config.get("include_calendar", True)
    include_fear_greed: bool = config.get("include_fear_greed", True)
    include_treasury: bool = config.get("include_treasury", True)

    # -- 1. General news ------------------------------------------------------
    news_list = fetch_all_news()

    # -- 2. CryptoPanic -------------------------------------------------------
    crypto_news = fetch_cryptopanic_news(cryptopanic_api_key)

    # -- 3. Asset-level sentiment ---------------------------------------------
    all_articles = news_list + crypto_news
    asset_sentiment: dict[str, float] = {}
    for asset in assets:
        name = asset.get("name", "")
        ticker = asset.get("ticker", "")
        if name:
            key = f"{name} ({ticker})" if ticker else name
            asset_sentiment[key] = calculate_asset_sentiment(
                name, ticker, all_articles, return_dict=config.get("return_dict", False),
                asset_class=asset.get("asset_class", "stock")
            )

    # -- 4. Overall market sentiment from all headlines -----------------------
    all_headlines = [a.get("title", "") for a in all_articles if a.get("title")]
    overall_sentiment = scale_sentiment(all_headlines)
    logger.info("Overall market sentiment score: %.4f", overall_sentiment)

    # -- 5. CNN Fear & Greed --------------------------------------------------
    fear_greed: dict = {}
    if include_fear_greed:
        fear_greed = fetch_fear_greed_stock()

    # -- 6. FRED macro data ---------------------------------------------------
    fred_data = fetch_fred_data(fred_api_key)

    # -- 7. US Treasury yields ------------------------------------------------
    treasury_yields: dict = {}
    if include_treasury:
        treasury_yields = fetch_treasury_yields()

    # -- 8. Economic calendar -------------------------------------------------
    economic_calendar: list[dict] = []
    if include_calendar:
        economic_calendar = fetch_economic_calendar()

    # -- Assemble final result ------------------------------------------------
    elapsed = (_now_utc() - start_time).total_seconds()
    logger.info("Research cycle completed in %.1f seconds.", elapsed)

    return {
        "timestamp":         start_time.isoformat(),
        "elapsed_seconds":   round(elapsed, 2),
        "news":              news_list,
        "cryptopanic_news":  crypto_news,
        "asset_sentiment":   asset_sentiment,
        "overall_sentiment": overall_sentiment,
        "fear_greed":        fear_greed,
        "fred":              fred_data,
        "treasury_yields":   treasury_yields,
        "economic_calendar": economic_calendar,
    }


# ---------------------------------------------------------------------------
# CLI / standalone test entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    _test_config: dict = {
        "fred_api_key":          os.getenv("FRED_API_KEY"),
        "cryptopanic_api_key":   os.getenv("CRYPTOPANIC_API_KEY"),
        "include_calendar":      True,
        "include_fear_greed":    True,
        "include_treasury":      True,
        "assets": [
            {"name": "Bitcoin",   "ticker": "BTC"},
            {"name": "Ethereum",  "ticker": "ETH"},
            {"name": "Apple",     "ticker": "AAPL"},
            {"name": "Tesla",     "ticker": "TSLA"},
            {"name": "S&P 500",   "ticker": "SPY"},
            {"name": "Gold",      "ticker": "GOLD"},
        ],
    }

    research = fetch_all_research(_test_config)

    print("\n" + "=" * 60)
    print("RESEARCH SUMMARY")
    print("=" * 60)
    print(f"Timestamp           : {research['timestamp']}")
    print(f"Elapsed             : {research['elapsed_seconds']}s")
    print(f"Total news articles : {len(research['news'])}")
    print(f"CryptoPanic articles: {len(research['cryptopanic_news'])}")
    print(f"Overall sentiment   : {research['overall_sentiment']}")
    print(f"Fear & Greed score  : {research['fear_greed'].get('score')}")
    print(f"Treasury curve shape: {research['treasury_yields'].get('curve_shape')}")
    print(f"Calendar events     : {len(research['economic_calendar'])}")
    print("\nAsset Sentiment:")
    for k, v in research["asset_sentiment"].items():
        print(f"  {k:30s} → {v:+.4f}")
    print("\nFRED (latest values):")
    if research["fred"]:
        for k, v in research["fred"].items():
            print(f"  {k:25s}: {v['latest']} ({v['latest_date']})")
    else:
        print("  (no FRED API key supplied)")
    print("\nTreasury Yields:")
    pprint.pprint(research["treasury_yields"].get("yields", {}))
    print("\nUpcoming Calendar Events (High impact):")
    for ev in research["economic_calendar"]:
        if ev.get("impact") == "High":
            print(
                f"  {ev['date']} {ev['time']:8s} [{ev['currency']}] "
                f"{ev['event']} | fcst={ev['forecast']} prev={ev['previous']}"
            )
