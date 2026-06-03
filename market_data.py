"""
market_data.py
==============
Autonomous Trading Research Agent — Market Data Module

Fetches price data for:
  - Traditional assets (stocks, indices, forex, commodities, bonds, ETFs) via yfinance in batches of 50
  - Crypto basic price data via CoinGecko /coins/markets (batch, low rate-limit impact)
  - Crypto OHLCV via CoinGecko /coins/{id}/ohlc — restricted to top 5 coins by market cap only
  - Fear & Greed Index via alternative.me
  - Global macro snapshot

CoinGecko rate-limit strategy:
  - 1.5-second proactive delay between ALL CoinGecko calls
  - Max 3 retries for OHLCV endpoints (not 6)
  - Max wait capped at 16 seconds per retry
  - OHLCV failures skip technical indicators gracefully — they never block the run
  - /coins/markets used for all basic price data (no rate-limit issues)
  - OHLCV fetched sequentially (not concurrently) to avoid burst rate-limiting

All network calls use exponential backoff (up to 5 retries, 16s cap).
On failure the asset is marked is_stale=True and error is populated.
Nothing crashes — every exception is caught and logged.
"""

from __future__ import annotations

import json
import logging
import time
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm
from bs4 import BeautifulSoup

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

logger = logging.getLogger("market_data")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"

GLOBAL_TICKERS: list[str] = [
    "SPY", "QQQ", "^GSPC", "^NDX", "^DJI", "^FTSE", "^N225", "^NSEI",
    "BTC-USD", "GC=F", "CL=F", "^TNX", "^VIX", "DX-Y.NYB", "HYG",
    "EURUSD=X", "USDINR=X",
]

GLOBAL_ASSET_CLASS_MAP: dict[str, str] = {
    "SPY":       "etf",
    "QQQ":       "etf",
    "^GSPC":     "index",
    "^NDX":      "index",
    "^DJI":      "index",
    "^FTSE":     "index",
    "^N225":     "index",
    "^NSEI":     "index",
    "BTC-USD":   "crypto",
    "GC=F":      "commodity",
    "CL=F":      "commodity",
    "^TNX":      "bond",
    "^VIX":      "index",
    "DX-Y.NYB":  "forex",
    "HYG":       "etf",
    "EURUSD=X":  "forex",
    "USDINR=X":  "forex",
}

COINGECKO_CALL_DELAY: float = 1.5   # seconds between consecutive CG calls (proactive rate-limit guard)
COINGECKO_OHLCV_MAX_RETRIES: int   = 3    # max retries for OHLCV endpoints (not general markets)
COINGECKO_MAX_WAIT: float           = 16.0 # cap on exponential backoff wait (seconds)
TOP_CRYPTO_OHLCV_LIMIT: int         = 5    # only fetch OHLCV for top N coins by market cap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _empty_asset(ticker: str, asset_class: str, name: str = "", error: str | None = None) -> dict:
    """Return a fully-structured asset dict with all fields set to sensible defaults."""
    return {
        "ticker":         ticker,
        "name":           name or ticker,
        "asset_class":    asset_class,
        "price":          float("nan"),
        "change_24h_pct": float("nan"),
        "change_7d_pct":  float("nan"),
        "change_30d_pct": float("nan"),
        "volume":         float("nan"),
        "market_cap":     float("nan"),
        "high_52w":       float("nan"),
        "low_52w":        float("nan"),
        "ohlcv":          pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
        "fetch_time":     _now_iso(),
        "is_stale":       True,
        "error":          error,
    }


def _safe_float(value: Any) -> float:
    """Convert *value* to float, returning nan on failure."""
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _pct_change_from_ohlcv(ohlcv: pd.DataFrame | None, days: int) -> float:
    """Compute n-day percentage change from close prices in *ohlcv*."""
    if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
        return float("nan")
    closes = ohlcv["close"].dropna()
    if len(closes) < 2:
        return float("nan")
    lookback = min(days, len(closes) - 1)
    old_price = closes.iloc[-(lookback + 1)]
    new_price = closes.iloc[-1]
    if old_price == 0 or np.isnan(old_price):
        return float("nan")
    return float((new_price - old_price) / old_price * 100)


def _format_price_with_tag(price: float, is_stale: bool = False) -> str:
    """Format price, appending [STALE] if data is stale."""
    if np.isnan(price):
        return "NaN [UNAVAILABLE]"
    tag = " [STALE]" if is_stale else ""
    return f"{price:.2f}{tag}"


def validate_asset(asset: dict) -> bool:
    """Validate that price > 0, volume > 0 (for stocks/crypto), and ohlcv has at least 50 rows."""
    price = asset.get("price")
    if price is None or np.isnan(price) or price <= 0:
        return False
        
    asset_class = asset.get("asset_class", "stock")
    if asset_class in ("stock", "crypto"):
        vol = asset.get("volume")
        if vol is None or np.isnan(vol) or vol <= 0:
            return False
            
    ohlcv = asset.get("ohlcv")
    if ohlcv is None or not isinstance(ohlcv, pd.DataFrame) or len(ohlcv) < 50:
        return False
        
    return True


def fallback_to_cache(ticker: str, asset_class: str, cache_path: str = "state/last-run.json") -> dict | None:
    """Attempt to fallback to previously cached state from state/last-run.json on validation failure."""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            
            cached_asset = None
            market_data = d.get("market_data", {})
            
            if ticker in market_data.get("traditional", {}):
                cached_asset = market_data["traditional"][ticker]
            elif ticker in market_data.get("crypto", {}):
                cached_asset = market_data["crypto"][ticker]
            else:
                for item in market_data.get("global_snapshot", []):
                    if item.get("ticker") == ticker:
                        cached_asset = item
                        break
            
            if cached_asset:
                price = cached_asset.get("price", float("nan"))
                if price is not None and not np.isnan(price) and price > 0:
                    dummy_ohlcv = pd.DataFrame(
                        {
                            "open": [price] * 50,
                            "high": [price] * 50,
                            "low": [price] * 50,
                            "close": [price] * 50,
                            "volume": [cached_asset.get("volume", 1000.0)] * 50
                        },
                        index=pd.date_range(end=pd.Timestamp.now(tz=timezone.utc), periods=50, freq="D")
                    )
                    return {
                        "ticker":         ticker,
                        "name":           cached_asset.get("name", ticker),
                        "asset_class":    asset_class,
                        "price":          price,
                        "change_24h_pct": cached_asset.get("change_24h_pct", float("nan")),
                        "change_7d_pct":  cached_asset.get("change_7d_pct", float("nan")),
                        "change_30d_pct": cached_asset.get("change_30d_pct", float("nan")),
                        "volume":         cached_asset.get("volume", float("nan")),
                        "market_cap":     cached_asset.get("market_cap", float("nan")),
                        "high_52w":       cached_asset.get("high_52w", float("nan")),
                        "low_52w":        cached_asset.get("low_52w", float("nan")),
                        "ohlcv":          dummy_ohlcv,
                        "fetch_time":     _now_iso(),
                        "is_stale":       True,
                        "error":          "Fetched data failed validation. Fallback to cached state.",
                    }
    except Exception as exc:
        logger.error("Failed to load cached asset %s: %s", ticker, exc)
    return None


def log_data_quality(ticker: str, asset: dict):
    """Log data quality score per asset."""
    ohlcv = asset.get("ohlcv")
    if ohlcv is None or not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
        logger.info("%s: data quality 0%% (All candles missing)", ticker)
        return
        
    total_rows = len(ohlcv)
    asset_class = asset.get("asset_class", "stock")
    
    cols_to_check = ["open", "high", "low", "close"]
    if asset_class in ("stock", "crypto"):
        cols_to_check.append("volume")
        
    missing = 0
    for idx, row in ohlcv.iterrows():
        is_missing = False
        for col in cols_to_check:
            if col in row:
                val = row[col]
                if val is None or np.isnan(val):
                    is_missing = True
                    break
            else:
                is_missing = True
                break
        if is_missing:
            missing += 1
            
    quality_score = round((total_rows - missing) / total_rows * 100)
    logger.info("%s: data quality %d%% (%d missing candle%s)", ticker, quality_score, missing, "" if missing == 1 else "s")


def fetch_vix_data_with_fallbacks() -> pd.DataFrame:
    """Fetch VIX data using primary and fallback channels.
    Primary: yf.download("^VIX", period="5d", interval="1d")
    Fallback 1: yf.Ticker("^VIX").fast_info
    Fallback 2: Scrape current VIX from https://www.cboe.com/tradable_products/vix/
    Fallback 3: Calculate synthetic VIX proxy from SPY options chain
    Fallback 4: Cache load from state/last-run.json
    Fallback 5: Safe default (15.0)
    """
    logger.info("Attempting to fetch VIX data...")
    is_stale = False
    vix_val = float("nan")
    
    # 1. Primary: yf.download
    try:
        df = yf.download("^VIX", period="5d", interval="1d", auto_adjust=True, progress=False, threads=False)
        if df is not None and not df.empty and "Close" in df.columns:
            val = df["Close"].dropna().iloc[-1]
            if val is not None and not np.isnan(val) and val > 0:
                logger.info("VIX primary fetch successful: %.2f", val)
                df = df.rename(columns={"adj close": "Close", "close": "Close"})
                df = df[["Close"]].copy()
                df["is_stale"] = False
                return df
    except Exception as exc:
        logger.warning("VIX Primary yf.download failed: %s", exc)
        
    # 2. Fallback 1: fast_info
    try:
        t = yf.Ticker("^VIX")
        if hasattr(t, "fast_info") and "lastPrice" in t.fast_info:
            val = t.fast_info["lastPrice"]
            if val is not None and not np.isnan(val) and val > 0:
                logger.info("VIX Fallback 1 (fast_info) successful: %.2f", val)
                vix_val = float(val)
    except Exception as exc:
        logger.warning("VIX Fallback 1 (fast_info) failed: %s", exc)
        
    # 3. Fallback 2: scrape CBOE
    if np.isnan(vix_val):
        try:
            r = requests.get("https://www.cboe.com/tradable_products/vix/", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                for p in soup.find_all("p"):
                    if p.text and "VIX Spot Price" in p.text:
                        parent = p.parent
                        if parent:
                            h2 = parent.find("h2")
                            if h2:
                                val_str = h2.text.replace("$", "").strip()
                                val = float(val_str)
                                if val > 0:
                                    logger.info("VIX Fallback 2 (CBOE scraping) successful: %.2f", val)
                                    vix_val = val
                                    break
        except Exception as exc:
            logger.warning("VIX Fallback 2 (CBOE scraping) failed: %s", exc)
            
    # 4. Fallback 3: synthetic VIX from SPY options
    if np.isnan(vix_val):
        try:
            spy = yf.Ticker("SPY")
            if spy.options:
                today = datetime.now(timezone.utc).date()
                target = today + timedelta(days=30)
                exp_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in spy.options]
                closest = min(exp_dates, key=lambda d: abs(d - target))
                opt = spy.option_chain(closest.strftime("%Y-%m-%d"))
                spy_price = spy.fast_info.get("lastPrice", 520.0)
                calls_near = opt.calls[abs(opt.calls["strike"] - spy_price) / spy_price <= 0.02]
                puts_near = opt.puts[abs(opt.puts["strike"] - spy_price) / spy_price <= 0.02]
                ivs = calls_near["impliedVolatility"].dropna().tolist() + puts_near["impliedVolatility"].dropna().tolist()
                if ivs:
                    val = sum(ivs) / len(ivs) * 100.0
                    logger.info("VIX Fallback 3 (Synthetic SPY options IV) successful: %.2f", val)
                    vix_val = val
        except Exception as exc:
            logger.warning("VIX Fallback 3 (Synthetic options) failed: %s", exc)
            
    # 5. Fallback 4: Cache load from last-run.json
    if np.isnan(vix_val):
        is_stale = True
        try:
            cache_path = "state/last-run.json"
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                val = d.get("macro_state", {}).get("vix")
                if val is not None and not np.isnan(val):
                    logger.info("VIX Fallback 4 (Cache load) successful: %.2f [STALE]", val)
                    vix_val = float(val)
        except Exception as exc:
            logger.warning("VIX Fallback 4 (Cache load) failed: %s", exc)
            
    # 6. Fallback 5: Hardcoded default
    if np.isnan(vix_val):
        is_stale = True
        vix_val = 15.0
        logger.info("VIX Fallback 5 (Hardcoded default) active: %.2f [STALE]", vix_val)
        
    idx = pd.date_range(end=pd.Timestamp.now(tz=timezone.utc), periods=5, freq="D")
    df = pd.DataFrame({"Close": [vix_val] * 5}, index=idx)
    df["is_stale"] = is_stale
    return df


# ---------------------------------------------------------------------------
# Core network utility — request_with_backoff
# ---------------------------------------------------------------------------

def request_with_backoff(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    max_retries: int = 5,
    max_wait: float = COINGECKO_MAX_WAIT,
) -> dict | None:
    """HTTP GET with exponential backoff and rate limit recovery.

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts after the initial request.
    max_wait : float
        Hard cap on exponential backoff sleep duration in seconds (default: 16s).
    """
    last_error: str = ""
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                wait = min(2 ** attempt * 2, max_wait)
                logger.warning("Rate-limited by %s — waiting %.0fs before retry %d", url, wait, attempt + 1)
                time.sleep(wait)
                last_error = f"HTTP 429 after attempt {attempt + 1}"
                continue

            if resp.status_code in (500, 502, 503, 504):
                wait = min(2 ** attempt * 2, max_wait)
                logger.warning("Server error %d from %s — waiting %.0fs", resp.status_code, url, wait)
                time.sleep(wait)
                last_error = f"HTTP {resp.status_code} after attempt {attempt + 1}"
                continue

            logger.error("Non-retryable HTTP %d from %s", resp.status_code, url)
            return None

        except requests.exceptions.Timeout:
            wait = min(2 ** attempt * 2, max_wait)
            logger.warning("Timeout on %s — waiting %.0fs", url, wait)
            last_error = "Timeout"
            time.sleep(wait)

        except requests.exceptions.ConnectionError as exc:
            wait = min(2 ** attempt * 2, max_wait)
            logger.warning("Connection error on %s — waiting %.0fs", url, wait)
            last_error = str(exc)
            time.sleep(wait)

        except Exception as exc:
            logger.error("Unexpected error fetching %s: %s", url, exc)
            return None

    logger.error("All %d retries exhausted for %s. Last error: %s", max_retries, url, last_error)
    return None

# ---------------------------------------------------------------------------
# Batched yfinance fetches
# ---------------------------------------------------------------------------

def fetch_traditional_batch(
    batch: list[str],
    asset_class_map: dict[str, str],
    period: str | None = None,
    is_retry: bool = False,
) -> dict[str, dict]:
    """Download OHLCV data for a batch of 50 traditional tickers using a single yfinance request.
    
    PHASE 1 UPGRADE: Uses 1y for stocks/crypto (EMA200 needs 200+ candles), 6mo for others.
    """
    results = {}
    if not batch:
        return results

    # Determine period dynamically if not specified: 1y for stocks/crypto, 6mo for others
    if period is None:
        has_stock_or_crypto = any(asset_class_map.get(t, "stock") in ("stock", "crypto") for t in batch)
        period = "1y" if has_stock_or_crypto else "6mo"

    try:
        logger.info("Executing yfinance batch download for %d tickers with period=%s...", len(batch), period)
        df = yf.download(
            batch,
            period=period,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True
        )

        if df is None or df.empty:
            for t in batch:
                results[t] = _empty_asset(t, asset_class_map.get(t, "stock"), error="yfinance batch download returned empty")
            return results

        is_multi = isinstance(df.columns, pd.MultiIndex)

        for t in batch:
            asset_class = asset_class_map.get(t, "stock")
            candidate = None
            try:
                if t == "^VIX":
                    vix_df = fetch_vix_data_with_fallbacks()
                    price = float(vix_df["Close"].iloc[-1])
                    is_stale = bool(vix_df.get("is_stale", pd.Series([False]*len(vix_df))).iloc[-1])
                    
                    chg_24h = 0.0
                    if len(vix_df) >= 2:
                        prev_vix = vix_df["Close"].iloc[-2]
                        chg_24h = (price - prev_vix) / prev_vix * 100.0 if prev_vix != 0 else 0.0
                        
                    std_ohlcv = vix_df.rename(columns={"Close": "close"})
                    std_ohlcv.columns = [c.lower() for c in std_ohlcv.columns]
                    for col in ["open", "high", "low", "volume"]:
                        if col not in std_ohlcv.columns:
                            std_ohlcv[col] = float("nan")
                    std_ohlcv = std_ohlcv[["open", "high", "low", "close", "volume"]].copy()
                    
                    candidate = {
                        "ticker":         t,
                        "name":           t,
                        "asset_class":    asset_class,
                        "price":          price,
                        "change_24h_pct": chg_24h,
                        "change_7d_pct":  0.0,
                        "change_30d_pct": 0.0,
                        "volume":         0.0,
                        "market_cap":     float("nan"),
                        "high_52w":       price,
                        "low_52w":        price,
                        "ohlcv":          std_ohlcv,
                        "fetch_time":     _now_iso(),
                        "is_stale":       is_stale,
                        "error":          "STALE VIX" if is_stale else None,
                    }
                elif is_multi:
                    if t not in df.columns.levels[0]:
                        candidate = _empty_asset(t, asset_class, error="Ticker not returned in batch")
                    else:
                        ticker_df = df[t].dropna(how="all")
                else:
                    ticker_df = df.dropna(how="all")

                if candidate is None:
                    if ticker_df.empty:
                        candidate = _empty_asset(t, asset_class, error="OHLCV is empty")
                    else:
                        ticker_df.columns = [c.lower() for c in ticker_df.columns]
                        ticker_df = ticker_df.rename(columns={"adj close": "close"})

                        # Pad missing columns with nan
                        for col in ["open", "high", "low", "close", "volume"]:
                            if col not in ticker_df.columns:
                                ticker_df[col] = float("nan")

                        ticker_df = ticker_df[["open", "high", "low", "close", "volume"]].copy()

                        if ticker_df.index.tzinfo is None:
                            ticker_df.index = ticker_df.index.tz_localize("UTC")
                        else:
                            ticker_df.index = ticker_df.index.tz_convert("UTC")

                        price = _safe_float(ticker_df["close"].iloc[-1])
                        volume = _safe_float(ticker_df["volume"].iloc[-1])

                        chg_24h = _pct_change_from_ohlcv(ticker_df, 1)
                        chg_7d = _pct_change_from_ohlcv(ticker_df, 5)
                        chg_30d = _pct_change_from_ohlcv(ticker_df, 21)

                        high_52w = _safe_float(ticker_df["high"].max())
                        low_52w = _safe_float(ticker_df["low"].min())

                        candidate = {
                            "ticker":         t,
                            "name":           t,
                            "asset_class":    asset_class,
                            "price":          price,
                            "change_24h_pct": chg_24h,
                            "change_7d_pct":  chg_7d,
                            "change_30d_pct": chg_30d,
                            "volume":         volume,
                            "market_cap":     float("nan"),
                            "high_52w":       high_52w,
                            "low_52w":        low_52w,
                            "ohlcv":          ticker_df,
                            "fetch_time":     _now_iso(),
                            "is_stale":       np.isnan(price),
                            "error":          None if not np.isnan(price) else "Price is NaN",
                        }

            except Exception as exc:
                logger.error("Error processing %s in batch: %s", t, exc)
                candidate = _empty_asset(t, asset_class, error=str(exc))

            # --- Validation Pass ---
            is_valid = validate_asset(candidate)
            if not is_valid:
                if not is_retry:
                    logger.warning("%s failed validation (price=%s, volume=%s, rows=%d). Retrying once...", 
                                   t, candidate.get("price"), candidate.get("volume"), len(candidate.get("ohlcv", [])))
                    candidate = fetch_traditional_asset(t, asset_class, period=period, is_retry=True)
                    is_valid = validate_asset(candidate)
                
                if not is_valid:
                    logger.warning("%s failed validation after retry. Falling back to cache...", t)
                    cached = fallback_to_cache(t, asset_class)
                    if cached:
                        candidate = cached
                        is_valid = True
                    else:
                        logger.error("%s failed all validation and cache attempts. Marking as stale.", t)
                        candidate = _empty_asset(t, asset_class, error="Data validation failed")
                        is_valid = False

            if is_valid:
                log_data_quality(t, candidate)

            results[t] = candidate

    except Exception as exc:
        logger.error("Failed to execute yfinance batch: %s", exc)
        for t in batch:
            results[t] = _empty_asset(t, asset_class_map.get(t, "stock"), error=str(exc))

    return results


def fetch_all_traditional(watchlist: dict) -> dict:
    """Fetch all traditional tickers concurrently in batches of 50."""
    results: dict[str, dict] = {}
    category_map = {
        "us_stocks":      "stock",
        "indian_stocks":  "stock",
        "indices":        "index",
        "commodities":    "commodity",
        "forex":          "forex",
        "bonds":          "bond",
        "etfs":           "etf",
        "stocks":         "stock",  # backward compatibility
    }

    asset_class_map = {}
    tickers_list = []

    for cat, asset_class in category_map.items():
        tickers = watchlist.get(cat, [])
        if not isinstance(tickers, list):
            continue
        for t in tickers:
            if isinstance(t, str) and t.strip():
                t = t.strip().upper()
                asset_class_map[t] = asset_class
                tickers_list.append(t)

    # De-duplicate tickers
    tickers_list = list(dict.fromkeys(tickers_list))

    # Split into batches of 50
    batches = [tickers_list[i : i + 50] for i in range(0, len(tickers_list), 50)]
    if not batches:
        return results

    logger.info("Executing concurrent fetching for %d traditional tickers across %d batches...", len(tickers_list), len(batches))

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_traditional_batch, batch, asset_class_map): batch for batch in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading watchlists in batches", unit="batch"):
            try:
                batch_res = fut.result()
                results.update(batch_res)
            except Exception as exc:
                logger.error("Batch download raised exception: %s", exc)

    return results

# ---------------------------------------------------------------------------
# CoinGecko concurrent price fetches
# ---------------------------------------------------------------------------

def _coingecko_get(
    endpoint: str,
    params: dict | None = None,
    max_retries: int = 5,
) -> dict | list | None:
    """Call a CoinGecko endpoint with proactive rate-limit delay and exponential backoff.

    A 1.5-second delay is inserted BEFORE every call to proactively avoid hitting
    CoinGecko's free-tier rate limit (30 req/min). The max_wait is always capped at
    COINGECKO_MAX_WAIT (16s).
    """
    url = f"{COINGECKO_BASE}{endpoint}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "LUNA/1.0",
    }
    time.sleep(COINGECKO_CALL_DELAY)  # proactive 1.5s delay between all CoinGecko calls
    return request_with_backoff(
        url,
        params=params,
        headers=headers,
        max_retries=max_retries,
        max_wait=COINGECKO_MAX_WAIT,
    )


def fetch_crypto_markets(coin_ids: list[str]) -> list[dict]:
    """Fetch cryptocurrency market pricing in batches of 50."""
    if not coin_ids:
        return []

    all_market_data: list[dict] = []
    batch_size = 50

    for i in range(0, len(coin_ids), batch_size):
        batch = coin_ids[i : i + batch_size]
        params = {
            "vs_currency":           "usd",
            "ids":                   ",".join(batch),
            "order":                 "market_cap_desc",
            "per_page":              str(len(batch)),
            "page":                  "1",
            "sparkline":             "false",
            "price_change_percentage": "1h,24h,7d,30d",
        }
        result = _coingecko_get("/coins/markets", params=params)
        if isinstance(result, list):
            all_market_data.extend(result)
        else:
            logger.warning("CoinGecko markets batch returned invalid response.")

    return all_market_data


def fetch_crypto_ohlcv(coin_id: str, days: int = 365) -> pd.DataFrame | None:
    """Fetch cryptocurrency OHLCV price history from CoinGecko.

    Uses a hard cap of COINGECKO_OHLCV_MAX_RETRIES (3) retries — much lower than the
    general markets endpoint. If all retries are exhausted, returns None so the caller
    can mark technical indicators as unavailable without blocking the broader run.
    """
    endpoint = f"/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": str(days)}
    result = _coingecko_get(endpoint, params=params, max_retries=COINGECKO_OHLCV_MAX_RETRIES)

    if result is None or not isinstance(result, list) or len(result) == 0:
        logger.warning(
            "OHLCV fetch returned no data for %s after %d retries — "
            "technical indicators will be marked unavailable.",
            coin_id, COINGECKO_OHLCV_MAX_RETRIES,
        )
        return None

    try:
        records = []
        for row in result:
            if len(row) < 5:
                continue
            ts_ms, o, h, l, c = row[0], row[1], row[2], row[3], row[4]
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            records.append({"datetime": dt, "open": o, "high": h, "low": l, "close": c})

        df = pd.DataFrame(records)
        df.set_index("datetime", inplace=True)
        df["volume"] = float("nan")
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df.sort_index(inplace=True)
        return df
    except Exception as exc:
        logger.error("Failed to parse crypto OHLCV for %s: %s", coin_id, exc)
        return None


def fetch_all_crypto(crypto_list: list[Any]) -> dict:
    """Fetch crypto assets using CoinGecko with a careful rate-limit strategy.

    Strategy
    --------
    1. Basic price data  — /coins/markets fetched in batches of 50 for ALL coins.
       This endpoint has a high quota and is fine to hit freely.
    2. OHLCV data        — /coins/{id}/ohlc fetched SEQUENTIALLY (not concurrently)
       with COINGECKO_CALL_DELAY (1.5s) between calls, and ONLY for the top
       TOP_CRYPTO_OHLCV_LIMIT (5) coins by market cap returned from step 1.
       Max retries capped at COINGECKO_OHLCV_MAX_RETRIES (3).
    3. OHLCV failure     — If a coin's OHLCV fetch fails after retries, that coin
       gets an empty OHLCV DataFrame and ohlcv_unavailable=True flag. Its price
       data and pct-change fields from /coins/markets are still retained.
    """
    results: dict[str, dict] = {}
    if not crypto_list:
        return results

    coin_ids: list[str] = []
    coin_symbols: dict[str, str] = {}

    for c in crypto_list:
        if isinstance(c, dict):
            cid = c.get("id")
            csym = c.get("symbol")
            if cid and csym:
                coin_ids.append(cid.lower())
                coin_symbols[cid.lower()] = csym.upper()
        elif isinstance(c, str):
            coin_ids.append(c.lower())
            coin_symbols[c.lower()] = c.upper()

    coin_ids = list(dict.fromkeys(coin_ids))

    # ── Step 1: Fetch market prices for ALL coins via /coins/markets ──────────
    logger.info("Fetching basic price data for %d coins via /coins/markets...", len(coin_ids))
    market_data_list = fetch_crypto_markets(coin_ids)
    market_lookup: dict[str, dict] = {
        item.get("id", ""): item for item in market_data_list if item.get("id")
    }

    # ── Step 2: Determine top-N coins by market cap for OHLCV fetching ────────
    # Sort by market_cap descending; coins not in market_lookup get market_cap=0
    def _mcap(cid: str) -> float:
        mkt = market_lookup.get(cid)
        return _safe_float(mkt.get("market_cap")) if mkt else 0.0

    sorted_by_mcap = sorted(coin_ids, key=_mcap, reverse=True)
    ohlcv_eligible  = sorted_by_mcap[:TOP_CRYPTO_OHLCV_LIMIT]
    ohlcv_skipped   = sorted_by_mcap[TOP_CRYPTO_OHLCV_LIMIT:]

    logger.info(
        "OHLCV fetch: top %d coins eligible (%s) — %d smaller coins skipped to preserve rate limits.",
        len(ohlcv_eligible),
        ", ".join(coin_symbols.get(c, c) for c in ohlcv_eligible),
        len(ohlcv_skipped),
    )

    # ── Step 3: Sequential OHLCV fetch for top-N only ────────────────────────
    ohlcv_results: dict[str, pd.DataFrame | None] = {}

    for cid in tqdm(ohlcv_eligible, desc="Fetching Crypto OHLCV (top 5)", unit="coin"):
        try:
            ohlcv_results[cid] = fetch_crypto_ohlcv(cid, days=365)  # Fetch 1y lookback
        except Exception as exc:
            logger.error("Crypto OHLCV fetch raised exception for %s: %s", cid, exc)
            ohlcv_results[cid] = None

    # Mark skipped coins explicitly — no OHLCV attempt made
    for cid in ohlcv_skipped:
        ohlcv_results[cid] = None

    # ── Step 4: Assemble final asset records ─────────────────────────────────
    for cid in coin_ids:
        mkt = market_lookup.get(cid)
        symbol = coin_symbols.get(cid, cid.upper()) + "-USD"
        ohlcv_was_attempted = cid in ohlcv_eligible

        if mkt is None:
            asset = _empty_asset(symbol, "crypto", error="CoinGecko market pricing unavailable")
            asset["ohlcv_unavailable"] = True
        else:
            price = _safe_float(mkt.get("current_price"))
            asset = {
                "ticker":            symbol,
                "name":              mkt.get("name", cid.title()),
                "asset_class":       "crypto",
                "price":             price,
                "change_24h_pct":    _safe_float(mkt.get("price_change_percentage_24h")),
                "change_7d_pct":     _safe_float(mkt.get("price_change_percentage_7d")),
                "change_30d_pct":    _safe_float(mkt.get("price_change_percentage_30d")),
                "volume":            _safe_float(mkt.get("total_volume")),
                "market_cap":        _safe_float(mkt.get("market_cap")),
                "high_52w":          _safe_float(mkt.get("ath")),
                "low_52w":           _safe_float(mkt.get("atl")),
                "ohlcv":             pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
                "ohlcv_unavailable": False,
                "fetch_time":        _now_iso(),
                "is_stale":          np.isnan(price),
                "error":             None if not np.isnan(price) else "Price is NaN",
            }

        # Inject OHLCV if available; otherwise mark technical indicators as N/A
        ohlcv = ohlcv_results.get(cid)
        if ohlcv is not None and not ohlcv.empty:
            asset["ohlcv"] = ohlcv
            asset["ohlcv_unavailable"] = False
            # Fill any missing pct-change fields from OHLCV (more precise)
            if np.isnan(asset.get("change_24h_pct", float("nan"))):
                asset["change_24h_pct"] = _pct_change_from_ohlcv(ohlcv, 1)
            if np.isnan(asset.get("change_7d_pct", float("nan"))):
                asset["change_7d_pct"] = _pct_change_from_ohlcv(ohlcv, 7)
            if np.isnan(asset.get("change_30d_pct", float("nan"))):
                asset["change_30d_pct"] = _pct_change_from_ohlcv(ohlcv, 30)
        elif ohlcv_was_attempted:
            # OHLCV was attempted but failed — mark as unavailable, keep market price data
            asset["ohlcv_unavailable"] = True
            if not asset.get("error"):
                asset["error"] = "OHLCV fetch failed after retries — technical indicators unavailable"
            logger.info(
                "%s: OHLCV unavailable, technical indicators skipped. Price data retained.", symbol
            )
        else:
            # OHLCV was intentionally skipped (not in top-N)
            asset["ohlcv_unavailable"] = True

        # --- Validation Pass for Crypto ---
        if ohlcv_was_attempted:
            is_valid = validate_asset(asset)
            if not is_valid:
                logger.warning("%s failed validation (price=%s, volume=%s, rows=%d). Retrying crypto OHLCV once...", 
                               symbol, asset.get("price"), asset.get("volume"), len(asset.get("ohlcv", [])))
                try:
                    retry_ohlcv = fetch_crypto_ohlcv(cid, days=365)
                    if retry_ohlcv is not None and not retry_ohlcv.empty:
                        asset["ohlcv"] = retry_ohlcv
                        asset["ohlcv_unavailable"] = False
                        is_valid = validate_asset(asset)
                except Exception as exc:
                    logger.error("%s: retry OHLCV fetch failed: %s", symbol, exc)
                
                if not is_valid:
                    logger.warning("%s failed validation after retry. Falling back to cache...", symbol)
                    cached = fallback_to_cache(symbol, "crypto")
                    if cached:
                        asset = cached
                        is_valid = True
                    else:
                        logger.error("%s failed validation and cache. Marking as stale.", symbol)
                        asset["ohlcv_unavailable"] = True
                        asset["is_stale"] = True
                        if not asset.get("error"):
                            asset["error"] = "Data validation failed"
                        is_valid = False
            
            if is_valid:
                log_data_quality(symbol, asset)

        results[coin_symbols.get(cid, cid.upper())] = asset

    return results

# ---------------------------------------------------------------------------
# Traditional asset fetch (reused fallback/inspect)
# ---------------------------------------------------------------------------

def fetch_traditional_asset(
    ticker: str,
    asset_class: str,
    period: str | None = None,
    is_retry: bool = False,
) -> dict:
    """Fetch a single traditional asset (reused as single-symbol fallback)."""
    batch_res = fetch_traditional_batch([ticker], {ticker: asset_class}, period=period, is_retry=is_retry)
    return batch_res.get(ticker, _empty_asset(ticker, asset_class, error="Fetch failed"))

# ---------------------------------------------------------------------------
# Fear & Greed Index
# ---------------------------------------------------------------------------

def fetch_fear_greed() -> dict:
    """Fetch Fear & Greed Index with exponential backoff recovery."""
    result = request_with_backoff(FEAR_GREED_URL, max_retries=5)
    stale_response = {
        "value":                None,
        "value_classification": None,
        "timestamp":            None,
        "fetch_time":           _now_iso(),
        "is_stale":             True,
        "error":                "Fear & Greed index fetch failed",
    }

    if result is None:
        return stale_response

    try:
        data_list = result.get("data", [])
        if not data_list:
            return stale_response

        entry = data_list[0]
        return {
            "value":                int(entry.get("value", 50)),
            "value_classification": str(entry.get("value_classification", "Neutral")),
            "timestamp":            str(entry.get("timestamp", "")),
            "fetch_time":           _now_iso(),
            "is_stale":             False,
            "error":                None,
        }
    except Exception as exc:
        logger.error("Failed to parse Fear & Greed response: %s", exc)
        stale_response["error"] = str(exc)
        return stale_response

# ---------------------------------------------------------------------------
# Global Macro Snapshot
# ---------------------------------------------------------------------------

def fetch_global_snapshot() -> dict:
    """Fetch a macro snapshot for predefined intermarket tickers concurrently."""
    asset_class_map = {t: GLOBAL_ASSET_CLASS_MAP.get(t, "index") for t in GLOBAL_TICKERS}
    results = fetch_traditional_batch(GLOBAL_TICKERS, asset_class_map)
    return results

# ---------------------------------------------------------------------------
# Master fetch_all_market_data Orchestrator
# ---------------------------------------------------------------------------

def fetch_all_market_data(watchlist: dict) -> dict:
    """Execute all price, indicator, snapshot and sentiment downloads concurrently."""
    start = datetime.now(tz=timezone.utc)
    logger.info("=== fetch_all_market_data started ===")

    # Traditional Assets
    traditional = {}
    try:
        traditional = fetch_all_traditional(watchlist)
    except Exception as exc:
        logger.error("fetch_all_traditional failed: %s", exc)

    # Crypto Assets
    crypto = {}
    crypto_list = watchlist.get("crypto", [])
    try:
        crypto = fetch_all_crypto(crypto_list)
    except Exception as exc:
        logger.error("fetch_all_crypto failed: %s", exc)

    # Fear & Greed
    try:
        fear_greed = fetch_fear_greed()
    except Exception as exc:
        logger.error("fetch_fear_greed failed: %s", exc)
        fear_greed = {
            "value": None, "value_classification": None,
            "timestamp": None, "fetch_time": _now_iso(),
            "is_stale": True, "error": str(exc),
        }

    # Global Snapshot
    global_snapshot = {}
    try:
        global_snapshot = fetch_global_snapshot()
    except Exception as exc:
        logger.error("fetch_global_snapshot failed: %s", exc)

    end = datetime.now(tz=timezone.utc)
    elapsed = (end - start).total_seconds()
    logger.info("=== fetch_all_market_data complete in %.1fs ===", elapsed)

    return {
        "traditional":     traditional,
        "crypto":          crypto,
        "fear_greed":      fear_greed,
        "global_snapshot": global_snapshot,
        "fetch_time":      end.isoformat(),
    }


def summarise_market_data(market_data: dict, *, include_ohlcv: bool = False) -> str:
    """Return human-readable summary of the loaded market data."""
    lines = []
    fetch_time = market_data.get("fetch_time", "unknown")
    lines.append(f"Market Data Snapshot - {fetch_time}")
    lines.append("=" * 60)

    def _fmt_asset(a: dict) -> str:
        price = a.get("price", float("nan"))
        chg24 = a.get("change_24h_pct", float("nan"))
        stale = "⚠ STALE" if a.get("is_stale") else ""
        ohlcv_info = ""
        if include_ohlcv:
            df = a.get("ohlcv", pd.DataFrame())
            ohlcv_info = f"  OHLCV: {df.shape[0]} rows"
        price_str = f"{price:>12.4f}" if not np.isnan(price) else f"{'N/A':>12}"
        chg_str = f"{chg24:>+7.2f}%" if not np.isnan(chg24) else f"{'N/A':>8}"
        return f"  {a.get('ticker', '?'):20s} {price_str}  {chg_str}  {stale}{ohlcv_info}"

    trad = market_data.get("traditional", {})
    if trad:
        lines.append(f"\nTraditional ({len(trad)} assets):")
        for _t, a in trad.items():
            lines.append(_fmt_asset(a))

    crypto = market_data.get("crypto", {})
    if crypto:
        lines.append(f"\nCrypto ({len(crypto)} assets):")
        for _cid, a in crypto.items():
            lines.append(_fmt_asset(a))

    return "\n".join(lines)
