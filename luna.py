"""
agent.py
--------
Main orchestration entry point for the autonomous trading research agent.
Coordinates all modules: market data, indicators, research, scanning, macro
analysis, reporting, alert detection, and notifications.

Usage:
    python agent.py --run           # single full cycle
    python agent.py --quick         # price + alerts only
    python agent.py --macro         # macro dashboard update only
    python agent.py --scan          # opportunity scan only
    python agent.py --alert-check   # price refresh + alert check
    python agent.py --schedule      # continuous scheduled mode
    python agent.py --daily-brief   # push daily digest immediately
"""

import argparse
import json
import logging
import os
import textwrap
import time
from datetime import datetime, timezone
from typing import Optional

import schedule
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Module imports — all trading-agent sub-modules
# ---------------------------------------------------------------------------
from market_data import fetch_all_market_data
from indicators import enrich_all_assets
from researcher import fetch_all_research
from scanner import run_full_scan
from macro import analyze_macro, fetch_treasury_data
from reporter import (
    generate_all_reports,
    load_state,
    save_state,
    REPORTS_DIR,
    STATE_DIR,
)
from notifier import notify_run_complete, notify_alert, notify_daily_summary

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

class ColoredFormatter(logging.Formatter):
    GREY = "\033[90m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD_RED = "\033[1;31m"
    RESET = "\033[0m"
    MAGENTA = "\033[95m"
    CYAN = "\033[36m"
    
    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: BLUE,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED
    }

    def format(self, record):
        lvl_color = self.LEVEL_COLORS.get(record.levelno, '')
        log_fmt = f"%(asctime)s [{lvl_color}%(levelname)s{self.RESET}] \033[36m%(name)s\033[0m: %(message)s"
        
        msg = str(record.msg)
        if "Step " in msg:
            msg = f"\033[1m\033[95m{msg}\033[0m"
        elif "completed" in msg or "successful" in msg or "✓" in msg:
            msg = f"\033[92m{msg}\033[0m"
        elif "failed" in msg or "error" in msg or "🚨" in msg:
            msg = f"\033[91m{msg}\033[0m"
            
        orig_msg = record.msg
        record.msg = msg
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        res = formatter.format(record)
        record.msg = orig_msg
        return res

# Configure logging handlers
file_handler = logging.FileHandler("trading-agent.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(ColoredFormatter())

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

logger = logging.getLogger("agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WATCHLIST_PATH = os.path.join(os.path.dirname(__file__), "watchlist.json")
ENV_PATH       = os.path.join(os.path.dirname(__file__), ".env")

# Alert thresholds
PRICE_MOVE_THRESHOLD_PCT = 3.0   # percent
RSI_OVERSOLD             = 30.0
RSI_OVERBOUGHT           = 70.0
VIX_SPIKE_THRESHOLD_PCT  = 20.0  # percent intra-session
DXY_MOVE_THRESHOLD_PCT   = 0.5   # percent
YIELD_MOVE_THRESHOLD_BPS = 10.0  # basis points


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """
    Load environment variables from .env (if present) and return them as a dict.

    Keys typically expected:
        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
        FRED_API_KEY, CRYPTOPANIC_API_KEY, etc.

    Returns
    -------
    dict: All env variables visible to the process (plus any loaded from .env).
    """
    if os.path.exists(ENV_PATH):
        load_dotenv(dotenv_path=ENV_PATH, override=False)
        logger.debug("Loaded .env from %s", ENV_PATH)
    else:
        logger.debug(".env file not found at %s — relying on system env vars.", ENV_PATH)

    config = dict(os.environ)
    logger.info("Config loaded (%d environment variables available).", len(config))
    return config


def load_watchlist() -> dict:
    """
    Load the watchlist from watchlist.json.

    Expected structure:
    {
        "stocks":      ["AAPL", "MSFT", ...],
        "crypto":      ["bitcoin", "ethereum", ...],   # CoinGecko IDs
        "forex":       ["EURUSD=X", ...],
        "commodities": ["GC=F", "CL=F", ...],
        "bonds":       ["^TNX", ...],
        "indices":     ["^GSPC", "^NSEI", ...],
        "etfs":        ["SPY", "QQQ", ...],
        "macro":       ["^VIX", "^TNX", "DX-Y.NYB", ...]
    }

    Returns
    -------
    dict: Watchlist grouped by asset class. Returns {} on failure.
    """
    if not os.path.exists(WATCHLIST_PATH):
        logger.warning(
            "watchlist.json not found at %s — returning empty watchlist.", WATCHLIST_PATH
        )
        return {}

    try:
        with open(WATCHLIST_PATH, "r", encoding="utf-8") as fh:
            watchlist = json.load(fh)
        total = sum(len(v) for v in watchlist.values() if isinstance(v, list))
        logger.info("Watchlist loaded: %d assets across %d categories.", total, len(watchlist))
        return watchlist
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse watchlist.json: %s", exc)
        return {}
    except OSError as exc:
        logger.error("Cannot read watchlist.json: %s", exc)
        return {}


def _build_research_config(watchlist: dict, config: dict) -> dict:
    """
    Merge watchlist asset info into a config dict for fetch_all_research().

    fetch_all_research(config) expects:
        fred_api_key, cryptopanic_api_key, assets (list of {name, ticker}),
        include_calendar, include_fear_greed, include_treasury.
    """
    # Collect flat list of {name, ticker} for sentiment analysis
    assets: list[dict] = []

    # Traditional assets: stocks, indices, forex, commodities, bonds, etfs
    for category in ("stocks", "indices", "forex", "commodities", "bonds", "etfs"):
        for ticker in watchlist.get(category, []):
            if isinstance(ticker, str):
                assets.append({"name": ticker, "ticker": ticker})

    # Crypto: may be a list of CoinGecko IDs (strings) or {id, symbol} dicts
    for item in watchlist.get("crypto", []):
        if isinstance(item, str):
            assets.append({"name": item, "ticker": item.upper()})
        elif isinstance(item, dict):
            assets.append({
                "name":   item.get("id", item.get("name", "")),
                "ticker": item.get("symbol", item.get("id", "")).upper(),
            })

    return {
        **config,
        "fred_api_key":         config.get("FRED_API_KEY") or config.get("fred_api_key"),
        "cryptopanic_api_key":  config.get("CRYPTOPANIC_API_KEY") or config.get("cryptopanic_api_key"),
        "assets":               assets,
        "include_calendar":     True,
        "include_fear_greed":   True,
        "include_treasury":     True,
    }


def _build_enriched_flat(market_data: dict) -> dict:
    """
    Flatten fetch_all_market_data() output into a flat ticker→asset dict
    suitable for enrich_all_assets() and scan_all_assets().

    market_data structure:
        {
            "traditional": { "AAPL": {...}, "^GSPC": {...}, ... },
            "crypto":      { "bitcoin": {...}, "ethereum": {...}, ... },
            "fear_greed":  {...},
            "global_snapshot": {...},
            "fetch_time":  "...",
        }
    """
    flat: dict = {}

    # Traditional assets
    for ticker, asset in market_data.get("traditional", {}).items():
        flat[ticker] = asset

    # Crypto assets
    for coin_id, asset in market_data.get("crypto", {}).items():
        # Use the ticker symbol (e.g. "BTC-USD") as key if available, else coin_id
        key = asset.get("ticker", coin_id)
        flat[key] = asset

    # Include global snapshot tickers if not already present
    for ticker, asset in market_data.get("global_snapshot", {}).items():
        if ticker not in flat:
            flat[ticker] = asset

    return flat


def _flatten_enriched_for_reports(enriched_data: dict) -> dict:
    """
    Group flat enriched_data back into category buckets for reporter.generate_watchlist_status().

    Returns:
        {
            "stocks":      [...],
            "crypto":      [...],
            "forex":       [...],
            "commodities": [...],
            "indices":     [...],
            "bonds":       [...],
            "etfs":        [...],
        }
    """
    categories: dict[str, list] = {
        "stocks": [], "crypto": [], "forex": [],
        "commodities": [], "indices": [], "bonds": [], "etfs": [],
    }
    for ticker, asset in enriched_data.items():
        if not isinstance(asset, dict):
            continue
        ac = asset.get("asset_class", "").lower()
        if ac in categories:
            categories[ac].append(asset)
        elif "stock" in ac:
            categories["stocks"].append(asset)
        elif "crypto" in ac:
            categories["crypto"].append(asset)
        elif "forex" in ac or "currency" in ac:
            categories["forex"].append(asset)
        elif "commodity" in ac or "oil" in ac or "gold" in ac:
            categories["commodities"].append(asset)
        elif "index" in ac or "indice" in ac:
            categories["indices"].append(asset)
        elif "bond" in ac or "treasury" in ac or "yield" in ac:
            categories["bonds"].append(asset)
        elif "etf" in ac:
            categories["etfs"].append(asset)
        else:
            categories["stocks"].append(asset)  # fallback bucket

    return {k: v for k, v in categories.items() if v}


def _build_market_data_for_report(market_data: dict, enriched_data: dict) -> dict:
    """
    Build the market_data dict in the shape reporter.generate_daily_brief() expects:
        {
            "global_snapshot": [list of asset dicts],
            "top_movers":      {"gainers": [...], "losers": [...]},
        }
    """
    # Assemble global_snapshot list from flat enriched or raw global_snapshot
    snapshot_assets: list[dict] = []
    gs = market_data.get("global_snapshot", {})
    for ticker, asset in gs.items():
        snap = {
            "ticker":    ticker,
            "name":      asset.get("name", ticker),
            "price":     asset.get("price"),
            "change_24h": asset.get("change_24h_pct"),
            "change_7d":  asset.get("change_7d_pct"),
        }
        snapshot_assets.append(snap)

    # Top movers from flat enriched data
    movers: list[dict] = []
    for ticker, asset in enriched_data.items():
        chg = asset.get("change_24h_pct") or asset.get("change_24h")
        if chg is not None:
            try:
                movers.append({"ticker": ticker, "change_24h": float(chg)})
            except (TypeError, ValueError):
                pass

    movers.sort(key=lambda x: x["change_24h"], reverse=True)
    gainers = movers[:3] if movers else []
    losers  = movers[-3:][::-1] if movers else []

    return {
        "global_snapshot": snapshot_assets,
        "top_movers": {
            "gainers": gainers,
            "losers":  losers,
        },
    }


def _build_macro_state_flat(macro_result: dict) -> dict:
    """
    Flatten the nested macro.analyze_macro() result into the flat dict that
    reporter.generate_macro_dashboard() and check_alerts() expect.

    macro.analyze_macro() returns:
        {
            "timestamp": ...,
            "regime": {"regime": "RISK-ON", "risk_on_score": 4, ...},
            "vix":    {"current": 17.3, "change_pct": 1.2, "spiked": False, ...},
            "dollar": {"level": 104.2, "trend": "rising", "change_5d_pct": 0.3, ...},
            "rates":  {"yield_10y": 4.5, "yield_2y": 5.1, "yield_curve": "inverted", ...},
            "inflation": {"cpi": 3.4, "pce": 2.7, "trend": "decelerating", ...},
            "intermarket": {...},
            "btc_dominance": {"btc_dominance": 53.2, "trend": "rising", ...},
            "macro_summary": "...",
        }
    """
    regime_dict      = macro_result.get("regime", {})
    vix_dict         = macro_result.get("vix", {})
    dollar_dict      = macro_result.get("dollar", {})
    rates_dict       = macro_result.get("rates", {})
    inflation_dict   = macro_result.get("inflation", {})
    btc_dom_dict     = macro_result.get("btc_dominance", {})

    return {
        # Top-level regime
        "regime":                  regime_dict.get("regime", "UNKNOWN"),
        "risk_on_score":           regime_dict.get("risk_on_score", 0),
        "risk_off_score":          regime_dict.get("risk_off_score", 0),
        "risk_on_signals":         regime_dict.get("risk_on_signals", []),
        "risk_off_signals":        regime_dict.get("risk_off_signals", []),
        "regime_changed":          regime_dict.get("regime_changed", False),

        # VIX
        "vix":                     vix_dict.get("current"),
        "vix_change_pct":          vix_dict.get("change_pct"),
        "vix_spiked":              vix_dict.get("spiked", False),
        "vix_status":              vix_dict.get("status", ""),

        # DXY
        "dxy":                     dollar_dict.get("level"),
        "dxy_trend":               dollar_dict.get("trend", "unknown"),
        "dxy_change_5d_pct":       dollar_dict.get("change_5d_pct"),
        "dxy_impact":              dollar_dict.get("impact", ""),

        # Yields
        "yield_10y":               rates_dict.get("yield_10y"),
        "yield_2y":                rates_dict.get("yield_2y"),
        "yield_30y":               rates_dict.get("yield_30y"),
        "yield_curve":             rates_dict.get("yield_curve", "unknown"),
        "yield_curve_inverted":    rates_dict.get("yield_curve_inverted", False),
        "yield_curve_spread":      rates_dict.get("yield_curve_spread_bps"),
        "rate_trend":              rates_dict.get("rate_trend", "unknown"),
        "rates_impact":            rates_dict.get("impact", ""),

        # Inflation
        "cpi":                     inflation_dict.get("cpi"),
        "pce":                     inflation_dict.get("pce"),
        "inflation_trend":         inflation_dict.get("trend", "unknown"),
        "inflation_impact":        inflation_dict.get("impact", ""),

        # BTC dominance
        "btc_dominance":           btc_dom_dict.get("btc_dominance"),
        "btc_dominance_trend":     btc_dom_dict.get("trend", "unknown"),
        "btc_dominance_impact":    btc_dom_dict.get("impact", ""),

        # Intermarket
        "intermarket":             macro_result.get("intermarket", {}),
        "correlations":            macro_result.get("correlations", {}),
        "macro_summary":           macro_result.get("macro_summary", ""),
        "timestamp":               macro_result.get("timestamp", ""),
        "sector_rotation":         macro_result.get("sector_rotation", {}),

        # Aliases for alert_check compatibility
        "yields": {
            "US10Y": rates_dict.get("yield_10y"),
            "US2Y":  rates_dict.get("yield_2y"),
            "US30Y": rates_dict.get("yield_30y"),
        },
    }


def _build_research_for_report(research: dict | None) -> dict:
    """
    Normalise the research dict into the shape reporter expects.

    researcher.fetch_all_research() returns:
        {
            "news": [...],
            "cryptopanic_news": [...],
            "fear_greed": {...},
            "fred": {...},
            "treasury_yields": {...},
            "economic_calendar": [...],
            "asset_sentiment": {...},
            "overall_sentiment": float,
        }

    reporter expects research_data with:
        headlines, risks, events_this_week, calendar, treasury_data, fred_data
    """
    if not research:
        return {
            "headlines": [],
            "risks": ["Research data unavailable for this cycle."],
            "events_this_week": [],
            "calendar": [],
            "treasury_data": {},
            "fred_data": {},
        }

    # Map news articles to the headline format reporter expects
    news = research.get("news", []) + research.get("cryptopanic_news", [])
    headlines = [
        {
            "title":  a.get("title", ""),
            "source": a.get("source", "Unknown"),
            "impact": a.get("summary", "See article for details."),
        }
        for a in news[:10]
    ]

    # Build events_this_week from economic_calendar
    calendar = research.get("economic_calendar", [])
    events_this_week = [
        {
            "date":     ev.get("date", ""),
            "event":    ev.get("event", ""),
            "currency": ev.get("currency", ""),
        }
        for ev in calendar[:10]
        if ev.get("impact", "").lower() in ("high", "medium")
    ]

    # Build treasury_data in the maturity→rate format reporter expects
    treasury_raw = research.get("treasury_yields", {})
    treasury_yields_dict = treasury_raw.get("yields", {})
    treasury_data = {
        mat.lower(): val
        for mat, val in treasury_yields_dict.items()
        if val is not None
    }

    # Build fred_data in the key→float format reporter expects
    fred_raw = research.get("fred") or {}
    fred_data: dict = {}
    for key, val_dict in fred_raw.items():
        if isinstance(val_dict, dict):
            fred_data[key] = val_dict.get("latest")
        else:
            fred_data[key] = val_dict

    # Add fear_greed from research if available
    fg = research.get("fear_greed", {})

    return {
        "headlines":         headlines,
        "risks":             _build_risks(research),
        "events_this_week":  events_this_week,
        "calendar":          calendar,
        "treasury_data":     treasury_data,
        "fred_data":         fred_data,
        "fear_greed":        fg,
        "overall_sentiment": research.get("overall_sentiment", 0.0),
        "asset_sentiment":   research.get("asset_sentiment", {}),
    }


def _build_risks(research: dict) -> list:
    """Derive key risks from research data."""
    risks: list[str] = []
    cal = research.get("economic_calendar", [])
    for ev in cal:
        if ev.get("impact", "").lower() == "high":
            risks.append(
                f"High-impact event: {ev.get('event', '?')} "
                f"({ev.get('currency', '')} — {ev.get('date', '')})"
            )
    fg = research.get("fear_greed", {})
    score = fg.get("score")
    if score is not None:
        if score < 25:
            risks.append(f"Extreme Fear in markets (Fear & Greed: {score:.0f}/100)")
        elif score > 80:
            risks.append(f"Extreme Greed — correction risk elevated (Fear & Greed: {score:.0f}/100)")
    if not risks:
        risks = ["No specific risks flagged at this time."]
    return risks[:5]


def _opportunities_to_list(ranked: dict) -> list[dict]:
    """
    Convert rank_opportunities() dict result into a flat list of opportunity dicts.

    rank_opportunities() returns:
        {
            "bullish": [...],  # each is a score_result dict from score_asset()
            "bearish": [...],
            "neutral": [...],
            "breakouts": [...], "breakdowns": [...], ...
            "summary": {...},
        }
    """
    all_opps: list[dict] = []

    for opp in ranked.get("bullish", []):
        all_opps.append(_format_opp(opp, "bullish"))

    for opp in ranked.get("bearish", []):
        all_opps.append(_format_opp(opp, "bearish"))

    # Tag breakout/breakdown/etc. candidates onto existing opps or add as new
    breakout_tickers = {b["ticker"] for b in ranked.get("breakouts", [])}
    squeeze_tickers  = {s["ticker"] for s in ranked.get("squeezes", [])}
    momentum_tickers = {m["ticker"] for m in ranked.get("momentum", [])}

    for opp in all_opps:
        t = opp.get("ticker", "")
        if t in breakout_tickers:
            opp["tag"] = "breakout"
            opp["new_high"] = False
        if t in squeeze_tickers:
            opp["bb_squeeze"] = True
        if t in momentum_tickers:
            opp["new_high"] = True

    # Sort by abs(score) descending
    all_opps.sort(key=lambda x: abs(x.get("score", 0)), reverse=True)
    return all_opps


def _format_opp(score_result: dict, bias: str) -> dict:
    """
    Convert a scanner.score_asset() result into the opportunity dict
    format that reporter.build_opportunity_block() expects.
    """
    ticker   = score_result.get("ticker", "?")
    signals  = score_result.get("signals", [])
    score    = score_result.get("score", 0)
    breakdown = score_result.get("breakdown_str", "N/A")

    from memory import get_asset_context_summary
    context_summary = get_asset_context_summary(ticker, bias)
    
    primary_signals = "; ".join(signals[:3]) if signals else "See score breakdown."
    if context_summary:
        reasoning = f"{primary_signals} — {context_summary}"
    else:
        reasoning = primary_signals

    return {
        "ticker":           ticker,
        "name":             score_result.get("name", ticker),
        "asset_class":      score_result.get("asset_class", "N/A"),
        "price":            score_result.get("price"),
        "change_24h":       score_result.get("change_24h_pct") or score_result.get("change_24h"),
        "score":            score,
        "score_breakdown":  breakdown,
        "support":          score_result.get("support"),
        "resistance":       score_result.get("resistance"),
        "reasoning":        reasoning,
        "catalyst":         "; ".join(signals[3:5]) if len(signals) > 3 else "N/A",
        "invalidation_level": score_result.get("invalidation_level"),
        "bias":             bias,
        "new_high":         score_result.get("new_high", False),
        "bb_squeeze":       score_result.get("bb_squeeze", False),
        "tag":              score_result.get("tag", ""),
        "correlation_break": score_result.get("correlation_break", False),
        "correlation_note": score_result.get("correlation_note", ""),
    }


# ---------------------------------------------------------------------------
# Alert detection
# ---------------------------------------------------------------------------

def check_alerts(
    enriched_data: dict,
    macro_state: dict,
    last_state: Optional[dict],
) -> list:
    """
    Scan current enriched market data and macro state for noteworthy events.

    Detects:
    - Any watchlist asset that moved >3% since last check
    - RSI crossed 30 (oversold) or 70 (overbought)
    - Price broke above resistance or below support
    - EMA golden cross / death cross
    - Macro regime change
    - VIX spike >20%
    - DXY moved >0.5%
    - Bond yields moved >10 basis points
    """
    alerts = []

    if last_state is None:
        logger.info("First run — no previous state to compare against. Skipping alert checks.")
        return alerts

    prev_assets  = last_state.get("assets", {})
    prev_macro   = last_state.get("macro_state", {})
    prev_regime  = prev_macro.get("regime", "")
    prev_yields  = prev_macro.get("yields", {})
    prev_vix     = prev_macro.get("vix", {}) if isinstance(prev_macro.get("vix"), dict) else {}
    prev_dxy     = prev_macro.get("dxy", {}) if isinstance(prev_macro.get("dxy"), dict) else {}

    # ------------------------------------------------------------------
    # 1. Per-asset checks
    # ------------------------------------------------------------------
    for symbol, data in enriched_data.items():
        if not isinstance(data, dict):
            continue

        current_price  = data.get("price")
        current_rsi    = data.get("rsi")
        ema_short      = data.get("ema20") or data.get("ema_short")
        ema_long       = data.get("ema50") or data.get("ema_long")
        support        = data.get("support")
        resistance     = data.get("resistance")

        prev_asset     = prev_assets.get(symbol, {})
        prev_price     = prev_asset.get("price")
        prev_rsi       = prev_asset.get("rsi")
        prev_ema_short = prev_asset.get("ema_short")
        prev_ema_long  = prev_asset.get("ema_long")

        # ---- 1a. Large price move ----
        if current_price is not None and prev_price is not None and prev_price != 0:
            try:
                pct_change = ((float(current_price) - float(prev_price)) / abs(float(prev_price))) * 100.0
                if abs(pct_change) >= PRICE_MOVE_THRESHOLD_PCT:
                    direction = "UP" if pct_change > 0 else "DOWN"
                    severity  = "HIGH" if abs(pct_change) >= 5.0 else "MEDIUM"
                    alerts.append({
                        "type":     "PRICE_MOVE",
                        "asset":    symbol,
                        "message":  f"Moved {pct_change:+.2f}% since last check ({direction})",
                        "severity": severity,
                        "context":  f"Prev: {float(prev_price):.4f}  →  Now: {float(current_price):.4f}",
                    })
            except (TypeError, ValueError):
                pass

        # ---- 1b. RSI threshold crossings ----
        if current_rsi is not None and prev_rsi is not None:
            try:
                cr, pr = float(current_rsi), float(prev_rsi)
                if cr <= RSI_OVERSOLD < pr:
                    alerts.append({
                        "type": "RSI_OVERSOLD", "asset": symbol,
                        "message": f"RSI crossed into oversold ({cr:.1f} ≤ {RSI_OVERSOLD})",
                        "severity": "MEDIUM", "context": f"RSI: {pr:.1f} → {cr:.1f}",
                    })
                elif cr >= RSI_OVERBOUGHT > pr:
                    alerts.append({
                        "type": "RSI_OVERBOUGHT", "asset": symbol,
                        "message": f"RSI crossed into overbought ({cr:.1f} ≥ {RSI_OVERBOUGHT})",
                        "severity": "MEDIUM", "context": f"RSI: {pr:.1f} → {cr:.1f}",
                    })
                elif pr <= RSI_OVERSOLD < cr:
                    alerts.append({
                        "type": "RSI_RECOVERY", "asset": symbol,
                        "message": f"RSI recovering from oversold ({pr:.1f} → {cr:.1f})",
                        "severity": "LOW", "context": "Potential bullish reversal signal.",
                    })
                elif pr >= RSI_OVERBOUGHT > cr:
                    alerts.append({
                        "type": "RSI_PULLBACK", "asset": symbol,
                        "message": f"RSI pulling back from overbought ({pr:.1f} → {cr:.1f})",
                        "severity": "LOW", "context": "Potential bearish reversal signal.",
                    })
            except (TypeError, ValueError):
                pass

        # ---- 1c. Breakout above resistance ----
        if current_price is not None and resistance is not None and prev_price is not None:
            try:
                if float(prev_price) <= float(resistance) < float(current_price):
                    alerts.append({
                        "type": "BREAKOUT_RESISTANCE", "asset": symbol,
                        "message": f"Price broke ABOVE resistance at {float(resistance):.4f}",
                        "severity": "HIGH",
                        "context": f"Prev: {float(prev_price):.4f}  →  Now: {float(current_price):.4f}",
                    })
            except (TypeError, ValueError):
                pass

        # ---- 1d. Breakdown below support ----
        if current_price is not None and support is not None and prev_price is not None:
            try:
                if float(prev_price) >= float(support) > float(current_price):
                    alerts.append({
                        "type": "BREAKDOWN_SUPPORT", "asset": symbol,
                        "message": f"Price broke BELOW support at {float(support):.4f}",
                        "severity": "HIGH",
                        "context": f"Prev: {float(prev_price):.4f}  →  Now: {float(current_price):.4f}",
                    })
            except (TypeError, ValueError):
                pass

        # ---- 1e/1f. EMA crosses ----
        if all(x is not None for x in [ema_short, ema_long, prev_ema_short, prev_ema_long]):
            try:
                es, el, pes, pel = (
                    float(ema_short), float(ema_long),
                    float(prev_ema_short), float(prev_ema_long),
                )
                if pes <= pel and es > el:
                    alerts.append({
                        "type": "EMA_GOLDEN_CROSS", "asset": symbol,
                        "message": "EMA Golden Cross — short EMA crossed above long EMA (bullish)",
                        "severity": "MEDIUM",
                        "context": f"EMA_short {pes:.4f}→{es:.4f}  EMA_long {pel:.4f}→{el:.4f}",
                    })
                elif pes >= pel and es < el:
                    alerts.append({
                        "type": "EMA_DEATH_CROSS", "asset": symbol,
                        "message": "EMA Death Cross — short EMA crossed below long EMA (bearish)",
                        "severity": "HIGH",
                        "context": f"EMA_short {pes:.4f}→{es:.4f}  EMA_long {pel:.4f}→{el:.4f}",
                    })
            except (TypeError, ValueError):
                pass

    # ------------------------------------------------------------------
    # 2. Macro-level checks
    # ------------------------------------------------------------------

    # ---- 2a. Regime change ----
    current_regime = macro_state.get("regime", "")
    if prev_regime and current_regime and current_regime != prev_regime:
        alerts.append({
            "type": "REGIME_CHANGE", "asset": "MACRO",
            "message": f"Macro regime changed: {prev_regime} → {current_regime}",
            "severity": "HIGH",
            "context": "Review all open positions and opportunity rankings.",
        })

    # ---- 2b. VIX spike ----
    cur_vix_val  = macro_state.get("vix") if not isinstance(macro_state.get("vix"), dict) else macro_state["vix"].get("current")
    prev_vix_val = prev_vix.get("current") or prev_vix.get("value")
    if cur_vix_val is not None and prev_vix_val is not None:
        try:
            vix_pct = ((float(cur_vix_val) - float(prev_vix_val)) / float(prev_vix_val)) * 100.0
            if vix_pct >= VIX_SPIKE_THRESHOLD_PCT:
                alerts.append({
                    "type": "VIX_SPIKE", "asset": "VIX",
                    "message": f"VIX spiked {vix_pct:+.1f}% in session (risk-off signal)",
                    "severity": "HIGH",
                    "context": f"VIX: {float(prev_vix_val):.2f} → {float(cur_vix_val):.2f}",
                })
        except (TypeError, ValueError):
            pass

    # ---- 2c. DXY move ----
    cur_dxy_val  = macro_state.get("dxy") if not isinstance(macro_state.get("dxy"), dict) else macro_state["dxy"].get("level")
    prev_dxy_val = prev_dxy.get("level") or prev_dxy.get("value")
    if cur_dxy_val is not None and prev_dxy_val is not None:
        try:
            dxy_pct = ((float(cur_dxy_val) - float(prev_dxy_val)) / float(prev_dxy_val)) * 100.0
            if abs(dxy_pct) >= DXY_MOVE_THRESHOLD_PCT:
                direction = "UP" if dxy_pct > 0 else "DOWN"
                alerts.append({
                    "type": "DXY_MOVE", "asset": "DXY",
                    "message": f"US Dollar (DXY) moved {dxy_pct:+.2f}% in session ({direction})",
                    "severity": "MEDIUM",
                    "context": f"DXY: {float(prev_dxy_val):.3f} → {float(cur_dxy_val):.3f}",
                })
        except (TypeError, ValueError):
            pass

    # ---- 2d. Bond yield moves ----
    current_yields = macro_state.get("yields", {})
    all_yield_keys = set(current_yields.keys()) | set(prev_yields.keys())
    for tenor in all_yield_keys:
        cur_y  = current_yields.get(tenor)
        prev_y = prev_yields.get(tenor)
        if cur_y is None or prev_y is None:
            continue
        try:
            move_bps = (float(cur_y) - float(prev_y)) * 100.0
            if abs(move_bps) >= YIELD_MOVE_THRESHOLD_BPS:
                direction = "HIGHER" if move_bps > 0 else "LOWER"
                severity  = "HIGH" if abs(move_bps) >= 20 else "MEDIUM"
                alerts.append({
                    "type": "YIELD_MOVE", "asset": tenor,
                    "message": f"{tenor} yield moved {move_bps:+.1f}bps {direction}",
                    "severity": severity,
                    "context": f"{tenor}: {float(prev_y):.3f}% → {float(cur_y):.3f}%",
                })
        except (TypeError, ValueError):
            pass

    logger.info(
        "Alert check complete: %d alerts (%d HIGH, %d MEDIUM, %d LOW).",
        len(alerts),
        sum(1 for a in alerts if a.get("severity") == "HIGH"),
        sum(1 for a in alerts if a.get("severity") == "MEDIUM"),
        sum(1 for a in alerts if a.get("severity") == "LOW"),
    )
    return alerts


# ---------------------------------------------------------------------------
# Step timer helpers
# ---------------------------------------------------------------------------

def _log_step(step: str, step_num: int, total: int) -> float:
    logger.info("── Step %d/%d: %s", step_num, total, step)
    return time.monotonic()


def _log_done(step: str, t0: float) -> None:
    logger.info("   ✓ %s completed in %.2fs", step, time.monotonic() - t0)


# ---------------------------------------------------------------------------
# Run cycles
# ---------------------------------------------------------------------------

def run_full_cycle(config: dict) -> None:
    """
    Execute a complete research & analysis cycle.

    Steps:
        1.  Load watchlist
        2.  Load last state (for alert diffing)
        3.  Fetch all market data
        4.  Enrich with technical indicators
        5.  Fetch research (news, calendar, sentiment, FRED)
        6.  Run full scan (score + pattern detection + rank)
        7.  Analyze macro regime
        8.  Generate all reports (Markdown files)
        9.  Check alerts vs previous state
        10. Send notifications (console + Telegram)
        11. Save new state
    """
    cycle_start = time.monotonic()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("═" * 60)
    logger.info("FULL CYCLE STARTED at %s", ts)
    logger.info("═" * 60)

    TOTAL_STEPS = 11

    try:
        # Step 1 — Watchlist
        t0 = _log_step("Load watchlist", 1, TOTAL_STEPS)
        watchlist = load_watchlist()
        _log_done("Load watchlist", t0)

        # Step 2 — Last state
        t0 = _log_step("Load last state", 2, TOTAL_STEPS)
        last_state = load_state()
        
        # Check pending call outcomes (Module 6 outcome checks & weight tuning)
        try:
            from calls_tracker import check_pending_outcomes
            check_pending_outcomes()
        except Exception as exc:
            logger.error("Failed to check pending outcomes: %s", exc)
            
        _log_done("Load last state", t0)

        # Step 3 — Market data
        t0 = _log_step("Fetch all market data", 3, TOTAL_STEPS)
        market_data = fetch_all_market_data(watchlist)   # no config param needed
        if not market_data:
            logger.warning("No market data returned — aborting cycle.")
            return
        _log_done("Fetch all market data", t0)

        # Step 4 — Flatten + indicators
        t0 = _log_step("Calculate technical indicators", 4, TOTAL_STEPS)
        flat_data     = _build_enriched_flat(market_data)
        enriched_data = enrich_all_assets(flat_data)    # no config param needed
        
        # Update asset memory (Module 8)
        try:
            from memory import update_asset_memory
            update_asset_memory(enriched_data)
        except Exception as exc:
            logger.error("Failed to update asset memory: %s", exc)
            
        _log_done("Calculate technical indicators", t0)

        # Step 5 — Research
        t0 = _log_step("Fetch research (news/calendar/sentiment/FRED)", 5, TOTAL_STEPS)
        research_config = _build_research_config(watchlist, config)
        research_config["return_dict"] = True  # Enable rich credibility sentiment details
        research_raw    = fetch_all_research(research_config)
        research        = _build_research_for_report(research_raw)
        
        # Merge sentiment scores into enriched_data (Module 11 integration)
        try:
            asset_sentiment = research.get("asset_sentiment", {})
            for ticker, asset in enriched_data.items():
                sentiment_val = 0.0
                sentiment_breakdown = "No news"
                sentiment_unverified = False
                
                # Find matching key in asset_sentiment
                for sent_k, val_dict in asset_sentiment.items():
                    if f"({ticker})" in sent_k or sent_k == ticker or (ticker.endswith(".NS") and f"({ticker[:-3]})" in sent_k):
                        if isinstance(val_dict, dict):
                            sentiment_val = val_dict.get("score", 0.0)
                            sentiment_breakdown = val_dict.get("breakdown", "No news")
                            sentiment_unverified = val_dict.get("unverified", False)
                        else:
                            sentiment_val = float(val_dict)
                            sentiment_breakdown = f"Sentiment: {sentiment_val:+.1f}"
                        break
                
                asset["sentiment"] = sentiment_val
                asset["sentiment_score"] = sentiment_val
                asset["sentiment_breakdown"] = sentiment_breakdown
                asset["sentiment_unverified"] = sentiment_unverified
        except Exception as exc:
            logger.error("Failed to merge sentiment data: %s", exc)
            
        _log_done("Fetch research", t0)

        # Step 6 — Macro
        t0 = _log_step("Analyze macro regime", 6, TOTAL_STEPS)
        treasury_data = fetch_treasury_data()
        fred_data     = research_raw.get("fred") or {}
        macro_result  = analyze_macro(
            market_data   = {"regime_assets": {}},   # will fetch fresh internally
            treasury_data = treasury_data,
            fred_data     = fred_data,
            last_state    = last_state or {},
        )
        macro_state = _build_macro_state_flat(macro_result)
        logger.info("   Regime: %s", macro_state.get("regime", "UNKNOWN"))
        _log_done("Analyze macro", t0)

        # Step 7 — Full scan (score + patterns + rank)
        t0 = _log_step("Run full scan (score + rank + patterns)", 7, TOTAL_STEPS)
        calendar  = research.get("calendar", [])
        regime = macro_state.get("regime", "UNKNOWN")
        ranked    = run_full_scan(enriched_data, calendar, regime=regime)
        opps_list = _opportunities_to_list(ranked)
        logger.info("   %d opportunities ranked (%d bullish, %d bearish).",
                    len(opps_list),
                    len(ranked.get("bullish", [])),
                    len(ranked.get("bearish", [])))
                    
        # Log new setups to calls-log.json (Module 6 setups log)
        try:
            from calls_tracker import log_new_call
            for opp_cat in ["bullish", "bearish"]:
                for opp in ranked.get(opp_cat, []):
                    if opp.get("score") is not None and abs(opp.get("score", 0)) >= 6:
                        price = opp.get("price")
                        if price is not None:
                            # Construct opportunity in the shape expected by calls_tracker
                            opp_for_log = {
                                "ticker": opp.get("ticker"),
                                "direction": opp_cat,
                                "score": opp.get("score"),
                                "asset_class": opp.get("asset_class"),
                                "breakdown": opp.get("breakdown", {}),
                                "support_resistance": {
                                    "support": opp.get("support") if isinstance(opp.get("support"), list) else ([opp.get("support")] if opp.get("support") else []),
                                    "resistance": opp.get("resistance") if isinstance(opp.get("resistance"), list) else ([opp.get("resistance")] if opp.get("resistance") else [])
                                }
                            }
                            log_new_call(opp_for_log, price, regime)
        except Exception as exc:
            logger.error("Failed to log new setup call: %s", exc)
            
        _log_done("Full scan", t0)

        # Step 8 — Reports
        t0 = _log_step("Generate all reports", 8, TOTAL_STEPS)
        report_market_data = _build_market_data_for_report(market_data, enriched_data)
        enriched_by_cat    = _flatten_enriched_for_reports(enriched_data)
        alerts_for_state   = check_alerts(enriched_data, macro_state, last_state)

        generate_all_reports(
            market_data   = report_market_data,
            enriched_data = enriched_by_cat,
            opportunities = opps_list,
            macro_state   = macro_state,
            research_data = research,
            alerts        = alerts_for_state,
            config        = config,
        )
        _log_done("Generate all reports", t0)

        # Precalculate portfolio status before notification
        portfolio_status = None
        try:
            from portfolio import calculate_portfolio_status
            portfolio_status = calculate_portfolio_status(market_data=report_market_data)
        except Exception as exc:
            logger.error("Failed to precalculate portfolio status: %s", exc)

        # Step 9 — Alerts
        t0 = _log_step("Check alerts vs last state", 9, TOTAL_STEPS)
        alerts = alerts_for_state  # already computed above
        _log_done("Check alerts", t0)

        # Step 10 — Notifications
        t0 = _log_step("Send notifications", 10, TOTAL_STEPS)
        notify_run_complete(macro_state, opps_list, alerts, config, portfolio_status=portfolio_status)
        for alert in alerts:
            if alert.get("severity") == "HIGH":
                notify_alert(alert, config)
        _log_done("Send notifications", t0)

        # Step 11 — Save state
        t0 = _log_step("Save new state", 11, TOTAL_STEPS)
        save_state(
            market_data   = report_market_data,
            macro_state   = macro_state,
            opportunities = opps_list,
            timestamp     = datetime.now(timezone.utc),
            alerts        = alerts,
        )
        _log_done("Save new state", t0)

        # Update Catalyst Economic Calendar report
        try:
            logger.info("Updating LUNA Economic Calendar schedule...")
            from catalysts import generate_catalyst_calendar
            generate_catalyst_calendar()
            logger.info("LUNA Portfolio and Catalyst reports successfully compiled.")
        except Exception as exc:
            logger.error("Failed to compile catalyst reports in full cycle: %s", exc)

    except KeyboardInterrupt:
        logger.warning("Full cycle interrupted by user.")
        raise
    except Exception as exc:
        logger.exception("Full cycle failed with unexpected error: %s", exc)
        try:
            bot_token = config.get("TELEGRAM_BOT_TOKEN")
            chat_id   = config.get("TELEGRAM_CHAT_ID")
            if bot_token and chat_id:
                from notifier import send_telegram
                send_telegram(
                    f"🚨 *LUNS ERROR*\n`{str(exc)[:200]}`",
                    bot_token, chat_id,
                )
        except Exception:
            pass
    finally:
        elapsed = time.monotonic() - cycle_start
        logger.info("FULL CYCLE COMPLETE — total time: %.1fs", elapsed)
        logger.info("═" * 60)


def run_quick_cycle(config: dict) -> None:
    """
    Lightweight price + alert cycle (no news research, no scanner).

    Steps: load watchlist → load state → fetch prices → indicators →
           macro → generate macro dashboard → alerts → notify → save state.
    """
    cycle_start = time.monotonic()
    logger.info("── QUICK CYCLE STARTED at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    try:
        watchlist  = load_watchlist()
        last_state = load_state()

        market_data = fetch_all_market_data(watchlist)
        if not market_data:
            logger.warning("No market data returned — aborting quick cycle.")
            return

        flat_data     = _build_enriched_flat(market_data)
        enriched_data = enrich_all_assets(flat_data)

        treasury_data = fetch_treasury_data()
        macro_result  = analyze_macro(
            market_data   = {"regime_assets": {}},
            treasury_data = treasury_data,
            fred_data     = {},
            last_state    = last_state or {},
        )
        macro_state = _build_macro_state_flat(macro_result)

        # Generate macro dashboard and watchlist status only
        report_market_data = _build_market_data_for_report(market_data, enriched_data)
        enriched_by_cat    = _flatten_enriched_for_reports(enriched_data)

        generate_all_reports(
            market_data   = report_market_data,
            enriched_data = enriched_by_cat,
            opportunities = [],
            macro_state   = macro_state,
            research_data = {},
            alerts        = [],
            config        = config,
        )

        # Precalculate portfolio status before notification
        portfolio_status = None
        try:
            from portfolio import calculate_portfolio_status
            portfolio_status = calculate_portfolio_status(market_data=report_market_data)
        except Exception as exc:
            logger.error("Failed to precalculate portfolio status: %s", exc)

        alerts = check_alerts(enriched_data, macro_state, last_state)
        notify_run_complete(macro_state, [], alerts, config, portfolio_status=portfolio_status)
        for alert in alerts:
            if alert.get("severity") == "HIGH":
                notify_alert(alert, config)

        save_state(
            market_data   = report_market_data,
            macro_state   = macro_state,
            opportunities = [],
            timestamp     = datetime.now(timezone.utc),
            alerts        = alerts,
        )

        # Update Catalyst Economic Calendar report
        try:
            logger.info("Updating LUNA Economic Calendar schedule...")
            from catalysts import generate_catalyst_calendar
            generate_catalyst_calendar()
            logger.info("LUNA Portfolio and Catalyst reports successfully compiled.")
        except Exception as exc:
            logger.error("Failed to compile catalyst reports in quick cycle: %s", exc)

    except KeyboardInterrupt:
        logger.warning("Quick cycle interrupted by user.")
        raise
    except Exception as exc:
        logger.exception("Quick cycle failed: %s", exc)
    finally:
        logger.info("QUICK CYCLE COMPLETE — %.1fs", time.monotonic() - cycle_start)


def run_alert_check(config: dict) -> None:
    """
    Fast alert check: load last state, fetch current prices, detect alerts, notify.
    Does NOT fetch news, run scanner, or generate reports.
    """
    t_start = time.monotonic()
    logger.info("── ALERT CHECK at %s", datetime.now().strftime("%H:%M:%S"))

    try:
        watchlist  = load_watchlist()
        last_state = load_state()

        market_data = fetch_all_market_data(watchlist)
        if not market_data:
            logger.warning("No market data for alert check — skipping.")
            return

        flat_data     = _build_enriched_flat(market_data)
        enriched_data = enrich_all_assets(flat_data)

        treasury_data = fetch_treasury_data()
        macro_result  = analyze_macro(
            market_data   = {"regime_assets": {}},
            treasury_data = treasury_data,
            fred_data     = {},
            last_state    = last_state or {},
        )
        macro_state = _build_macro_state_flat(macro_result)
        alerts      = check_alerts(enriched_data, macro_state, last_state)

        if alerts:
            logger.info("%d alert(s) detected — sending notifications.", len(alerts))
            for alert in alerts:
                notify_alert(alert, config)
        else:
            logger.info("No alerts triggered.")

        # Persist updated prices so next check has fresh baseline
        save_state(
            market_data   = _build_market_data_for_report(market_data, enriched_data),
            macro_state   = macro_state,
            opportunities = [],
            timestamp     = datetime.now(timezone.utc),
            alerts        = alerts,
        )

    except Exception as exc:
        logger.exception("Alert check failed: %s", exc)
    finally:
        logger.info("ALERT CHECK DONE — %.1fs", time.monotonic() - t_start)


def run_macro_only(config: dict) -> None:
    """
    Fetch macro/regime assets, analyze macro, update macro-dashboard.md only.
    """
    t_start = time.monotonic()
    logger.info("── MACRO UPDATE at %s", datetime.now().strftime("%H:%M:%S"))

    try:
        watchlist = load_watchlist()
        last_state = load_state()

        # Fetch only macro assets to reduce API usage
        macro_watchlist = {
            "macro":    watchlist.get("macro", []),
            "indices":  watchlist.get("indices", [])[:5],
            "etfs":     [t for t in watchlist.get("etfs", []) if t in ("SPY", "QQQ", "TLT", "HYG")],
        }

        market_data   = fetch_all_market_data(macro_watchlist)
        if not market_data:
            logger.warning("No macro data returned.")
            return

        flat_data     = _build_enriched_flat(market_data)
        enriched_data = enrich_all_assets(flat_data)

        # Fetch research for macro context
        research_config = _build_research_config(macro_watchlist, config)
        research_raw    = fetch_all_research(research_config)
        research        = _build_research_for_report(research_raw)

        treasury_data = fetch_treasury_data()
        fred_data     = research_raw.get("fred") or {}
        macro_result  = analyze_macro(
            market_data   = {"regime_assets": {}},
            treasury_data = treasury_data,
            fred_data     = fred_data,
            last_state    = last_state or {},
        )
        macro_state = _build_macro_state_flat(macro_result)

        # Update all reports with macro context
        report_market_data = _build_market_data_for_report(market_data, enriched_data)
        enriched_by_cat    = _flatten_enriched_for_reports(enriched_data)

        generate_all_reports(
            market_data   = report_market_data,
            enriched_data = enriched_by_cat,
            opportunities = [],
            macro_state   = macro_state,
            research_data = research,
            alerts        = [],
            config        = config,
        )

        logger.info("Macro regime: %s", macro_state.get("regime", "UNKNOWN"))

    except Exception as exc:
        logger.exception("Macro-only run failed: %s", exc)
    finally:
        logger.info("MACRO UPDATE DONE — %.1fs", time.monotonic() - t_start)


def run_scan_only(config: dict) -> None:
    """
    Load last state prices, run scanner, update opportunities.md only.
    Reuses cached prices from last_state rather than fetching fresh market data.
    """
    t_start = time.monotonic()
    logger.info("── SCAN ONLY at %s", datetime.now().strftime("%H:%M:%S"))

    try:
        watchlist  = load_watchlist()
        last_state = load_state()

        if last_state is None:
            logger.warning("No previous state — running quick cycle to build initial state.")
            run_quick_cycle(config)
            return

        # Re-use cached enriched asset data from last saved state
        cached_market = last_state.get("market_data", {})
        if not cached_market:
            logger.warning("Cached state has no market_data — fetching fresh data.")
            market_data   = fetch_all_market_data(watchlist)
            flat_data     = _build_enriched_flat(market_data)
            enriched_data = enrich_all_assets(flat_data)
        else:
            # Build a fake enriched dict from the cached flat snapshot
            enriched_data = {}
            for a in cached_market.get("global_snapshot", []):
                t = a.get("ticker", "")
                if t:
                    enriched_data[t] = a
            logger.info("Using cached asset records from last state: %d assets.", len(enriched_data))

        research_config = _build_research_config(watchlist, config)
        research_raw    = fetch_all_research(research_config)
        research        = _build_research_for_report(research_raw)
        calendar        = research.get("calendar", [])

        ranked    = run_full_scan(enriched_data, calendar)
        opps_list = _opportunities_to_list(ranked)

        macro_state = last_state.get("macro_state", {})

        # Update opportunities + daily brief only
        report_market_data = cached_market if cached_market else {}
        enriched_by_cat    = _flatten_enriched_for_reports(enriched_data)

        generate_all_reports(
            market_data   = report_market_data,
            enriched_data = enriched_by_cat,
            opportunities = opps_list,
            macro_state   = macro_state,
            research_data = research,
            alerts        = [],
            config        = config,
        )

        logger.info("%d opportunities ranked.", len(opps_list))

    except Exception as exc:
        logger.exception("Scan-only run failed: %s", exc)
    finally:
        logger.info("SCAN DONE — %.1fs", time.monotonic() - t_start)


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def setup_schedule(config: dict) -> None:
    """
    Configure and run the recurring schedule.

    Schedule:
    - Full cycle     : every 4 hours
    - Alert check    : every 30 minutes
    - Daily brief    : every day at 06:30 local time (full cycle)
    - Macro dashboard: every 2 hours

    Blocks forever until KeyboardInterrupt.
    """
    logger.info("Setting up schedule...")

    schedule.every(4).hours.do(run_full_cycle, config=config)
    logger.info("  Scheduled: full cycle every 4 hours")

    schedule.every(30).minutes.do(run_alert_check, config=config)
    logger.info("  Scheduled: alert check every 30 minutes")

    schedule.every().day.at("06:30").do(run_full_cycle, config=config)
    logger.info("  Scheduled: daily brief (full cycle) at 06:30")

    schedule.every(2).hours.do(run_macro_only, config=config)
    logger.info("  Scheduled: macro dashboard every 2 hours")

    # Run an immediate full cycle on startup
    logger.info("Running initial full cycle on scheduler startup...")
    run_full_cycle(config)

    logger.info("Scheduler running. Press Ctrl+C to stop.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")


# ---------------------------------------------------------------------------
# Daily brief helper
# ---------------------------------------------------------------------------

def _run_daily_brief(config: dict) -> None:
    """Fetch fresh data and push a full daily digest to console + Telegram."""
    try:
        watchlist       = load_watchlist()
        market_data     = fetch_all_market_data(watchlist)
        flat_data       = _build_enriched_flat(market_data)
        enriched_data   = enrich_all_assets(flat_data)
        research_config = _build_research_config(watchlist, config)
        research_raw    = fetch_all_research(research_config)
        research        = _build_research_for_report(research_raw)
        calendar        = research.get("calendar", [])

        ranked    = run_full_scan(enriched_data, calendar)
        opps_list = _opportunities_to_list(ranked)

        treasury_data = fetch_treasury_data()
        fred_data     = research_raw.get("fred") or {}
        macro_result  = analyze_macro(
            market_data   = {"regime_assets": {}},
            treasury_data = treasury_data,
            fred_data     = fred_data,
            last_state    = load_state() or {},
        )
        macro_state = _build_macro_state_flat(macro_result)

        notify_daily_summary(
            market_data   = {t: {"price": a.get("price"), "pct_change": a.get("change_24h_pct")}
                             for t, a in enriched_data.items()},
            opportunities = opps_list,
            macro_state   = macro_state,
            calendar      = calendar,
            config        = config,
        )
    except Exception as exc:
        logger.exception("Daily brief failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def print_welcome_screen() -> None:
    logo = """
\033[95m\033[1m██╗     ██╗   ██╗███╗   ██╗ █████╗ 
██║     ██║   ██║████╗  ██║██╔══██╗
██║     ██║   ██║██╔██╗ ██║███████║
██║     ██║   ██║██║╚██╗██║██╔══██║
███████╗╚██████╔╝██║ ╚████║██║  ██║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝\033[0m

\033[1m LUNA v1.0.0\033[0m
 A premium, spec-driven autonomous trading research & analysis system.

 \033[92m✓\033[0m Installed command/luna
 \033[92m✓\033[0m Installed outcome scoring tracker
 \033[92m✓\033[0m Installed cross-asset correlation engine

 \033[1mDone! Run one of the following commands to get started:\033[0m
  \033[36mluna --run\033[0m            Run the full intermarket research & analysis cycle
  \033[36mluna --ask "query"\033[0m    Ask LUNA a natural language trading question
  \033[36mluna --portfolio\033[0m      Update and show the investment portfolio status report
  \033[36mluna --performance\033[0m    Generate the historical accuracy performance report
  \033[36mluna --macro\033[0m          Update the macro dashboard and yield spreads
  \033[36mluna --check-outcomes\033[0m Check active call outcomes and auto-tune weights
  \033[36mluna --quick\033[0m          Execute a fast price and watchlist alert scan
"""
    print(logo)

def main() -> None:
    """
    Parse CLI arguments and dispatch to the appropriate run function.

    Modes
    -----
    --run           Single full research + analysis cycle.
    --quick         Price + indicator + alert cycle (no news fetch).
    --macro         Macro regime analysis and dashboard update only.
    --scan          Opportunity scan using cached prices.
    --alert-check   Fetch current prices and check for triggered alerts.
    --schedule      Continuous mode: run all tasks on defined schedule.
    --daily-brief   Send the daily digest immediately (for testing).
    """
    parser = argparse.ArgumentParser(
        description="Autonomous Trading Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python agent.py --run            # Full analysis cycle
              python agent.py --quick          # Fast price + alert pass
              python agent.py --macro          # Update macro dashboard
              python agent.py --scan           # Re-scan opportunities
              python agent.py --alert-check    # Check for triggered alerts
              python agent.py --schedule       # Start continuous scheduler
              python agent.py --daily-brief    # Push today's daily brief now
        """),
    )

    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument("--run",         action="store_true", help="Full research and analysis cycle.")
    mode_group.add_argument("--quick",       action="store_true", help="Quick price + alert cycle (no news).")
    mode_group.add_argument("--macro",       action="store_true", help="Macro analysis and macro dashboard update only.")
    mode_group.add_argument("--scan",        action="store_true", help="Opportunity scan using cached prices.")
    mode_group.add_argument("--alert-check", action="store_true", dest="alert_check", help="Fetch prices and check for triggered alerts.")
    mode_group.add_argument("--schedule",    action="store_true", help="Run in continuous scheduled mode (blocks until Ctrl+C).")
    mode_group.add_argument("--daily-brief", action="store_true", dest="daily_brief", help="Send a full daily digest to console and Telegram.")
    mode_group.add_argument("--check-outcomes", action="store_true", dest="check_outcomes", help="Check pending call outcomes and update weights.")
    mode_group.add_argument("--performance", action="store_true", help="Regenerate agent-performance.md only.")
    mode_group.add_argument("--portfolio",   action="store_true", help="Generate the investment portfolio status report.")
    mode_group.add_argument("--ask", type=str, metavar="QUESTION", help="Ask a natural language question against all loaded data.")
    mode_group.add_argument("--dashboard",   action="store_true", help="Launch the interactive LUNA web dashboard server.")

    args   = parser.parse_args()
    config = load_config()

    if not any([args.run, args.quick, args.macro, args.scan, args.alert_check, args.schedule, args.daily_brief, args.check_outcomes, args.performance, args.portfolio, args.ask, args.dashboard]):
        print_welcome_screen()
        return

    if args.run:
        logger.info("Mode: FULL CYCLE")
        run_full_cycle(config)

    elif args.quick:
        logger.info("Mode: QUICK CYCLE")
        run_quick_cycle(config)

    elif args.macro:
        logger.info("Mode: MACRO ONLY")
        run_macro_only(config)

    elif args.scan:
        logger.info("Mode: SCAN ONLY")
        run_scan_only(config)

    elif args.alert_check:
        logger.info("Mode: ALERT CHECK")
        run_alert_check(config)

    elif args.schedule:
        logger.info("Mode: SCHEDULED (continuous)")
        setup_schedule(config)

    elif args.daily_brief:
        logger.info("Mode: DAILY BRIEF")
        _run_daily_brief(config)

    elif args.check_outcomes:
        logger.info("Mode: CHECK OUTCOMES")
        from calls_tracker import check_pending_outcomes
        check_pending_outcomes()

    elif args.performance:
        logger.info("Mode: GENERATE PERFORMANCE REPORT")
        from reporter import generate_performance_report
        generate_performance_report()

    elif args.portfolio:
        logger.info("Mode: PORTFOLIO STATUS")
        from portfolio import calculate_portfolio_status
        calculate_portfolio_status()

    elif args.ask:
        logger.info("Mode: NATURAL LANGUAGE ASK")
        from query import ask_question
        ask_question(args.ask)

    elif args.dashboard:
        logger.info("Mode: INTERACTIVE WEB DASHBOARD")
        from dashboard import start_dashboard_server
        start_dashboard_server()


if __name__ == "__main__":
    main()
