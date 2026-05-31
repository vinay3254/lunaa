"""
memory.py
=========
Handles long-term asset memory (Module 8) for the autonomous trading research agent.
Tracks rolling 30-day context per asset to distinguish short-term moves from long-term trends.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd

logger = logging.getLogger("memory")

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "state", "asset-memory.json")

def load_memory() -> dict:
    """Load asset memory from state/asset-memory.json. Returns empty dict if file not found/invalid."""
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    if not os.path.exists(MEMORY_PATH):
        logger.debug("Memory file %s not found. Initialising new memory.", MEMORY_PATH)
        return {}
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except Exception as exc:
        logger.error("Failed to load asset memory: %s", exc)
        return {}

def save_memory(memory: dict) -> None:
    """Save asset memory to state/asset-memory.json."""
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
        logger.debug("Asset memory saved to %s", MEMORY_PATH)
    except Exception as exc:
        logger.error("Failed to save asset memory: %s", exc)

def _calculate_trend(prices: pd.Series, period: int) -> str:
    """
    Determine the trend (uptrend, downtrend, ranging) over a given period
    using multi-bar segment means to filter out short-term noise.
    """
    if prices is None or len(prices) < 3:
        return "ranging"
    
    sub_series = prices.tail(period)
    n = len(sub_series)
    if n < 3:
        return "ranging"
    
    # Split into 3 equal segments
    seg_size = n // 3
    if seg_size < 1:
        return "ranging"
    
    seg1 = sub_series.iloc[:seg_size]
    seg2 = sub_series.iloc[seg_size:2*seg_size]
    seg3 = sub_series.iloc[2*seg_size:]
    
    m1 = seg1.mean()
    m2 = seg2.mean()
    m3 = seg3.mean()
    
    if pd.isna(m1) or pd.isna(m2) or pd.isna(m3):
        return "ranging"
    
    if m3 > m2 > m1:
        return "uptrend"
    elif m3 < m2 < m1:
        return "downtrend"
    return "ranging"

def update_asset_memory(enriched_data: dict, current_date_str: str | None = None) -> dict:
    """
    Iterate over all enriched assets and update their rolling 30-day memory.
    
    Parameters
    ----------
    enriched_data : dict
        Enriched asset dictionary from indicators.enrich_all_assets()
    current_date_str : str, optional
        ISO/date string for today. Defaults to today's date in YYYY-MM-DD.
        
    Returns
    -------
    dict: The full updated memory structure.
    """
    if not current_date_str:
        current_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
    memory = load_memory()
    
    for symbol, data in enriched_data.items():
        if "error" in data and data["error"]:
            continue
            
        ohlcv = data.get("ohlcv")
        if ohlcv is None or not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
            continue
            
        close_prices = ohlcv["close"].dropna()
        if close_prices.empty:
            continue
            
        current_close = float(close_prices.iloc[-1])
        prev_close = float(close_prices.iloc[-2]) if len(close_prices) >= 2 else current_close
        
        # Initialise empty state if not tracked yet
        if symbol not in memory:
            memory[symbol] = {
                "trend_7d": "ranging",
                "trend_30d": "ranging",
                "avg_daily_move_30d": 0.0,
                "rsi_history_7d": [],
                "significant_levels_hit": [],
                "sentiment_history_7d": [],
                "sentiment_trend_7d": "stable",
                "consecutive_red_days": 0,
                "consecutive_green_days": 0,
                "last_golden_cross": None,
                "last_death_cross": None,
                "notes": []
            }
            
        m = memory[symbol]
        
        # 1. Recalculate 7d and 30d trends
        m["trend_7d"] = _calculate_trend(close_prices, 9) # 9 bars divided into three 3-bar segments
        m["trend_30d"] = _calculate_trend(close_prices, 30) # 30 bars divided into three 10-day segments
        
        # 2. Average daily move (30d)
        pct_changes = close_prices.tail(30).pct_change().dropna().abs() * 100
        m["avg_daily_move_30d"] = round(float(pct_changes.mean()), 4) if not pct_changes.empty else 0.0
        
        # 3. RSI History (last 7 values)
        current_rsi = data.get("rsi")
        if current_rsi is not None:
            m["rsi_history_7d"].append(round(float(current_rsi), 2))
            m["rsi_history_7d"] = m["rsi_history_7d"][-7:]
            
        # 4. Sentiment Trend (last 7 values)
        current_sentiment = data.get("sentiment_score", 0.0) # standard keyword sentiment or weighted sentiment
        m["sentiment_history_7d"].append(round(float(current_sentiment), 2))
        m["sentiment_history_7d"] = m["sentiment_history_7d"][-7:]
        
        # Calculate sentiment trend: average of last 2 days vs previous 5 days
        if len(m["sentiment_history_7d"]) >= 3:
            s_hist = m["sentiment_history_7d"]
            recent_avg = np.mean(s_hist[-2:])
            prior_avg = np.mean(s_hist[:-2])
            diff = recent_avg - prior_avg
            if diff > 0.1:
                m["sentiment_trend_7d"] = "improving"
            elif diff < -0.1:
                m["sentiment_trend_7d"] = "deteriorating"
            else:
                m["sentiment_trend_7d"] = "stable"
                
        # 5. Consecutive red/green days
        if current_close > prev_close:
            m["consecutive_green_days"] += 1
            m["consecutive_red_days"] = 0
        elif current_close < prev_close:
            m["consecutive_red_days"] += 1
            m["consecutive_green_days"] = 0
        else:
            # no change
            pass
            
        # 6. Checks for Golden / Death cross (50 EMA vs 200 EMA)
        ema_50 = data.get("ema_50")
        ema_200 = data.get("ema_200")
        if ema_50 is not None and ema_200 is not None:
            # Let's inspect the EMA series if available to detect cross direction
            ohlcv_ema_50 = ohlcv["close"].ewm(span=50, adjust=False).mean()
            ohlcv_ema_200 = ohlcv["close"].ewm(span=200, adjust=False).mean()
            if len(ohlcv_ema_50) >= 2:
                curr_ema_50, curr_ema_200 = ohlcv_ema_50.iloc[-1], ohlcv_ema_200.iloc[-1]
                prev_ema_50, prev_ema_200 = ohlcv_ema_50.iloc[-2], ohlcv_ema_200.iloc[-2]
                
                # Check for Golden Cross
                if prev_ema_50 <= prev_ema_200 and curr_ema_50 > curr_ema_200:
                    m["last_golden_cross"] = current_date_str
                    note_msg = f"{current_date_str}: Golden Cross detected (50 EMA crossed above 200 EMA)"
                    m["notes"].append(note_msg)
                    logger.info("[%s] Golden Cross detected!", symbol)
                # Check for Death Cross
                elif prev_ema_50 >= prev_ema_200 and curr_ema_50 < curr_ema_200:
                    m["last_death_cross"] = current_date_str
                    note_msg = f"{current_date_str}: Death Cross detected (50 EMA crossed below 200 EMA)"
                    m["notes"].append(note_msg)
                    logger.info("[%s] Death Cross detected!", symbol)
                    
        # 7. Level breaches (Support/Resistance / 52w High-Low)
        levels = data.get("support_resistance", {})
        supports = levels.get("support", [])
        resistances = levels.get("resistance", [])
        
        # Check support breaches
        for level in supports:
            level = float(level)
            if prev_close >= level and current_close < level:
                sig_hit = {"type": "support_break", "level": level, "date": current_date_str}
                m["significant_levels_hit"].append(sig_hit)
                m["notes"].append(f"{current_date_str}: broke key support at ${level:,.2f}")
                logger.info("[%s] Broke key support level at $%s", symbol, level)
                break # log one breach per run to avoid spam
                
        # Check resistance breaches/rejections
        for level in resistances:
            level = float(level)
            # Breakout: close above resistance
            if prev_close <= level and current_close > level:
                sig_hit = {"type": "resistance_break", "level": level, "date": current_date_str}
                m["significant_levels_hit"].append(sig_hit)
                m["notes"].append(f"{current_date_str}: broke above key resistance at ${level:,.2f}")
                logger.info("[%s] Broke above resistance level at $%s", symbol, level)
                break
            # Rejection: intraday high went above level but closed back below it
            elif "high" in ohlcv.columns:
                curr_high = float(ohlcv["high"].iloc[-1])
                if prev_close < level and curr_high >= level and current_close < level:
                    sig_hit = {"type": "resistance_reject", "level": level, "date": current_date_str}
                    m["significant_levels_hit"].append(sig_hit)
                    m["notes"].append(f"{current_date_str}: rejected resistance level at ${level:,.2f}")
                    logger.info("[%s] Rejected resistance level at $%s", symbol, level)
                    break
                    
        # Check 52w high/low breaches
        high_52w = data.get("high_52w")
        low_52w = data.get("low_52w")
        if high_52w is not None and current_close >= float(high_52w):
            sig_hit = {"type": "new_52w_high", "level": float(high_52w), "date": current_date_str}
            m["significant_levels_hit"].append(sig_hit)
            m["notes"].append(f"{current_date_str}: achieved a new 52-week high at ${current_close:,.2f}")
            logger.info("[%s] Achieved new 52w high!", symbol)
        elif low_52w is not None and current_close <= float(low_52w):
            sig_hit = {"type": "new_52w_low", "level": float(low_52w), "date": current_date_str}
            m["significant_levels_hit"].append(sig_hit)
            m["notes"].append(f"{current_date_str}: hit a new 52-week low at ${current_close:,.2f}")
            logger.info("[%s] Hit new 52w low!", symbol)
            
        # Clean significant levels to last 10
        m["significant_levels_hit"] = m["significant_levels_hit"][-10:]
        
        # Trim notes to last 10 entries per asset
        m["notes"] = m["notes"][-10:]
        
    save_memory(memory)
    return memory

def get_asset_context_summary(symbol: str, direction: str, current_price: float | None = None) -> str:
    """
    Generate a 1-sentence longitudinal context summary from asset memory
    to enrich opportunities reports.
    """
    memory = load_memory()
    if symbol not in memory:
        return ""
        
    m = memory[symbol]
    parts = []
    
    # 1. Consecutive green/red days
    c_red = m.get("consecutive_red_days", 0)
    c_green = m.get("consecutive_green_days", 0)
    if c_red > 0:
        parts.append(f"{c_red} consecutive red days")
    elif c_green > 0:
        parts.append(f"{c_green} consecutive green days")
        
    # 2. Last significant level hit
    sig_hits = m.get("significant_levels_hit", [])
    if sig_hits:
        last_hit = sig_hits[-1]
        t_type = last_hit.get("type", "").replace("_", " ")
        lvl = last_hit.get("level", 0.0)
        dt = last_hit.get("date", "")
        try:
            date_obj = datetime.strptime(dt, "%Y-%m-%d")
            date_str = date_obj.strftime("%b %d")
        except Exception:
            date_str = dt
        parts.append(f"broke key {t_type} at ${lvl:,.2f} on {date_str}")
        
    # 3. 30d trend
    trend_30d = m.get("trend_30d", "ranging")
    parts.append(f"currently in 30-day {trend_30d}")
    
    # 4. Verdict
    verdict = ""
    if direction == "bullish":
        if trend_30d == "downtrend":
            verdict = "oversold bounce possible but trend is against longs"
        elif trend_30d == "uptrend":
            verdict = "strong trend alignment for longs"
        else:
            verdict = "consolidation play"
    elif direction == "bearish":
        if trend_30d == "uptrend":
            verdict = "overbought pullback possible but trend is against shorts"
        elif trend_30d == "downtrend":
            verdict = "strong trend alignment for shorts"
        else:
            verdict = "consolidation play"
            
    summary = "context: " + ", ".join(parts)
    if verdict:
        summary += f". {verdict}."
    else:
        summary += "."
        
    return summary
