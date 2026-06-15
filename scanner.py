"""
scanner.py — Autonomous Trading Research Agent: Opportunity Scanner & Scorer
=============================================================================
Scans all enriched assets, scores them on a -10 to +10 scale, and detects
pattern-based opportunity categories (breakouts, squeezes, crossovers, etc.).

PHASE 2+ UPGRADE: Integrated ML predictions and confidence scoring.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

try:
    import ml_engine
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import sentiment_engine
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCORE_MIN = -10
SCORE_MAX = 10

BULLISH_THRESHOLD = 6
BEARISH_THRESHOLD = -6

RSI_VERY_OVERSOLD = 30
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
RSI_VERY_OVERBOUGHT = 70

VOLUME_SPIKE_RATIO = 1.5       # volume > 1.5x avg  → bullish
VOLUME_DRY_RATIO = 0.7         # volume < 0.70x avg → drying up / bearish

BB_LOWER_ZONE = 0.1            # price within 10 % of band width from lower band
BB_UPPER_ZONE = 0.1            # price within 10 % of band width from upper band

BREAKOUT_TOLERANCE = 0.015     # within 1.5 % of resistance
BREAKDOWN_TOLERANCE = 0.015    # within 1.5 % of support
MEAN_REVERSION_THRESHOLD = 0.30  # 30 % from 200 EMA triggers mean-reversion flag
OVERSOLD_RSI_THRESHOLD = 30
MOMENTUM_LOOKBACK = 252        # 52-week high lookback (trading days)

CORRELATION_LONG_WINDOW = 90   # days for long-term correlation baseline
CORRELATION_SHORT_WINDOW = 10  # days for recent correlation
CORRELATION_DIVERGENCE = 0.30  # correlation drop > 0.30 = break

MACD_CROSSOVER_LOOKBACK = 3    # bars to look back for crossover
EMA_CROSS_LOOKBACK = 5         # bars to look back for golden/death cross

BB_SQUEEZE_LOOKBACK = 20       # periods for BB-width minimum


# ===========================================================================
# INDIVIDUAL SIGNAL SCORERS
# ===========================================================================

def score_rsi(rsi: float) -> int:
    """
    Score RSI on a -2 to +2 scale.

    Args:
        rsi: Current RSI value (0-100). Non-finite or out-of-range → 0.

    Returns:
        +2 : RSI < 30  (very oversold — strong buy signal)
        +1 : RSI 30-35 (oversold)
         0 : RSI 35-65 (neutral)
        -1 : RSI 65-70 (overbought)
        -2 : RSI > 70  (very overbought — strong sell signal)
    """
    try:
        rsi = float(rsi)
    except (TypeError, ValueError):
        logger.debug("score_rsi: cannot convert value '%s' to float → 0", rsi)
        return 0

    if not np.isfinite(rsi):
        logger.debug("score_rsi: non-finite RSI value → 0")
        return 0

    if rsi < RSI_VERY_OVERSOLD:
        return 2
    elif rsi < RSI_OVERSOLD:
        return 1
    elif rsi <= RSI_OVERBOUGHT:
        return 0
    elif rsi <= RSI_VERY_OVERBOUGHT:
        return -1
    else:
        return -2


def score_macd(macd_data: dict) -> int:
    """
    Score MACD signal on a -2 to +2 scale.

    Expected keys in macd_data:
        macd        : list[float] | float — MACD line values (latest last)
        signal      : list[float] | float — signal line values (latest last)
        crossover   : str | None          — 'bullish', 'bearish', or None
        bars_since_crossover : int | None — how many bars ago the crossover happened

    Logic:
        +2 : Bullish crossover within last 3 bars
        +1 : MACD currently above signal (bullish momentum, no recent cross)
        -1 : MACD currently below signal (bearish momentum)
        -2 : Bearish crossover within last 3 bars
         0 : data missing / cannot determine

    Args:
        macd_data: Dictionary with MACD indicator data.

    Returns:
        Integer score in [-2, +2].
    """
    if not macd_data or not isinstance(macd_data, dict):
        logger.debug("score_macd: missing or invalid macd_data → 0")
        return 0

    # ---- Extract crossover information ----
    crossover = macd_data.get("crossover")
    bars_since = macd_data.get("bars_since_crossover")

    # If bars_since is available and within lookback, use it directly
    if crossover is not None and bars_since is not None:
        try:
            bars_since = int(bars_since)
        except (TypeError, ValueError):
            bars_since = None

        if bars_since is not None and bars_since <= MACD_CROSSOVER_LOOKBACK:
            if crossover == "bullish":
                return 2
            elif crossover == "bearish":
                return -2

    # ---- Detect crossover from raw arrays ----
    macd_vals = macd_data.get("macd")
    signal_vals = macd_data.get("signal")

    def _to_list(v):
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v if x is not None]
        if isinstance(v, (int, float)) and np.isfinite(float(v)):
            return [float(v)]
        return []

    macd_list = _to_list(macd_vals)
    signal_list = _to_list(signal_vals)

    if not macd_list or not signal_list:
        logger.debug("score_macd: could not extract MACD/signal values → 0")
        return 0

    # Align lengths
    min_len = min(len(macd_list), len(signal_list))
    macd_arr = np.array(macd_list[-min_len:])
    signal_arr = np.array(signal_list[-min_len:])

    # Look for crossover in last MACD_CROSSOVER_LOOKBACK bars
    lookback = min(MACD_CROSSOVER_LOOKBACK + 1, min_len)
    macd_slice = macd_arr[-lookback:]
    signal_slice = signal_arr[-lookback:]

    diff = macd_slice - signal_slice  # positive = MACD above signal

    # Crossover: sign change between consecutive bars
    if len(diff) >= 2:
        for i in range(len(diff) - 1, 0, -1):
            bars_ago = (len(diff) - 1 - i)
            if bars_ago > MACD_CROSSOVER_LOOKBACK:
                break
            prev_diff = diff[i - 1]
            curr_diff = diff[i]
            if prev_diff <= 0 < curr_diff:  # bullish crossover
                return 2
            if prev_diff >= 0 > curr_diff:  # bearish crossover
                return -2

    # No recent crossover — check current position
    current_macd = macd_arr[-1]
    current_signal = signal_arr[-1]

    if current_macd > current_signal:
        return 1
    elif current_macd < current_signal:
        return -1
    return 0


def score_ema_stack(price: float, ema20: float, ema50: float, ema200: float) -> int:
    """
    Score EMA stack alignment on a -2 to +2 scale.

    Full bullish  (price > ema20 > ema50 > ema200) → +2
    Partial bullish (price > ema50 > ema200 but not full) → +1
    Neutral / mixed → 0
    Partial bearish (price < ema50 < ema200 but not full) → -1
    Full bearish  (price < ema20 < ema50 < ema200) → -2

    Args:
        price : Current asset price.
        ema20 : 20-period EMA.
        ema50 : 50-period EMA.
        ema200: 200-period EMA.

    Returns:
        Integer score in [-2, +2].
    """
    def _valid(v) -> bool:
        try:
            return np.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    if not all(_valid(v) for v in (price, ema20, ema50, ema200)):
        logger.debug("score_ema_stack: one or more values invalid → 0")
        return 0

    price = float(price)
    ema20 = float(ema20)
    ema50 = float(ema50)
    ema200 = float(ema200)

    full_bullish = price > ema20 > ema50 > ema200
    full_bearish = price < ema20 < ema50 < ema200

    if full_bullish:
        return 2
    if full_bearish:
        return -2

    # Partial bullish: price and ema50 above ema200
    partial_bullish = (price > ema50) and (ema50 > ema200)
    # Partial bearish: price and ema50 below ema200
    partial_bearish = (price < ema50) and (ema50 < ema200)

    if partial_bullish:
        return 1
    if partial_bearish:
        return -1

    return 0


def score_volume(volume_data: dict) -> int:
    """
    Score volume relative to its 20-day average on a -1 to +1 scale.

    Expected keys in volume_data:
        current_volume   : float — today's volume
        avg_volume_20d   : float — 20-day average volume

    Logic:
        +1 : current_volume > 1.5 × avg  (spike — conviction)
        -1 : current_volume < 0.70 × avg (drying up — lack of conviction)
         0 : normal or data missing

    Args:
        volume_data: Dictionary with volume information.

    Returns:
        Integer score in [-1, +1].
    """
    if not volume_data or not isinstance(volume_data, dict):
        logger.debug("score_volume: missing or invalid volume_data → 0")
        return 0

    try:
        current_vol = float(volume_data.get("current_volume", 0))
        avg_vol = float(volume_data.get("avg_volume_20d", 0))
    except (TypeError, ValueError):
        logger.debug("score_volume: cannot parse volume values → 0")
        return 0

    if avg_vol <= 0 or not np.isfinite(current_vol) or not np.isfinite(avg_vol):
        logger.debug("score_volume: zero or non-finite average volume → 0")
        return 0

    ratio = current_vol / avg_vol

    if ratio >= VOLUME_SPIKE_RATIO:
        return 1
    elif ratio <= VOLUME_DRY_RATIO:
        return -1
    return 0


def score_bb_position(price: float, bb_data: dict) -> int:
    """
    Score Bollinger Band position on a -1 to +1 scale.

    Expected keys in bb_data:
        upper  : float — upper Bollinger Band
        middle : float — middle band (20-period SMA)
        lower  : float — lower Bollinger Band

    Logic:
        +1 : price near (within 10 % of band-width from) the lower band (bounce)
        -1 : price near the upper band (rejection)
         0 : price in the middle or data missing

    Args:
        price  : Current asset price.
        bb_data: Dictionary with Bollinger Band levels.

    Returns:
        Integer score in [-1, +1].
    """
    if not bb_data or not isinstance(bb_data, dict):
        logger.debug("score_bb_position: missing or invalid bb_data → 0")
        return 0

    try:
        upper = float(bb_data.get("upper", float("nan")))
        lower = float(bb_data.get("lower", float("nan")))
        price = float(price)
    except (TypeError, ValueError):
        logger.debug("score_bb_position: cannot parse BB/price values → 0")
        return 0

    if not all(np.isfinite(v) for v in (upper, lower, price)):
        logger.debug("score_bb_position: non-finite values → 0")
        return 0

    band_width = upper - lower
    if band_width <= 0:
        logger.debug("score_bb_position: band width is zero or negative → 0")
        return 0

    # Distance from lower band as fraction of total band width
    dist_from_lower = (price - lower) / band_width  # 0 = at lower, 1 = at upper

    if dist_from_lower <= BB_LOWER_ZONE:
        return 1   # near lower band → potential bounce
    elif dist_from_lower >= (1.0 - BB_UPPER_ZONE):
        return -1  # near upper band → potential rejection
    return 0


def score_sentiment(sentiment: float) -> int:
    """
    Score news/social sentiment on a -1 to +1 scale.

    The sentiment value is expected as a normalized score where:
        > 0 → net positive
        < 0 → net negative
        = 0 → neutral

    Typical range: -1.0 to +1.0 (e.g. from an NLP pipeline), but any
    signed float is accepted. The function only discriminates by sign.

    Args:
        sentiment: Net sentiment score (positive = bullish, negative = bearish).

    Returns:
        +1 : Net positive sentiment
        -1 : Net negative sentiment
         0 : Neutral or missing/invalid
    """
    try:
        sentiment = float(sentiment)
    except (TypeError, ValueError):
        logger.debug("score_sentiment: cannot convert value to float → 0")
        return 0

    if not np.isfinite(sentiment):
        logger.debug("score_sentiment: non-finite sentiment value → 0")
        return 0

    if sentiment > 0:
        return 1
    elif sentiment < 0:
        return -1
    return 0


def score_catalyst(asset_name: str, calendar: list) -> int:
    """
    Score upcoming catalyst events on a -1 to +1 scale.

    Looks for any calendar event in the next 3 days whose description or
    ticker field mentions the asset (case-insensitive substring match).
    High-impact events with a clear directional bias score ±1.

    Expected calendar entry structure (all keys optional but used when present):
        {
            "date"     : str | datetime  — event date (ISO 8601 or datetime)
            "ticker"   : str             — asset ticker
            "name"     : str             — asset or event name
            "title"    : str             — event title / description
            "impact"   : str             — 'High', 'Medium', 'Low'
            "direction": str             — 'bullish', 'bearish', or None
        }

    Args:
        asset_name : Asset ticker or name to look up (case-insensitive).
        calendar   : List of upcoming event dicts.

    Returns:
        +1 : High-impact bullish catalyst within 3 days
        -1 : High-impact bearish catalyst within 3 days
         0 : No relevant high-impact event found or data missing
    """
    if not asset_name or not calendar:
        return 0

    now = datetime.now(tz=timezone.utc)
    cutoff = now + timedelta(days=3)
    asset_lower = str(asset_name).lower()

    for event in calendar:
        if not isinstance(event, dict):
            continue

        # ---- Date check ----
        raw_date = event.get("date")
        if raw_date is None:
            continue

        try:
            if isinstance(raw_date, datetime):
                event_dt = raw_date
                if event_dt.tzinfo is None:
                    event_dt = event_dt.replace(tzinfo=timezone.utc)
            else:
                event_dt = datetime.fromisoformat(str(raw_date))
                if event_dt.tzinfo is None:
                    event_dt = event_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            logger.debug("score_catalyst: cannot parse date '%s' → skip event", raw_date)
            continue

        if not (now <= event_dt <= cutoff):
            continue

        # ---- Asset name / ticker match ----
        ticker_field = str(event.get("ticker", "")).lower()
        name_field = str(event.get("name", "")).lower()
        title_field = str(event.get("title", "")).lower()

        matched = (
            asset_lower in ticker_field
            or asset_lower in name_field
            or asset_lower in title_field
            or ticker_field in asset_lower   # e.g. event ticker 'AAPL' in 'AAPL'
        )

        if not matched:
            continue

        # ---- Impact and direction check ----
        impact = str(event.get("impact", "")).strip().lower()
        if impact != "high":
            continue

        direction = str(event.get("direction", "")).strip().lower()
        if direction == "bullish":
            return 1
        elif direction == "bearish":
            return -1
        # High-impact event present but direction not clear → treat as neutral
        logger.debug(
            "score_catalyst: high-impact event for '%s' has no clear direction", asset_name
        )

    return 0


# ===========================================================================
# COMPOSITE ASSET SCORER
# ===========================================================================

def _clamp(value: int, lo: int = SCORE_MIN, hi: int = SCORE_MAX) -> int:
    """Clamp integer value to [lo, hi]."""
    return max(lo, min(hi, value))


def _format_breakdown(components: dict) -> str:
    """
    Build a human-readable breakdown string, e.g.:
    'RSI:+2, MACD:+1, EMA:+2, Vol:+1, BB:0, Sent:+1, Cat:0 = +7'
    """
    parts = []
    for label, score in components.items():
        sign = "+" if score >= 0 else ""
        parts.append(f"{label}:{sign}{score}")
    total = sum(components.values())
    sign = "+" if total >= 0 else ""
    return ", ".join(parts) + f" = {sign}{total}"


def _determine_direction(score: int) -> str:
    """Map aggregate score to direction label."""
    if score >= BULLISH_THRESHOLD:
        return "bullish"
    elif score <= BEARISH_THRESHOLD:
        return "bearish"
    return "neutral"


def _collect_signals(components: dict, score: int) -> list[str]:
    """
    Convert component scores to human-readable signal strings for the result.
    Only non-zero components are included.
    """
    signal_map = {
        "RSI": {
            2: "RSI very oversold (<30) — strong buy",
            1: "RSI oversold (30-35)",
            -1: "RSI overbought (65-70)",
            -2: "RSI very overbought (>70) — strong sell",
        },
        "MACD": {
            2: "Bullish MACD crossover (last 3 bars)",
            1: "MACD above signal — bullish momentum",
            -1: "MACD below signal — bearish momentum",
            -2: "Bearish MACD crossover (last 3 bars)",
        },
        "EMA": {
            2: "Full bullish EMA stack (price>EMA20>EMA50>EMA200)",
            1: "Partial bullish EMA alignment",
            -1: "Partial bearish EMA alignment",
            -2: "Full bearish EMA stack (price<EMA20<EMA50<EMA200)",
        },
        "Vol": {
            1: "Volume spike (>1.5x 20d avg) — strong conviction",
            -1: "Volume drying up (<0.70x 20d avg)",
        },
        "BB": {
            1: "Price bouncing off lower Bollinger Band",
            -1: "Price rejecting upper Bollinger Band",
        },
        "Sent": {
            1: "Net positive news/social sentiment",
            -1: "Net negative news/social sentiment",
        },
        "Cat": {
            1: "High-impact bullish catalyst within 3 days",
            -1: "High-impact bearish catalyst within 3 days",
        },
    }

    signals = []
    for label, comp_score in components.items():
        if comp_score == 0:
            continue
        desc = signal_map.get(label, {}).get(comp_score, f"{label} score {comp_score:+.1f}")
        signals.append(desc)

    # Overall verdict
    if score >= BULLISH_THRESHOLD:
        signals.append(f"STRONG BUY CANDIDATE (score {score:+.1f})")
    elif score <= BEARISH_THRESHOLD:
        signals.append(f"STRONG SELL/SHORT CANDIDATE (score {score:+.1f})")

    return signals


def generate_tactical_card(asset: dict, score_result: dict, macro_state: dict) -> dict:
    """Generate tactical card details including confidence, supporting evidence, and entry/stop zones."""
    price = asset.get("price", 0.0)
    direction = score_result.get("direction", "neutral")
    ml_pred = score_result.get("ml_prediction", {})
    regime = macro_state.get("regime", "RISK-ON")
    ticker = asset.get("ticker", "UNKNOWN")
    
    # 1. Confidence Score & Stars
    is_fallback = ml_pred.get("fallback", True)
    if not is_fallback:
        prob = ml_pred.get("bullish_probability", 0.0) if direction == "bullish" else ml_pred.get("bearish_probability", 0.0)
        confidence = prob
    else:
        # Heuristic from rule-based score
        confidence = abs(score_result.get("score", 0.0)) / 10.0
        
    if confidence >= 0.8:
        stars = "★★★★★ HIGH CONFIDENCE"
    elif confidence >= 0.7:
        stars = "★★★★☆ HIGH CONFIDENCE"
    elif confidence >= 0.6:
        stars = "★★★☆☆ MEDIUM CONFIDENCE"
    elif confidence >= 0.5:
        stars = "★★☆☆☆ MEDIUM CONFIDENCE"
    else:
        stars = "★☆☆☆☆ LOW CONFIDENCE"
        
    # 2. Supporting Evidence (Factors Aligned)
    evidence = []
    aligned_count = 0
    total_factors = 6
    
    # RSI factor
    rsi = asset.get("rsi")
    if rsi is not None and not np.isnan(rsi):
        if direction == "bullish":
            if rsi < 40:
                evidence.append(f"✅ RSI {rsi:.1f} — oversold bounce setup")
                aligned_count += 1
            elif rsi > 70:
                evidence.append(f"❌ RSI {rsi:.1f} — overbought conditions")
            else:
                evidence.append(f"⚠️ RSI {rsi:.1f} — neutral zone")
        elif direction == "bearish":
            if rsi > 65:
                evidence.append(f"✅ RSI {rsi:.1f} — overbought rejection setup")
                aligned_count += 1
            elif rsi < 30:
                evidence.append(f"❌ RSI {rsi:.1f} — oversold conditions")
            else:
                evidence.append(f"⚠️ RSI {rsi:.1f} — neutral zone")
    else:
        evidence.append("⚠️ RSI data unavailable")
        
    # MACD factor
    macd_data = asset.get("macd", {})
    if macd_data and isinstance(macd_data, dict):
        cross = macd_data.get("crossover")
        if direction == "bullish":
            if cross == "bullish":
                evidence.append("✅ MACD bullish cross confirmed")
                aligned_count += 1
            elif cross == "bearish":
                evidence.append("❌ MACD bearish cross active")
            else:
                evidence.append("⚠️ MACD momentum neutral")
        elif direction == "bearish":
            if cross == "bearish":
                evidence.append("✅ MACD bearish cross confirmed")
                aligned_count += 1
            elif cross == "bullish":
                evidence.append("❌ MACD bullish cross active")
            else:
                evidence.append("⚠️ MACD momentum neutral")
    else:
        evidence.append("⚠️ MACD data unavailable")
        
    # Volume factor
    vol_data = asset.get("volume_anomaly", {})
    vol_ratio = 1.0
    if vol_data:
        vol_avg = vol_data.get("avg_20d", 0.0)
        vol_curr = vol_data.get("current", 0.0)
        if vol_avg > 0:
            vol_ratio = vol_curr / vol_avg
            
    if vol_ratio > 1.5:
        evidence.append(f"✅ Volume {vol_ratio:.1f}x average — institutional conviction")
        aligned_count += 1
    elif vol_ratio < 0.7:
        evidence.append(f"❌ Volume {vol_ratio:.1f}x average — low participation")
    else:
        evidence.append(f"⚠️ Volume {vol_ratio:.1f}x average — standard activity")
        
    # Catalyst factor
    cat_score = score_result.get("breakdown", {}).get("Cat", 0)
    if cat_score > 0 and direction == "bullish":
        evidence.append("✅ Near term bullish catalyst identified")
        aligned_count += 1
    elif cat_score < 0 and direction == "bearish":
        evidence.append("✅ Near term bearish catalyst identified")
        aligned_count += 1
    else:
        evidence.append("⚠️ No immediate calendar catalysts")
        
    # Regime factor
    if direction == "bullish":
        if regime == "RISK-ON":
            evidence.append(f"✅ Regime RISK-ON — strong market tailwinds")
            aligned_count += 1
        elif regime == "TRANSITIONING":
            evidence.append(f"⚠️ Regime TRANSITIONING — reduces conviction")
        else:
            evidence.append(f"❌ Regime RISK-OFF — overall market headwinds")
    elif direction == "bearish":
        if regime == "RISK-OFF":
            evidence.append(f"✅ Regime RISK-OFF — market conditions support shorts")
            aligned_count += 1
        elif regime == "TRANSITIONING":
            evidence.append(f"⚠️ Regime TRANSITIONING — reduces conviction")
        else:
            evidence.append(f"❌ Regime RISK-ON — overall market headwind for shorts")
            
    # EMA Stack / Trend factor
    ema_score = score_result.get("breakdown", {}).get("EMA", 0)
    if direction == "bullish":
        if ema_score > 0:
            evidence.append("✅ Bullish EMA stack confirmed")
            aligned_count += 1
        else:
            evidence.append("❌ EMA 200 above price — long term trend bearish")
    elif direction == "bearish":
        if ema_score < 0:
            evidence.append("✅ Bearish EMA stack confirmed")
            aligned_count += 1
        else:
            evidence.append("❌ EMA 200 below price — long term trend bullish")
            
    # 3. Model Info
    if not is_fallback:
        confidence_str = ml_pred.get("model_confidence", "MEDIUM")
        model_info = f"Model: Ensemble ({ml_pred.get('model_type', 'General')}) | Confidence: {confidence_str}"
        accuracy_info = f"Model accuracy last 30 days: {ml_pred.get('model_accuracy_last_30d', 0.5)*100:.0f}% | Sample size: {ml_pred.get('sample_size', 0)}"
    else:
        model_info = "Model: Rule-Based Fallback active"
        accuracy_info = "Model accuracy last 30 days: N/A"
        
    # 4. Backtest Win Rate
    backtest_info = "Backtest win rate (180d): N/A"
    try:
        import backtest
        perf = backtest.load_backtest_performance()
        if ticker in perf:
            t_perf = perf[ticker]
            if direction in ("bullish", "bearish"):
                win_rate = t_perf.get(direction, {}).get("win_rate", 0.0)
                tot_sigs = t_perf.get(direction, {}).get("total_signals", 0)
                if tot_sigs > 0:
                    backtest_info = f"Backtest win rate ({t_perf.get('days', 90)}d): {win_rate*100:.1f}% ({tot_sigs} signals)"
    except Exception:
        pass
        
    # 5. Entry Zone & Stop Loss Invalidation Zone & Time Horizon
    if price > 0:
        if direction == "bullish":
            entry_zone = f"${0.99*price:.2f} - ${1.01*price:.2f}"
            
            # Stop loss logic (support level or 0.95*price or EMA50)
            ema50 = asset.get("ema50")
            support = asset.get("support")
            
            if ema50 and ema50 < price:
                stop_val = ema50
                stop_reason = "closes below EMA50"
            elif support:
                sup_list = support if isinstance(support, list) else [support]
                valid_sups = [s for s in sup_list if s < price]
                if valid_sups:
                    stop_val = max(valid_sups)
                    stop_reason = f"closes below support ${stop_val:.2f}"
                else:
                    stop_val = 0.95 * price
                    stop_reason = "5% default stop"
            else:
                stop_val = 0.95 * price
                stop_reason = "5% default stop"
                
            stop_invalidation = f"Below ${stop_val:.2f} ({stop_reason})"
        elif direction == "bearish":
            entry_zone = f"${0.99*price:.2f} - ${1.01*price:.2f}"
            
            # Stop loss logic
            ema50 = asset.get("ema50")
            resistance = asset.get("resistance")
            
            if ema50 and ema50 > price:
                stop_val = ema50
                stop_reason = "closes above EMA50"
            elif resistance:
                res_list = resistance if isinstance(resistance, list) else [resistance]
                valid_res = [r for r in res_list if r > price]
                if valid_res:
                    stop_val = min(valid_res)
                    stop_reason = f"closes above resistance ${stop_val:.2f}"
                else:
                    stop_val = 1.05 * price
                    stop_reason = "5% default stop"
            else:
                stop_val = 1.05 * price
                stop_reason = "5% default stop"
                
            stop_invalidation = f"Above ${stop_val:.2f} ({stop_reason})"
        else:
            entry_zone = "N/A"
            stop_invalidation = "N/A"
    else:
        entry_zone = "N/A"
        stop_invalidation = "N/A"
        
    return {
        "confidence_score": round(confidence, 2),
        "confidence_stars": stars,
        "aligned_factors": f"{aligned_count}/{total_factors}",
        "evidence": evidence,
        "model_info": model_info,
        "accuracy_info": accuracy_info,
        "backtest_info": backtest_info,
        "entry_zone": entry_zone,
        "stop_invalidation": stop_invalidation,
        "time_horizon": "3-7 days"
    }


def score_asset(asset: dict, calendar: list, regime: str | None = None, macro_state: dict | None = None) -> dict:
    """
    Compute the composite opportunity score for a single asset.

    Args:
        asset    : Enriched asset dictionary. Expected keys (all optional but
                   scored when present):
                     - ticker        : str
                     - name          : str
                     - price         : float
                     - rsi           : float
                     - macd          : dict  (see score_macd)
                     - ema20         : float
                     - ema50         : float
                     - ema200        : float
                     - volume        : dict  (see score_volume)
                     - bb            : dict  (see score_bb_position)
                     - sentiment     : float
        calendar : List of upcoming event dicts (see score_catalyst).
        regime   : Optional market regime string.
        macro_state : Optional flat macro state dict.

    Returns:
        dict with keys:
            ticker      : str
            name        : str
            score       : float — clamped to [-10.0, +10.0]
            breakdown   : dict  — {'RSI': int, 'MACD': int, ...}
            breakdown_str: str  — formatted breakdown string
            direction   : str   — 'bullish' | 'bearish' | 'neutral'
            signals     : list[str]
    """
    if not asset or not isinstance(asset, dict):
        logger.warning("score_asset: received empty or invalid asset dict")
        return {
            "ticker": "UNKNOWN",
            "name": "UNKNOWN",
            "score": 0,
            "breakdown": {},
            "breakdown_str": "No data",
            "direction": "neutral",
            "signals": [],
        }

    ticker = str(asset.get("ticker", asset.get("symbol", "UNKNOWN")))
    name = str(asset.get("name", ticker))

    # ---- Individual component scores ----
    rsi_score = score_rsi(asset.get("rsi", float("nan")))
    macd_score = score_macd(asset.get("macd", {}))

    price = asset.get("price", float("nan"))
    ema20 = asset.get("ema20", float("nan"))
    ema50 = asset.get("ema50", float("nan"))
    ema200 = asset.get("ema200", float("nan"))
    ema_score = score_ema_stack(price, ema20, ema50, ema200)

    vol_score = score_volume(asset.get("volume", {}))
    bb_score = score_bb_position(price, asset.get("bb", {}))
    sent_score = score_sentiment(asset.get("sentiment", float("nan")))
    cat_score = score_catalyst(ticker, calendar) or score_catalyst(name, calendar)

    raw_components = {
        "RSI": rsi_score,
        "MACD": macd_score,
        "EMA": ema_score,
        "Vol": vol_score,
        "BB": bb_score,
        "Sent": sent_score,
        "Cat": cat_score,
    }

    # ---- Dynamic Weighting (Module 7) ----
    from calls_tracker import load_scoring_weights
    weights = load_scoring_weights(regime)

    weighted_components = {
        "RSI": round(rsi_score * weights.get("rsi", 2.0), 2),
        "MACD": round(macd_score * weights.get("macd", 2.0), 2),
        "EMA": round(ema_score * weights.get("ema_stack", 2.0), 2),
        "Vol": round(vol_score * weights.get("volume", 1.0), 2),
        "BB": round(bb_score * weights.get("bb_position", 1.0), 2),
        "Sent": round(sent_score * weights.get("sentiment", 1.0), 2),
        "Cat": round(cat_score * weights.get("catalyst", 1.0), 2),
    }

    raw_score = sum(weighted_components.values())
    clamped_score = max(-10.0, min(10.0, raw_score))
    clamped_score = round(clamped_score, 2)

    breakdown_str = _format_breakdown(weighted_components)
    direction = _determine_direction(clamped_score)
    signals = _collect_signals(raw_components, clamped_score)

    # ---- ML Scoring & Fallback Ladder Integration ----
    import ml_engine
    
    # Resolve macro_state if None (attempt loading latest saved state)
    if macro_state is None:
        try:
            import json, os
            if os.path.exists("state/last-run.json"):
                with open("state/last-run.json", "r", encoding="utf-8") as f:
                    macro_state = json.load(f).get("macro_state", {})
        except Exception:
            pass
    if macro_state is None:
        macro_state = {"regime": regime or "RISK-ON", "vix": 15.0}

    # Run ML prediction
    prediction = ml_engine.predict_opportunity(asset, macro_state)
    
    # Record features for continuous learning loop
    try:
        # Save features to feature-store.json
        ml_engine.record_asset_run(
            asset, 
            macro_state, 
            sentiment_score=asset.get("sentiment", 0.0), 
            news_volume=asset.get("news_volume", 0)
        )
    except Exception as exc:
        logger.error("Continuous learning record failed for %s: %s", ticker, exc)

    use_ml = False
    if not prediction.get("fallback", True):
        acc = prediction.get("model_accuracy_last_30d", 1.0)
        if acc >= 0.50:
            use_ml = True
        else:
            logger.warning("ML Model accuracy is %.2f (< 50%%). Falling back to rule-based scoring.", acc)

    if use_ml:
        # Override score with ensemble probability-driven rating
        clamped_score = round((prediction["bullish_probability"] - prediction["bearish_probability"]) * 10.0, 2)
        direction = _determine_direction(clamped_score)
        breakdown_str = f"ML ({prediction['model_type']}): Bull {prediction['bullish_probability']:.2f} | Bear {prediction['bearish_probability']:.2f}"
        logger.info("Scored %s via ML pipeline | Bull: %.2f Bear: %.2f -> Score: %s", 
                    ticker, prediction["bullish_probability"], prediction["bearish_probability"], clamped_score)
    else:
        logger.debug("Scored %s via Rule-Based Scorer | %s", ticker, breakdown_str)

    # ---- Backtest-based Signal Suppression ----
    try:
        import backtest
        if direction in ("bullish", "bearish") and backtest.is_signal_suppressed(ticker, direction):
            logger.warning("AUTO-SUPPRESSION: Suppressing %s %s signal (low historical win rate).", ticker, direction.upper())
            signals.append(f"⚠️ Signal SUPPRESSED — {direction.upper()} historical win rate is < 45% in backtesting.")
            clamped_score = 0.0
            direction = "neutral"
            breakdown_str = f"[SUPPRESSED] {breakdown_str}"
    except Exception as exc:
        logger.error("Failed to check signal suppression for %s: %s", ticker, exc)

    # ---- Tactical Card Generation ----
    score_result = {
        "score": clamped_score,
        "direction": direction,
        "breakdown": raw_components,
        "ml_prediction": prediction
    }
    tactical_card = generate_tactical_card(asset, score_result, macro_state)

    return {
        "ticker": ticker,
        "name": name,
        "score": clamped_score,
        "breakdown": raw_components,
        "breakdown_str": breakdown_str,
        "direction": direction,
        "signals": signals,
        "support": asset.get("support"),
        "resistance": asset.get("resistance"),
        "price": asset.get("price"),
        "asset_class": asset.get("asset_class", "stock"),
        "ml_prediction": prediction,
        "tactical_card": tactical_card
    }


# ===========================================================================
# SCAN ALL ASSETS
# ===========================================================================

def scan_all_assets(enriched_data: dict, calendar: list, regime: str | None = None, macro_state: dict | None = None) -> dict:
    """
    Score every asset in the enriched_data dictionary.

    Args:
        enriched_data : Dict mapping ticker → asset dict (as returned by the
                        data enrichment module).  The top-level dict may also
                        contain a 'assets' key whose value is the mapping.
        calendar      : List of upcoming event dicts.
        regime        : Optional market regime string.
        macro_state   : Optional flat macro state dict.

    Returns:
        dict with keys:
            scores      : dict[ticker → score_result_dict]
            bullish     : list of score_result_dicts sorted descending by score
            bearish     : list of score_result_dicts sorted ascending by score
            neutral     : list of score_result_dicts (everything else)
            all_ranked  : full list sorted by score descending
    """
    if not enriched_data or not isinstance(enriched_data, dict):
        logger.warning("scan_all_assets: received empty or invalid enriched_data")
        return {"scores": {}, "bullish": [], "bearish": [], "neutral": [], "all_ranked": []}

    assets_map = enriched_data.get("assets", enriched_data)
    if not isinstance(assets_map, dict):
        logger.warning("scan_all_assets: could not locate assets mapping in enriched_data")
        return {"scores": {}, "bullish": [], "bearish": [], "neutral": [], "all_ranked": []}

    scores: dict[str, dict] = {}
    for ticker, asset in assets_map.items():
        if not isinstance(asset, dict):
            logger.debug("scan_all_assets: skipping non-dict entry for ticker '%s'", ticker)
            continue
        if "ticker" not in asset:
            asset = {**asset, "ticker": ticker}
        result = score_asset(asset, calendar, regime=regime, macro_state=macro_state)
        scores[ticker] = result

    all_ranked = sorted(scores.values(), key=lambda r: r["score"], reverse=True)
    bullish = [r for r in all_ranked if r["score"] >= BULLISH_THRESHOLD]
    bearish = [r for r in sorted(scores.values(), key=lambda r: r["score"]) if r["score"] <= BEARISH_THRESHOLD]
    neutral = [r for r in all_ranked if BEARISH_THRESHOLD < r["score"] < BULLISH_THRESHOLD]

    logger.info(
        "scan_all_assets: scored %d assets | bullish=%d bearish=%d neutral=%d",
        len(scores), len(bullish), len(bearish), len(neutral),
    )

    return {
        "scores": scores,
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "all_ranked": all_ranked,
    }


# ===========================================================================
# PATTERN / OPPORTUNITY DETECTORS
# ===========================================================================

def _safe_float(value, default: float = float("nan")) -> float:
    """Safely convert a value to float, returning default on failure."""
    try:
        f = float(value)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _iter_assets(enriched_data: dict):
    """Yield (ticker, asset_dict) pairs from enriched_data."""
    assets_map = enriched_data.get("assets", enriched_data)
    if not isinstance(assets_map, dict):
        return
    for ticker, asset in assets_map.items():
        if isinstance(asset, dict):
            yield ticker, asset


def detect_breakout_candidates(enriched_data: dict) -> list[dict]:
    """
    Detect assets poised for an upside breakout.

    Criteria:
        - Price is within 1.5 % of a key resistance level
        - Current volume > 1.5× its 20-day average

    Expected asset keys:
        price          : float
        resistance     : float | list[float]  — one or more resistance levels
        volume         : dict with 'current_volume' and 'avg_volume_20d'

    Returns:
        List of dicts: {ticker, name, price, resistance, distance_pct, volume_ratio}
    """
    results = []
    for ticker, asset in _iter_assets(enriched_data):
        price = _safe_float(asset.get("price"))
        if not np.isfinite(price) or price <= 0:
            continue

        # Gather resistance levels
        raw_res = asset.get("resistance")
        if raw_res is None:
            continue
        resistance_levels = raw_res if isinstance(raw_res, list) else [raw_res]
        resistance_levels = [_safe_float(r) for r in resistance_levels if _safe_float(r) > 0]
        if not resistance_levels:
            continue

        # Volume ratio
        vol_data = asset.get("volume", {})
        try:
            cur_vol = float(vol_data.get("current_volume", 0))
            avg_vol = float(vol_data.get("avg_volume_20d", 0))
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 0.0
        except (TypeError, ValueError, ZeroDivisionError):
            vol_ratio = 0.0

        if vol_ratio < VOLUME_SPIKE_RATIO:
            continue

        for resistance in resistance_levels:
            dist_pct = (resistance - price) / price  # positive = price below resistance
            if 0 <= dist_pct <= BREAKOUT_TOLERANCE:
                results.append({
                    "ticker": ticker,
                    "name": asset.get("name", ticker),
                    "price": price,
                    "resistance": resistance,
                    "distance_pct": round(dist_pct * 100, 3),
                    "volume_ratio": round(vol_ratio, 2),
                    "signal": f"Price within {dist_pct*100:.2f}% of resistance {resistance:.4f} with {vol_ratio:.2f}x volume",
                })
                break  # report closest resistance only

    results.sort(key=lambda x: x["distance_pct"])
    logger.info("detect_breakout_candidates: found %d candidates", len(results))
    return results


def detect_breakdown_candidates(enriched_data: dict) -> list[dict]:
    """
    Detect assets poised for a downside breakdown.

    Criteria:
        - Price is within 1.5 % below a key support level
        - Current volume > 1.5× its 20-day average (confirms selling pressure)

    Expected asset keys:
        price    : float
        support  : float | list[float]
        volume   : dict with 'current_volume' and 'avg_volume_20d'

    Returns:
        List of dicts: {ticker, name, price, support, distance_pct, volume_ratio}
    """
    results = []
    for ticker, asset in _iter_assets(enriched_data):
        price = _safe_float(asset.get("price"))
        if not np.isfinite(price) or price <= 0:
            continue

        raw_sup = asset.get("support")
        if raw_sup is None:
            continue
        support_levels = raw_sup if isinstance(raw_sup, list) else [raw_sup]
        support_levels = [_safe_float(s) for s in support_levels if _safe_float(s) > 0]
        if not support_levels:
            continue

        vol_data = asset.get("volume", {})
        try:
            cur_vol = float(vol_data.get("current_volume", 0))
            avg_vol = float(vol_data.get("avg_volume_20d", 0))
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 0.0
        except (TypeError, ValueError, ZeroDivisionError):
            vol_ratio = 0.0

        if vol_ratio < VOLUME_SPIKE_RATIO:
            continue

        for support in support_levels:
            # Price just below support → breakdown
            dist_pct = (price - support) / support  # negative = price below support
            if -BREAKDOWN_TOLERANCE <= dist_pct <= 0:
                abs_dist = abs(dist_pct)
                results.append({
                    "ticker": ticker,
                    "name": asset.get("name", ticker),
                    "price": price,
                    "support": support,
                    "distance_pct": round(abs_dist * 100, 3),
                    "volume_ratio": round(vol_ratio, 2),
                    "signal": f"Price within {abs_dist*100:.2f}% below support {support:.4f} with {vol_ratio:.2f}x volume",
                })
                break

    results.sort(key=lambda x: x["distance_pct"])
    logger.info("detect_breakdown_candidates: found %d candidates", len(results))
    return results


def detect_bb_squeezes(enriched_data: dict) -> list[dict]:
    """
    Detect Bollinger Band squeeze conditions.

    A BB squeeze occurs when the band width (upper - lower) drops to its
    lowest level over the past 20 periods, signalling a volatility
    contraction that typically precedes a sharp expansion move.

    Expected asset keys:
        bb           : dict with:
            upper       : float     — current upper band
            lower       : float     — current lower band
            width_history: list[float] — band widths for last N periods (latest last)

    Returns:
        List of dicts: {ticker, name, current_width, min_width_20p, squeeze_ratio}
    """
    results = []
    for ticker, asset in _iter_assets(enriched_data):
        bb = asset.get("bb")
        if not isinstance(bb, dict):
            continue

        upper = _safe_float(bb.get("upper"))
        lower = _safe_float(bb.get("lower"))
        if not (np.isfinite(upper) and np.isfinite(lower) and upper > lower):
            continue

        current_width = upper - lower
        width_history = bb.get("width_history", [])

        if not width_history or len(width_history) < 2:
            # Cannot determine historical minimum without history
            continue

        try:
            widths = [float(w) for w in width_history if w is not None]
        except (TypeError, ValueError):
            continue

        # Use the last BB_SQUEEZE_LOOKBACK periods (excluding current)
        lookback_widths = widths[-BB_SQUEEZE_LOOKBACK - 1 : -1]
        if not lookback_widths:
            continue

        min_hist_width = min(lookback_widths)
        if min_hist_width <= 0:
            continue

        # Squeeze: current width is at or near the historical minimum
        if current_width <= min_hist_width:
            squeeze_ratio = round(current_width / min_hist_width, 4)
            price = _safe_float(asset.get("price"))
            middle = _safe_float(bb.get("middle"))
            results.append({
                "ticker": ticker,
                "name": asset.get("name", ticker),
                "price": price,
                "current_width": round(current_width, 6),
                "min_width_20p": round(min_hist_width, 6),
                "squeeze_ratio": squeeze_ratio,
                "bb_upper": round(upper, 4),
                "bb_lower": round(lower, 4),
                "bb_middle": round(middle, 4) if np.isfinite(middle) else None,
                "signal": (
                    f"BB width {current_width:.4f} at 20-period low "
                    f"(ratio {squeeze_ratio:.4f}) — breakout imminent"
                ),
            })

    results.sort(key=lambda x: x["squeeze_ratio"])
    logger.info("detect_bb_squeezes: found %d squeeze candidates", len(results))
    return results


def detect_ema_crossovers(enriched_data: dict) -> list[dict]:
    """
    Detect recent golden crosses (EMA20 crosses above EMA50) and
    death crosses (EMA20 crosses below EMA50) within the last 5 bars.

    Expected asset keys:
        ema20_history : list[float] — recent EMA20 values (latest last)
        ema50_history : list[float] — recent EMA50 values (latest last)
        price         : float

    Returns:
        List of dicts: {ticker, name, cross_type, bars_ago, ema20, ema50, price}
        cross_type is 'golden_cross' or 'death_cross'.
    """
    results = []
    lookback = EMA_CROSS_LOOKBACK + 1  # need N+1 bars to detect N crossovers

    for ticker, asset in _iter_assets(enriched_data):
        ema20_hist = asset.get("ema20_history", [])
        ema50_hist = asset.get("ema50_history", [])

        if not ema20_hist or not ema50_hist:
            continue

        try:
            e20 = [float(v) for v in ema20_hist if v is not None]
            e50 = [float(v) for v in ema50_hist if v is not None]
        except (TypeError, ValueError):
            continue

        min_len = min(len(e20), len(e50))
        if min_len < 2:
            continue

        e20 = e20[-lookback:]
        e50 = e50[-lookback:]
        min_len = min(len(e20), len(e50))

        cross_type = None
        bars_ago = None

        for i in range(min_len - 1, 0, -1):
            bars_from_end = min_len - 1 - i
            if bars_from_end > EMA_CROSS_LOOKBACK:
                break
            prev20, prev50 = e20[i - 1], e50[i - 1]
            curr20, curr50 = e20[i], e50[i]

            if prev20 <= prev50 and curr20 > curr50:
                cross_type = "golden_cross"
                bars_ago = bars_from_end
                break
            elif prev20 >= prev50 and curr20 < curr50:
                cross_type = "death_cross"
                bars_ago = bars_from_end
                break

        if cross_type is None:
            continue

        price = _safe_float(asset.get("price"))
        current_ema20 = e20[-1]
        current_ema50 = e50[-1]

        results.append({
            "ticker": ticker,
            "name": asset.get("name", ticker),
            "price": price,
            "cross_type": cross_type,
            "bars_ago": bars_ago,
            "ema20": round(current_ema20, 4),
            "ema50": round(current_ema50, 4),
            "signal": (
                f"{'Golden' if cross_type == 'golden_cross' else 'Death'} cross "
                f"(EMA20 vs EMA50) detected {bars_ago} bar(s) ago"
            ),
        })

    results.sort(key=lambda x: x["bars_ago"])
    logger.info("detect_ema_crossovers: found %d crossover candidates", len(results))
    return results


def detect_correlation_breaks(enriched_data: dict) -> list[dict]:
    """
    Detect pairs of assets that normally move together (high rolling 90d
    correlation) but are recently diverging (low 10d correlation).

    Pairs with a correlation drop > 0.30 are flagged.

    Expected asset keys:
        returns_history : list[float] — daily returns or close prices (latest last)
                          At least 90 data points recommended.

    Returns:
        List of dicts per diverging pair:
        {asset_a, asset_b, long_corr, short_corr, divergence, signal}
    """
    # Collect all assets with return histories
    assets_with_returns: dict[str, np.ndarray] = {}
    for ticker, asset in _iter_assets(enriched_data):
        raw = asset.get("returns_history") or asset.get("price_history")
        if not raw:
            continue
        try:
            vals = np.array([float(v) for v in raw if v is not None], dtype=float)
        except (TypeError, ValueError):
            continue
        # If this looks like prices (not returns), convert to returns
        if len(vals) >= 2 and np.nanmean(np.abs(vals)) > 1:
            vals = np.diff(vals) / vals[:-1]  # simple daily returns
        if len(vals) < CORRELATION_SHORT_WINDOW + 1:
            continue
        assets_with_returns[ticker] = vals

    tickers = list(assets_with_returns.keys())
    results = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            ta = tickers[i]
            tb = tickers[j]
            ra = assets_with_returns[ta]
            rb = assets_with_returns[tb]

            # Align lengths
            min_len = min(len(ra), len(rb))
            ra = ra[-min_len:]
            rb = rb[-min_len:]

            if min_len < CORRELATION_SHORT_WINDOW + 1:
                continue

            # Long-term correlation (up to last 90 days)
            long_len = min(min_len, CORRELATION_LONG_WINDOW)
            ra_long, rb_long = ra[-long_len:], rb[-long_len:]
            if np.std(ra_long) == 0 or np.std(rb_long) == 0:
                continue
            long_corr = float(np.corrcoef(ra_long, rb_long)[0, 1])

            # Short-term correlation (last 10 days)
            ra_short = ra[-CORRELATION_SHORT_WINDOW:]
            rb_short = rb[-CORRELATION_SHORT_WINDOW:]
            if np.std(ra_short) == 0 or np.std(rb_short) == 0:
                continue
            short_corr = float(np.corrcoef(ra_short, rb_short)[0, 1])

            divergence = long_corr - short_corr
            if not np.isfinite(divergence):
                continue

            if divergence >= CORRELATION_DIVERGENCE:
                results.append({
                    "asset_a": ta,
                    "asset_b": tb,
                    "long_corr": round(long_corr, 4),
                    "short_corr": round(short_corr, 4),
                    "divergence": round(divergence, 4),
                    "signal": (
                        f"{ta}/{tb} correlation dropped from {long_corr:.2f} (90d) "
                        f"to {short_corr:.2f} (10d) — divergence {divergence:.2f}"
                    ),
                })

    results.sort(key=lambda x: x["divergence"], reverse=True)
    logger.info("detect_correlation_breaks: found %d diverging pairs", len(results))
    return results


def detect_momentum_plays(enriched_data: dict) -> list[dict]:
    """
    Detect assets making new 52-week highs with increasing volume.

    Criteria:
        - Current price >= 52-week (MOMENTUM_LOOKBACK bars) high
        - Current volume > 1.5× 20-day average volume

    Expected asset keys:
        price        : float
        price_history: list[float] — at least 252 bars of close prices (latest last)
        volume       : dict with 'current_volume' and 'avg_volume_20d'

    Returns:
        List of dicts: {ticker, name, price, high_52w, volume_ratio}
    """
    results = []
    for ticker, asset in _iter_assets(enriched_data):
        price = _safe_float(asset.get("price"))
        if not np.isfinite(price) or price <= 0:
            continue

        price_hist = asset.get("price_history", [])
        if not price_hist or len(price_hist) < 2:
            continue

        try:
            hist = np.array([float(v) for v in price_hist if v is not None], dtype=float)
        except (TypeError, ValueError):
            continue

        # 52-week high from historical data (excluding today)
        lookback_hist = hist[-MOMENTUM_LOOKBACK - 1 : -1]
        if len(lookback_hist) == 0:
            continue
        high_52w = float(np.nanmax(lookback_hist))

        if price < high_52w:
            continue  # not a new high

        # Volume confirmation
        vol_data = asset.get("volume", {})
        try:
            cur_vol = float(vol_data.get("current_volume", 0))
            avg_vol = float(vol_data.get("avg_volume_20d", 0))
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 0.0
        except (TypeError, ValueError, ZeroDivisionError):
            vol_ratio = 0.0

        if vol_ratio < VOLUME_SPIKE_RATIO:
            continue

        results.append({
            "ticker": ticker,
            "name": asset.get("name", ticker),
            "price": round(price, 4),
            "high_52w": round(high_52w, 4),
            "volume_ratio": round(vol_ratio, 2),
            "signal": (
                f"New 52-week high ({price:.4f} >= {high_52w:.4f}) "
                f"with {vol_ratio:.2f}x volume"
            ),
        })

    results.sort(key=lambda x: x["volume_ratio"], reverse=True)
    logger.info("detect_momentum_plays: found %d momentum candidates", len(results))
    return results


def detect_mean_reversion_plays(enriched_data: dict) -> list[dict]:
    """
    Detect assets that are more than 30 % away from their 200-period EMA
    and therefore likely to mean-revert.

    Criteria:
        - |price - ema200| / ema200 > 30 %

    Both extended-to-upside (overbought → revert down) and
    extended-to-downside (oversold → revert up) are flagged with separate
    directions.

    Expected asset keys:
        price  : float
        ema200 : float

    Returns:
        List of dicts: {ticker, name, price, ema200, deviation_pct, direction}
    """
    results = []
    for ticker, asset in _iter_assets(enriched_data):
        price = _safe_float(asset.get("price"))
        ema200 = _safe_float(asset.get("ema200"))

        if not (np.isfinite(price) and np.isfinite(ema200) and ema200 > 0 and price > 0):
            continue

        deviation = (price - ema200) / ema200  # positive = above EMA200
        abs_dev = abs(deviation)

        if abs_dev <= MEAN_REVERSION_THRESHOLD:
            continue

        direction = "revert_down" if deviation > 0 else "revert_up"
        results.append({
            "ticker": ticker,
            "name": asset.get("name", ticker),
            "price": round(price, 4),
            "ema200": round(ema200, 4),
            "deviation_pct": round(deviation * 100, 2),
            "abs_deviation_pct": round(abs_dev * 100, 2),
            "direction": direction,
            "signal": (
                f"Price {deviation*100:+.1f}% from EMA200 ({ema200:.4f}) — "
                f"{'extended above' if deviation > 0 else 'extended below'} EMA200, "
                f"likely to {'pull back' if deviation > 0 else 'bounce'}"
            ),
        })

    results.sort(key=lambda x: x["abs_deviation_pct"], reverse=True)
    logger.info("detect_mean_reversion_plays: found %d mean-reversion candidates", len(results))
    return results


def detect_oversold_bounces(enriched_data: dict) -> list[dict]:
    """
    Detect assets that are deeply oversold (RSI < 30) and trading near
    a key support level.

    Criteria:
        - RSI < 30
        - Price within 2 % of nearest support level

    Expected asset keys:
        rsi     : float
        price   : float
        support : float | list[float]

    Returns:
        List of dicts: {ticker, name, rsi, price, support, distance_to_support_pct}
    """
    SUPPORT_PROXIMITY = 0.02   # within 2 % of support

    results = []
    for ticker, asset in _iter_assets(enriched_data):
        rsi = _safe_float(asset.get("rsi"))
        if not np.isfinite(rsi) or rsi >= OVERSOLD_RSI_THRESHOLD:
            continue

        price = _safe_float(asset.get("price"))
        if not np.isfinite(price) or price <= 0:
            continue

        raw_sup = asset.get("support")
        if raw_sup is None:
            # Can still flag as oversold even without support data
            results.append({
                "ticker": ticker,
                "name": asset.get("name", ticker),
                "rsi": round(rsi, 2),
                "price": round(price, 4),
                "support": None,
                "distance_to_support_pct": None,
                "signal": f"RSI deeply oversold at {rsi:.1f} — potential bounce",
            })
            continue

        support_levels = raw_sup if isinstance(raw_sup, list) else [raw_sup]
        support_levels = [_safe_float(s) for s in support_levels if _safe_float(s) > 0]

        closest_sup = None
        closest_dist = float("inf")
        for sup in support_levels:
            dist = abs(price - sup) / sup
            if dist < closest_dist:
                closest_dist = dist
                closest_sup = sup

        near_support = closest_sup is not None and closest_dist <= SUPPORT_PROXIMITY

        results.append({
            "ticker": ticker,
            "name": asset.get("name", ticker),
            "rsi": round(rsi, 2),
            "price": round(price, 4),
            "support": round(closest_sup, 4) if closest_sup else None,
            "distance_to_support_pct": round(closest_dist * 100, 3) if closest_sup else None,
            "near_support": near_support,
            "signal": (
                f"RSI {rsi:.1f} (very oversold) "
                + (
                    f"+ price within {closest_dist*100:.2f}% of support {closest_sup:.4f}"
                    if near_support
                    else "— no nearby support confirmed"
                )
            ),
        })

    results.sort(key=lambda x: x["rsi"])
    logger.info("detect_oversold_bounces: found %d oversold-bounce candidates", len(results))
    return results


# ===========================================================================
# RANK OPPORTUNITIES
# ===========================================================================

def rank_opportunities(scan_results: dict) -> dict:
    """
    Consolidate scan results and pattern detections into a single ranked
    opportunity report.

    Args:
        scan_results : The dictionary returned by scan_all_assets(), optionally
                       augmented with pattern-detection lists under the keys:
                           breakouts, breakdowns, squeezes, crossovers,
                           momentum, mean_reversion, correlation_breaks, oversold

    Returns:
        dict with keys:
            bullish           : list — score >= +6 (sorted desc by score)
            bearish           : list — score <= -6 (sorted asc by score, worst first)
            neutral           : list — everything else
            breakouts         : list — from detect_breakout_candidates
            breakdowns        : list — from detect_breakdown_candidates
            squeezes          : list — from detect_bb_squeezes
            crossovers        : list — from detect_ema_crossovers
            momentum          : list — from detect_momentum_plays
            mean_reversion    : list — from detect_mean_reversion_plays
            correlation_breaks: list — from detect_correlation_breaks
            oversold          : list — from detect_oversold_bounces
            summary           : dict — counts per category
    """
    if not scan_results or not isinstance(scan_results, dict):
        logger.warning("rank_opportunities: received empty or invalid scan_results")
        empty: dict[str, Any] = {
            "bullish": [], "bearish": [], "neutral": [],
            "breakouts": [], "breakdowns": [], "squeezes": [],
            "crossovers": [], "momentum": [], "mean_reversion": [],
            "correlation_breaks": [], "oversold": [], "summary": {},
        }
        return empty

    bullish = sorted(
        scan_results.get("bullish", []), key=lambda r: r.get("score", 0), reverse=True
    )
    bearish = sorted(
        scan_results.get("bearish", []), key=lambda r: r.get("score", 0)
    )
    neutral = scan_results.get("neutral", [])

    breakouts = scan_results.get("breakouts", [])
    breakdowns = scan_results.get("breakdowns", [])
    squeezes = scan_results.get("squeezes", [])
    crossovers = scan_results.get("crossovers", [])
    momentum = scan_results.get("momentum", [])
    mean_reversion = scan_results.get("mean_reversion", [])
    correlation_breaks = scan_results.get("correlation_breaks", [])
    oversold = scan_results.get("oversold", [])

    summary = {
        "bullish_count": len(bullish),
        "bearish_count": len(bearish),
        "neutral_count": len(neutral),
        "breakout_count": len(breakouts),
        "breakdown_count": len(breakdowns),
        "squeeze_count": len(squeezes),
        "crossover_count": len(crossovers),
        "momentum_count": len(momentum),
        "mean_reversion_count": len(mean_reversion),
        "correlation_break_count": len(correlation_breaks),
        "oversold_bounce_count": len(oversold),
        "total_assets_scored": len(scan_results.get("scores", {})),
    }

    logger.info(
        "rank_opportunities: bullish=%d bearish=%d neutral=%d | "
        "breakouts=%d breakdowns=%d squeezes=%d crossovers=%d "
        "momentum=%d mean_reversion=%d corr_breaks=%d oversold=%d",
        summary["bullish_count"], summary["bearish_count"], summary["neutral_count"],
        summary["breakout_count"], summary["breakdown_count"],
        summary["squeeze_count"], summary["crossover_count"],
        summary["momentum_count"], summary["mean_reversion_count"],
        summary["correlation_break_count"], summary["oversold_bounce_count"],
    )

    return {
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "breakouts": breakouts,
        "breakdowns": breakdowns,
        "squeezes": squeezes,
        "crossovers": crossovers,
        "momentum": momentum,
        "mean_reversion": mean_reversion,
        "correlation_breaks": correlation_breaks,
        "oversold": oversold,
        "summary": summary,
    }


# ===========================================================================
# CONVENIENCE: FULL PIPELINE
# ===========================================================================

def compute_enhanced_score_with_confidence(
    asset: dict,
    base_score: float,
    macro_state: dict | None = None,
    sentiment_score: float = 0.0,
) -> dict:
    """
    PHASE 2+ UPGRADE: Enhance base rule-based score with ML predictions and confidence.
    
    Returns enhanced prediction with:
    - Combined ML + rule-based probability
    - Model confidence level
    - Supporting evidence
    - Entry/exit zones
    - Stop loss invalidation
    """
    ticker = asset.get("ticker", "UNKNOWN")
    
    result = {
        "ticker": ticker,
        "base_rule_score": float(base_score),
        "bullish_probability": 0.5,     # Default neutral
        "bearish_probability": 0.5,
        "neutral_probability": 0.0,
        "model_confidence": "LOW",
        "confidence_stars": "★☆☆☆☆",
        "evidence": [],
        "top_features": [],
        "entry_zone_low": None,
        "entry_zone_high": None,
        "stop_loss": None,
        "time_horizon": "3-7 days",
        "ml_fallback": True,  # Started as fallback
        "model_accuracy_30d": 0.50,
        "sample_size": 0,
        "backtest_win_rate": 0.0,
    }
    
    if not macro_state:
        macro_state = {"regime": "RISK-ON", "vix": 15.0}
    
    price = asset.get("price", 0.0)
    if price <= 0:
        return result
    
    # Try to get ML prediction
    if ML_AVAILABLE:
        try:
            ml_pred = ml_engine.predict_opportunity(asset, macro_state)
            if ml_pred and not ml_pred.get("fallback", True):
                result["ml_fallback"] = False
                result["bullish_probability"] = ml_pred.get("bullish_probability", 0.5)
                result["bearish_probability"] = ml_pred.get("bearish_probability", 0.3)
                result["neutral_probability"] = ml_pred.get("neutral_probability", 0.2)
                result["model_confidence"] = ml_pred.get("model_confidence", "LOW")
                result["top_features"] = ml_pred.get("top_features", [])
                result["model_accuracy_30d"] = ml_pred.get("model_accuracy_last_30d", 0.5)
                result["sample_size"] = ml_pred.get("sample_size", 0)
                
                # Map confidence to stars
                if result["model_confidence"] == "HIGH":
                    result["confidence_stars"] = "★★★★★"
                elif result["model_confidence"] == "MEDIUM":
                    result["confidence_stars"] = "★★★☆☆"
                else:
                    result["confidence_stars"] = "★★☆☆☆"
        except Exception as exc:
            logger.debug("ML prediction failed for %s: %s", ticker, exc)
    
    # Rule-based evidence
    rsi = asset.get("rsi", 50.0)
    ema20 = asset.get("ema20", price)
    ema50 = asset.get("ema50", price)
    ema200 = asset.get("ema200", price)
    
    evidence_count = 0
    if rsi < 35:
        result["evidence"].append(f"✅ RSI {rsi:.0f} — oversold bounce setup")
        evidence_count += 1
    elif rsi > 65:
        result["evidence"].append(f"⚠️  RSI {rsi:.0f} — overbought caution")
        evidence_count += 1
    
    if price > ema20 and price < ema50:
        result["evidence"].append("✅ Price between EMA20 and EMA50 — bullish structure")
        evidence_count += 1
    
    if price > ema200:
        result["evidence"].append("✅ Price above EMA200 — long-term trend bullish")
        evidence_count += 1
    elif price < ema200:
        result["evidence"].append("❌ Price below EMA200 — long-term trend bearish")
    
    vol_data = asset.get("volume_anomaly", {})
    if vol_data.get("ratio", 1.0) > 1.5:
        result["evidence"].append(f"✅ Volume {vol_data.get('ratio', 1):.1f}x avg — institutional buying")
        evidence_count += 1
    
    sentiment_score = sentiment_score or asset.get("sentiment", 0.0)
    if sentiment_score > 0.3:
        result["evidence"].append(f"✅ Sentiment {sentiment_score:+.2f} — bullish narrative")
        evidence_count += 1
    elif sentiment_score < -0.3:
        result["evidence"].append(f"❌ Sentiment {sentiment_score:+.2f} — bearish narrative")
    
    macro_regime = macro_state.get("regime", "RISK-ON")
    if macro_regime == "RISK-OFF":
        result["evidence"].append("⚠️  Regime RISK-OFF — reduces conviction")
    
    # Calculate evidence ratio (0-6 factors)
    max_factors = 6
    evidence_pct = min(100, (evidence_count / max_factors) * 100)
    
    # Adjust probabilities based on evidence
    if evidence_count >= 4:
        result["bullish_probability"] = min(1.0, result["bullish_probability"] + 0.15)
        result["bearish_probability"] = max(0.0, result["bearish_probability"] - 0.10)
    elif evidence_count <= 1:
        result["bullish_probability"] = max(0.0, result["bullish_probability"] - 0.20)
        result["bearish_probability"] = min(1.0, result["bearish_probability"] + 0.15)
    
    # Normalize
    total = result["bullish_probability"] + result["bearish_probability"] + result["neutral_probability"]
    if total > 0:
        result["bullish_probability"] = round(result["bullish_probability"] / total, 2)
        result["bearish_probability"] = round(result["bearish_probability"] / total, 2)
        result["neutral_probability"] = round(1.0 - result["bullish_probability"] - result["bearish_probability"], 2)
    
    # Entry zones
    pct_to_ema20 = ((price - ema20) / ema20 * 100.0) if ema20 else 0.0
    if result["bullish_probability"] > 0.6:
        # Entry: 2-3% pullback from current
        result["entry_zone_low"] = round(price * 0.97, 2)
        result["entry_zone_high"] = round(price * 1.00, 2)
        # Stop: below EMA50
        result["stop_loss"] = round(ema50 * 0.98, 2)
    elif result["bearish_probability"] > 0.6:
        result["entry_zone_low"] = round(price * 1.00, 2)
        result["entry_zone_high"] = round(price * 1.03, 2)
        result["stop_loss"] = round(ema50 * 1.02, 2)
    
    return result


def run_full_scan_with_ml(enriched_data: dict, calendar: list, regime: str | None = None, macro_state: dict | None = None) -> dict:
    """
    PHASE 2+ UPGRADE: Full scan with ML predictions and confidence scoring.
    
    Same as run_full_scan but includes:
    - ML-powered predictions
    - Confidence levels
    - Entry/exit zones
    - Feature importance
    """
    logger.info("run_full_scan_with_ml: starting ML-enhanced scan pipeline")
    
    if not macro_state:
        macro_state = {"regime": regime or "RISK-ON", "vix": 15.0}
    
    # Step 1: Score all assets (rule-based)
    base_results = scan_all_assets(enriched_data, calendar, regime=regime, macro_state=macro_state)
    
    # Step 2: Enhance with ML + confidence
    enhanced_results = []
    for opp in base_results.get("opportunities", []):
        ticker = opp.get("ticker")
        asset = enriched_data.get(ticker)
        if asset:
            sentiment = opp.get("sentiment", 0.0)
            enhanced = compute_enhanced_score_with_confidence(
                asset, opp.get("score", 0.0), macro_state, sentiment
            )
            enhanced_results.append(enhanced)
    
    base_results["enhanced_predictions"] = enhanced_results
    
    # Step 3: Pattern detectors (unchanged)
    base_results["breakouts"] = detect_breakout_candidates(enriched_data)
    base_results["breakdowns"] = detect_breakdown_candidates(enriched_data)
    base_results["squeezes"] = detect_bb_squeezes(enriched_data)
    base_results["crossovers"] = detect_ema_crossovers(enriched_data)
    base_results["momentum"] = detect_momentum_plays(enriched_data)
    base_results["mean_reversion"] = detect_mean_reversion_plays(enriched_data)
    base_results["correlation_breaks"] = detect_correlation_breaks(enriched_data)
    base_results["oversold"] = detect_oversold_bounces(enriched_data)
    
    # Step 4: Rank
    ranked = rank_opportunities(base_results)
    logger.info("run_full_scan_with_ml: pipeline complete with %d enhanced predictions", len(enhanced_results))
    
    return ranked


# Backward-compatible alias — luna.py imports run_full_scan which was the
# original name before it was upgraded to run_full_scan_with_ml.
run_full_scan = run_full_scan_with_ml


# ===========================================================================
# MODULE SELF-TEST (python scanner.py)
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # ---- Minimal synthetic data for smoke-test ----
    _calendar = [
        {
            "date": (datetime.now(tz=timezone.utc) + timedelta(days=1)).isoformat(),
            "ticker": "SYNTH",
            "title": "SYNTH Earnings Release",
            "impact": "High",
            "direction": "bullish",
        }
    ]

    _price_hist = list(np.linspace(90, 105, 260))   # trending up
    _ema20_hist = list(np.linspace(88, 103, 30))
    _ema50_hist = list(np.linspace(85, 100, 60))

    _synthetic_assets = {
        "SYNTH": {
            "ticker": "SYNTH",
            "name": "Synthetic Corp",
            "price": 106.0,
            "rsi": 28.0,
            "macd": {
                "macd": [0.5, 0.8, 1.1, 1.4],
                "signal": [0.4, 0.6, 0.7, 0.9],
            },
            "ema20": 103.0,
            "ema50": 100.0,
            "ema200": 95.0,
            "ema20_history": _ema20_hist,
            "ema50_history": _ema50_hist,
            "volume": {"current_volume": 2_500_000, "avg_volume_20d": 1_000_000},
            "bb": {
                "upper": 110.0,
                "middle": 103.0,
                "lower": 96.0,
                "width_history": [15.0, 14.5, 14.0, 13.5, 13.0, 12.8, 12.5, 12.0],
            },
            "sentiment": 0.6,
            "support": [100.0, 95.0],
            "resistance": [108.0],
            "price_history": _price_hist,
            "returns_history": list(np.diff(_price_hist) / np.array(_price_hist[:-1])),
        },
        "BEAR_CO": {
            "ticker": "BEAR_CO",
            "name": "Bear Corp",
            "price": 48.0,
            "rsi": 75.0,
            "macd": {
                "macd": [1.2, 0.8, 0.3, -0.2],
                "signal": [1.0, 1.0, 0.9, 0.8],
            },
            "ema20": 46.0,
            "ema50": 50.0,
            "ema200": 55.0,
            "ema20_history": list(np.linspace(55, 46, 30)),
            "ema50_history": list(np.linspace(58, 50, 60)),
            "volume": {"current_volume": 800_000, "avg_volume_20d": 1_000_000},
            "bb": {
                "upper": 58.0,
                "middle": 52.0,
                "lower": 46.0,
                "width_history": [12.0] * 10,
            },
            "sentiment": -0.4,
            "support": [45.0],
            "resistance": [55.0, 60.0],
            "price_history": list(np.linspace(65, 48, 260)),
            "returns_history": list(np.diff(np.linspace(65, 48, 260)) / np.linspace(65, 48, 260)[:-1]),
        },
    }

    _results = run_full_scan({"assets": _synthetic_assets}, _calendar)

    print("\n" + "=" * 60)
    print("FULL SCAN RESULTS — SUMMARY")
    print("=" * 60)
    for k, v in _results["summary"].items():
        print(f"  {k:<30}: {v}")

    print("\n--- BULLISH OPPORTUNITIES ---")
    for r in _results["bullish"]:
        print(f"  {r['ticker']:10} | score {r['score']:+3d} | {r['breakdown_str']}")
        for sig in r["signals"]:
            print(f"               ↳ {sig}")

    print("\n--- BEARISH OPPORTUNITIES ---")
    for r in _results["bearish"]:
        print(f"  {r['ticker']:10} | score {r['score']:+3d} | {r['breakdown_str']}")
        for sig in r["signals"]:
            print(f"               ↳ {sig}")

    print("\n--- MOMENTUM PLAYS ---")
    for m in _results["momentum"]:
        print(f"  {m['ticker']:10} | {m['signal']}")

    print("\n--- MEAN REVERSION PLAYS ---")
    for m in _results["mean_reversion"]:
        print(f"  {m['ticker']:10} | {m['signal']}")

    print("\nSmoke-test complete.\n")
