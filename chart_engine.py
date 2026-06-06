"""
chart_engine.py
===============
LUNA Chart Analysis Engine — institutional-grade chart analysis with vision AI.

When triggered via `python luna.py --chart <image_path>`:
  1. Loads the chart image from the provided path
  2. Loads relevant asset data from state/last-run.json if the asset is tracked
  3. Passes both the image and LUNA context to the analysis engine (Gemini Vision)
  4. Outputs full analysis in the terminal using rich formatting
  5. Saves the analysis to reports/chart-analyses/{asset}_{timestamp}.md
  6. Logs the setup to state/calls-log.json for outcome tracking

Dependencies:
  pip install anthropic Pillow rich
  pip install google-generativeai   # optional Gemini fallback
  pip install openai                # optional OpenAI fallback
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("chart_engine")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).parent
STATE_DIR = _BASE_DIR / "state"
REPORTS_DIR = _BASE_DIR / "reports" / "chart-analyses"
CALLS_LOG_PATH = STATE_DIR / "calls-log.json"
LAST_RUN_PATH = STATE_DIR / "last-run.json"

# ---------------------------------------------------------------------------
# LUNA Chart Analysis System Prompt
# ---------------------------------------------------------------------------
CHART_SYSTEM_PROMPT = """You are LUNA's Chart Analysis Engine — an elite institutional trader and risk manager with 20+ years of experience across equities, crypto, forex, and commodities.

When a chart image is provided, perform a complete institutional-grade trading analysis. Never force a trade. Capital preservation is the priority above all else.

ANALYSIS FRAMEWORK:

Step 1 — Asset Identification
- Identify the asset/market (stock, crypto, forex, commodity, index)
- Identify the timeframe (1m, 5m, 15m, 1h, 4h, 1D, 1W)
- Note current price and approximate date/time if visible

Step 2 — Market Structure
- Higher Highs (HH) + Higher Lows (HL) = Bullish structure
- Lower Highs (LH) + Lower Lows (LL) = Bearish structure
- Equal highs/lows = Range-bound
- Identify Break of Structure (BOS) and Change of Character (CHoCH)

Step 3 — Supply and Demand Zones
- Identify strong supply zones (areas where price dropped aggressively)
- Identify strong demand zones (areas where price rallied aggressively)
- Rate zone strength: STRONG (tested once), MODERATE (tested twice), WEAK (3+ tests)

Step 4 — Key Levels
- Absolute support and resistance
- Previous swing highs/lows
- Round number psychological levels
- Weekly/Monthly highs and lows if visible

Step 5 — Trendlines and Channels
- Draw primary trendline (3+ touches)
- Ascending/descending channels
- Trendline breaks with volume confirmation

Step 6 — Candlestick Patterns
- Reversal: Pin bar, Engulfing, Doji, Morning/Evening Star, Hammer, Shooting Star
- Continuation: Inside bar, Marubozu, Three soldiers/crows
- Rejection wicks

Step 7 — Indicators (if visible)
- RSI: oversold <30, overbought >70, divergences, 50-line momentum
- MACD: histogram expansion/contraction, signal crossovers
- Moving Averages: 200MA context, dynamic support/resistance
- Volume: breakout confirmation, trend strength, reversal spikes

Step 8 — Recommended Free Indicators (2 max for TradingView free tier)
- Trending: 200 EMA + RSI
- Ranging: RSI + Bollinger Bands
- Breakout: Volume + RSI
- Reversal: RSI + MACD

TRADE SETUP RULES:
- Entry Criteria: price at key structural level, candlestick confirmation, indicator confluence, R:R ≥ 1:2, clear invalidation
- Position Sizing: $10,000 account, 1% risk = $100 per trade
- TP Methodology: TP1 50% exit, TP2 30% exit, TP3 20% exit

OUTPUT FORMAT — Use EXACTLY this format:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌙 LUNA CHART ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ASSET: [Name + Ticker]
TIMEFRAME: [Timeframe]
CURRENT PRICE: [Price]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TREND: [BULLISH / BEARISH / RANGE-BOUND]
STRUCTURE: [HH/HL or LH/LL or Range — describe last 3 swings]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY LEVELS:
Resistance 3: [price]
Resistance 2: [price]
Resistance 1: [price] ← nearest
Current Price: [price]
Support 1: [price] ← nearest
Support 2: [price]
Support 3: [price]

SUPPLY ZONES: [price range] — [STRONG/MODERATE/WEAK]
DEMAND ZONES: [price range] — [STRONG/MODERATE/WEAK]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETUP: [BUY / SELL / NO TRADE — WAIT]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ENTRY: [price] — [reason]
STOP LOSS: [price] — [reason]
INVALIDATION: [exact condition]

TAKE PROFIT:
TP1: [price] | +[X]% | R:R 1:[X] | Exit 50% position
TP2: [price] | +[X]% | R:R 1:[X] | Exit 30% position
TP3: [price] | +[X]% | R:R 1:[X] | Exit remaining 20%

OVERALL R:R: 1:[X]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POSITION SIZING ($10,000 account, 1% risk):
Risk amount: $100
Stop distance: [X]%
Position size: [X units/shares/contracts]
Max loss if stopped: $100 (1% of account)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONFLUENCE CHECKLIST:
✅/❌ Market structure aligned
✅/❌ Price at key level
✅/❌ Candlestick confirmation
✅/❌ RSI confluence
✅/❌ MACD confluence
✅/❌ Volume confirmation
✅/❌ Trend alignment (higher timeframe)
✅/❌ R:R minimum 1:2

CONFIDENCE: [X/10]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED INDICATORS (Free — pick 2):
Primary: [Indicator + settings] — [why]
Secondary: [Indicator + settings] — [why]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REASONING:
[Detailed paragraph covering why this setup is high probability, what market structure reveals, what smart money is likely doing, what would make you wrong, what you are waiting for if NO TRADE]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  RISK WARNING:
This is technical analysis output only.
Not financial advice. Always do your own research.
Never risk more than you can afford to lose.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HARD RULES:
1. Never force a trade — if setup is not clean say NO TRADE
2. Never recommend less than 1:2 R:R under any circumstances
3. Always show exact position size calculation
4. Always give exact price levels — never say "around" or "approximately"
5. Always explain the invalidation condition precisely
6. Confidence below 6/10 = NO TRADE automatically
7. Always recommend the optimal 2 free TradingView indicators for that specific setup
8. Smart money concepts take priority: order blocks, fair value gaps, liquidity sweeps
9. When in doubt — NO TRADE. Capital preservation above everything
"""

# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """
    Load image from disk and return (base64_data, mime_type).
    Supports PNG, JPEG, WEBP, GIF.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Chart image not found: {image_path}")

    ext = path.suffix.lower()
    mime_map = {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif":  "image/gif",
    }
    mime_type = mime_map.get(ext, "image/png")

    with open(path, "rb") as f:
        raw = f.read()

    b64 = base64.b64encode(raw).decode("utf-8")
    logger.info("Loaded chart image: %s (%.1f KB, %s)", path.name, len(raw) / 1024, mime_type)
    return b64, mime_type


# ---------------------------------------------------------------------------
# Asset detection from filename
# ---------------------------------------------------------------------------

def _infer_asset_from_path(image_path: str) -> str:
    """
    Try to infer ticker/asset name from the image filename.
    E.g. 'NVDA_4h.png' -> 'NVDA', 'BTCUSDT_1d.png' -> 'BTC'
    """
    name = Path(image_path).stem.upper()
    # Strip common timeframe suffixes
    for suffix in ["_1M", "_5M", "_15M", "_1H", "_4H", "_1D", "_1W", "_D", "_H", "_W"]:
        name = name.replace(suffix, "")
    # Remove trailing underscores/dashes
    name = name.strip("_-").strip()
    # Map common pairs
    alias_map = {
        "BTCUSDT": "BTC",
        "ETHUSDT": "ETH",
        "BTCUSD":  "BTC",
        "ETHUSD":  "ETH",
    }
    return alias_map.get(name, name)


# ---------------------------------------------------------------------------
# LUNA context loader
# ---------------------------------------------------------------------------

def load_luna_context(ticker: str) -> dict:
    """
    Load relevant asset data from state/last-run.json for a given ticker.
    Returns a dict of available LUNA data, or empty dict if not found.
    """
    if not LAST_RUN_PATH.exists():
        logger.debug("state/last-run.json not found — no LUNA context available.")
        return {}

    try:
        with open(LAST_RUN_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read last-run.json: %s", exc)
        return {}

    ticker_upper = ticker.upper()

    # Search global_snapshot
    snapshot = state.get("market_data", {}).get("global_snapshot", [])
    asset_data: dict = {}
    for asset in snapshot:
        t = asset.get("ticker", "").upper()
        if t == ticker_upper or t.startswith(ticker_upper):
            asset_data = asset
            break

    # Search opportunities for this ticker
    opps = state.get("opportunities", [])
    opp_data: dict = {}
    for opp in opps:
        if opp.get("ticker", "").upper() == ticker_upper:
            opp_data = opp
            break

    # Macro state summary
    macro = state.get("macro_state", {})
    macro_context = {
        "regime":     macro.get("regime", "UNKNOWN"),
        "vix":        macro.get("vix"),
        "dxy":        macro.get("dxy"),
        "yield_10y":  macro.get("yield_10y"),
        "risk_on":    macro.get("risk_on_score"),
        "risk_off":   macro.get("risk_off_score"),
        "summary":    macro.get("macro_summary", ""),
    }

    result = {
        "ticker":       ticker_upper,
        "price":        asset_data.get("price"),
        "change_24h":   asset_data.get("change_24h"),
        "change_7d":    asset_data.get("change_7d"),
        "macro":        macro_context,
        "opportunity":  opp_data,
        "last_updated": state.get("timestamp", ""),
    }

    # Filter out None values for cleanliness
    return {k: v for k, v in result.items() if v is not None and v != {} and v != ""}


def _build_luna_context_block(ticker: str, context: dict) -> str:
    """Build a human-readable LUNA context block to inject into the prompt."""
    if not context:
        return ""

    lines = [f"\n--- LUNA LIVE DATA for {ticker} ---"]

    if context.get("price"):
        lines.append(f"Current Price: ${context['price']:,.4f}")
    if context.get("change_24h") is not None:
        lines.append(f"24h Change: {context['change_24h']:+.2f}%")
    if context.get("change_7d") is not None:
        lines.append(f"7d Change: {context['change_7d']:+.2f}%")

    macro = context.get("macro", {})
    if macro:
        lines.append(f"\nMacro Regime: {macro.get('regime', 'N/A')}")
        if macro.get("vix"):
            lines.append(f"VIX: {macro['vix']:.2f}")
        if macro.get("dxy"):
            lines.append(f"DXY: {macro['dxy']:.3f}")
        if macro.get("yield_10y"):
            lines.append(f"10Y Yield: {macro['yield_10y']:.3f}%")
        lines.append(f"Risk-On Score: {macro.get('risk_on', 'N/A')}/6")

    opp = context.get("opportunity", {})
    if opp:
        lines.append(f"\nLUNA Opportunity Score: {opp.get('score', 'N/A')}")
        lines.append(f"LUNA Bias: {opp.get('bias', 'N/A').upper()}")
        reasoning = opp.get("reasoning", "")
        if reasoning:
            lines.append(f"LUNA Reasoning: {reasoning[:200]}")

    if context.get("last_updated"):
        lines.append(f"\nData as of: {context['last_updated'][:19]}")

    lines.append("--- END LUNA DATA ---\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AI backend: Claude Vision (PRIMARY)
# ---------------------------------------------------------------------------

def _call_claude(
    image_b64: str,
    mime_type: str,
    luna_context_block: str,
    api_key: str,
    model: str = "claude-opus-4-5",
) -> str:
    """Call Anthropic Claude Vision API with image + context."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic is required for chart analysis.\n"
            "Install it with: pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=api_key)

    user_text = (
        "Analyze this trading chart completely using the LUNA Chart Analysis framework.\n"
        "Apply all 8 analysis steps and output the full report in the exact format specified."
    )

    if luna_context_block:
        user_text += (
            "\n\nBelow is LUNA's live market data for this asset. "
            "Cross-reference it with your chart analysis. "
            "If LUNA data confirms the chart setup, add ✅ LUNA DATA CONFIRMS. "
            "If it contradicts, add ⚠️ LUNA DATA CONFLICTS — investigate before entry.\n"
            + luna_context_block
        )

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=CHART_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            }
        ],
    )

    return message.content[0].text.strip()


# ---------------------------------------------------------------------------
# AI backend: Gemini Vision (SECONDARY fallback)
# ---------------------------------------------------------------------------

def _call_gemini(
    image_b64: str,
    mime_type: str,
    luna_context_block: str,
    api_key: str,
    model: str = "gemini-1.5-flash",
) -> str:
    """Call Google Gemini Vision API with image + context."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai is required for Gemini fallback.\n"
            "Install it with: pip install google-generativeai"
        )

    genai.configure(api_key=api_key)

    model_instance = genai.GenerativeModel(
        model_name=model,
        system_instruction=CHART_SYSTEM_PROMPT,
    )

    user_prompt = (
        "Analyze this trading chart completely using the LUNA Chart Analysis framework.\n"
        "Apply all 8 analysis steps and output the full report in the exact format specified."
    )

    if luna_context_block:
        user_prompt += (
            "\n\nBelow is LUNA's live market data for this asset. "
            "Cross-reference it with your chart analysis. "
            "If LUNA data confirms the chart setup, add ✅ LUNA DATA CONFIRMS. "
            "If it contradicts, add ⚠️ LUNA DATA CONFLICTS — investigate before entry.\n"
            + luna_context_block
        )

    import google.generativeai.types as gtypes
    image_part = gtypes.BlobPart(data=base64.b64decode(image_b64), mime_type=mime_type)

    response = model_instance.generate_content(
        [user_prompt, image_part],
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=4096,
        ),
    )

    return response.text.strip()


# ---------------------------------------------------------------------------
# AI backend: OpenAI GPT-4o Vision (fallback)
# ---------------------------------------------------------------------------

def _call_openai_vision(
    image_b64: str,
    mime_type: str,
    luna_context_block: str,
    api_key: str,
    model: str = "gpt-4o",
) -> str:
    """Call OpenAI GPT-4 Vision API as fallback."""
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai is required as fallback for chart analysis.\n"
            "Install it with: pip install openai"
        )

    client = openai.OpenAI(api_key=api_key)

    user_content = [
        {
            "type": "text",
            "text": (
                "Analyze this trading chart completely using the LUNA Chart Analysis framework.\n"
                "Apply all 8 analysis steps and output the full report in the exact format specified."
                + ("\n\nLUNA Live Data:\n" + luna_context_block if luna_context_block else "")
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_b64}",
                "detail": "high",
            },
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CHART_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=4096,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Analysis dispatcher
# ---------------------------------------------------------------------------

def run_chart_analysis(image_path: str, config: dict) -> str:
    """
    Main entry point for chart analysis.

    Backend priority order:
      1. Claude (Anthropic)  — PRIMARY  — set ANTHROPIC_API_KEY in .env
      2. Gemini (Google)     — SECONDARY — set GEMINI_API_KEY in .env
      3. GPT-4o (OpenAI)    — TERTIARY  — set OPENAI_API_KEY in .env

    Steps:
      1. Load image from disk
      2. Detect asset ticker from filename
      3. Load LUNA context from state/last-run.json
      4. Call the first available vision AI backend
      5. Return full analysis text
    """
    # Detect ticker
    ticker = _infer_asset_from_path(image_path)
    logger.info("Detected ticker from filename: %s", ticker)

    # Load image
    image_b64, mime_type = load_image_as_base64(image_path)

    # Load LUNA context
    luna_context = load_luna_context(ticker)
    luna_block = _build_luna_context_block(ticker, luna_context)

    if luna_context:
        logger.info("LUNA context loaded for %s — will cross-reference with chart.", ticker)
    else:
        logger.info("No LUNA context found for %s — proceeding with chart-only analysis.", ticker)

    # --- Backend 1: Claude (PRIMARY) ---
    claude_key = config.get("ANTHROPIC_API_KEY")
    claude_model = config.get("CLAUDE_CHART_MODEL", "claude-opus-4-5")
    if claude_key:
        logger.info("Using Claude %s for chart analysis...", claude_model)
        try:
            analysis = _call_claude(image_b64, mime_type, luna_block, claude_key, model=claude_model)
            logger.info("Claude analysis complete.")
            return analysis
        except Exception as exc:
            logger.warning("Claude failed (%s) — trying Gemini fallback...", exc)

    # --- Backend 2: Gemini (SECONDARY) ---
    gemini_key = config.get("GEMINI_API_KEY") or config.get("GOOGLE_API_KEY")
    if gemini_key:
        logger.info("Using Google Gemini Vision for chart analysis...")
        try:
            analysis = _call_gemini(image_b64, mime_type, luna_block, gemini_key)
            logger.info("Gemini analysis complete.")
            return analysis
        except Exception as exc:
            logger.warning("Gemini failed (%s) — trying OpenAI fallback...", exc)

    # --- Backend 3: OpenAI GPT-4o (TERTIARY) ---
    openai_key = config.get("OPENAI_API_KEY")
    if openai_key:
        logger.info("Using OpenAI GPT-4o Vision for chart analysis...")
        try:
            analysis = _call_openai_vision(image_b64, mime_type, luna_block, openai_key)
            logger.info("OpenAI analysis complete.")
            return analysis
        except Exception as exc:
            logger.error("OpenAI also failed: %s", exc)
            raise

    raise RuntimeError(
        "No vision AI API key configured.\n"
        "Add one of the following to your .env file:\n"
        "  ANTHROPIC_API_KEY=your_key   (PRIMARY — Claude, recommended)\n"
        "  GEMINI_API_KEY=your_key      (SECONDARY — free tier available)\n"
        "  OPENAI_API_KEY=your_key      (TERTIARY — GPT-4o)\n"
        "Get a Claude API key at: https://console.anthropic.com/"
    )


# ---------------------------------------------------------------------------
# Report saving
# ---------------------------------------------------------------------------

def save_chart_analysis_report(
    analysis_text: str,
    image_path: str,
    ticker: str,
) -> Path:
    """
    Save the chart analysis to reports/chart-analyses/{ticker}_{timestamp}.md
    Returns the path to the saved file.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ticker = re.sub(r"[^A-Za-z0-9_\-\.]", "_", ticker)
    filename = f"{safe_ticker}_{ts}.md"
    report_path = REPORTS_DIR / filename

    header = (
        f"# LUNA Chart Analysis — {ticker}\n\n"
        f"**Chart Image:** `{image_path}`  \n"
        f"**Analyzed At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n"
        "---\n\n"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(header + analysis_text)

    logger.info("Chart analysis saved: %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# Calls-log integration
# ---------------------------------------------------------------------------

def _extract_setup_from_analysis(analysis_text: str, ticker: str) -> Optional[dict]:
    """
    Parse the analysis text to extract setup details for calls-log.json.
    Returns a dict compatible with calls_tracker.log_new_call() or None if NO TRADE.
    """
    text = analysis_text

    # Check if it's a NO TRADE signal
    if "NO TRADE" in text.upper() and "SETUP: NO TRADE" in text.upper():
        logger.info("Chart analysis output NO TRADE — not logging to calls-log.")
        return None

    # Extract direction
    direction = "neutral"
    if re.search(r"SETUP:\s*(BUY|LONG)", text, re.IGNORECASE):
        direction = "bullish"
    elif re.search(r"SETUP:\s*(SELL|SHORT)", text, re.IGNORECASE):
        direction = "bearish"
    else:
        return None  # Can't determine direction

    # Extract current price
    price_match = re.search(r"CURRENT PRICE:\s*\$?([\d,]+\.?\d*)", text, re.IGNORECASE)
    price = None
    if price_match:
        try:
            price = float(price_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Extract entry price
    entry_match = re.search(r"ENTRY:\s*\$?([\d,]+\.?\d*)", text, re.IGNORECASE)
    entry_price = price
    if entry_match:
        try:
            entry_price = float(entry_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Extract stop loss
    sl_match = re.search(r"STOP LOSS:\s*\$?([\d,]+\.?\d*)", text, re.IGNORECASE)
    stop_loss = None
    if sl_match:
        try:
            stop_loss = float(sl_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Extract TP1
    tp1_match = re.search(r"TP1:\s*\$?([\d,]+\.?\d*)", text, re.IGNORECASE)
    tp1 = None
    if tp1_match:
        try:
            tp1 = float(tp1_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Extract confidence
    conf_match = re.search(r"CONFIDENCE:\s*(\d+)/10", text, re.IGNORECASE)
    confidence = 5
    if conf_match:
        try:
            confidence = int(conf_match.group(1))
        except ValueError:
            pass

    if confidence < 6:
        logger.info("Chart confidence %d/10 < 6 — not logging to calls-log.", confidence)
        return None

    if not entry_price:
        return None

    return {
        "ticker":        ticker,
        "direction":     direction,
        "score":         confidence,
        "asset_class":   "stock",   # will be overridden if detectable
        "breakdown":     {},
        "price_at_call": entry_price,
        "support_resistance": {
            "support":    [stop_loss] if stop_loss else [],
            "resistance": [tp1] if tp1 else [],
        },
        "source":        "chart_analysis",
    }


def log_chart_setup_to_calls(analysis_text: str, ticker: str, regime: str = "UNKNOWN") -> None:
    """
    Parse analysis and log the setup to state/calls-log.json for outcome tracking.
    """
    setup = _extract_setup_from_analysis(analysis_text, ticker)
    if not setup:
        return

    # Load existing calls log
    os.makedirs(str(STATE_DIR), exist_ok=True)
    calls_data: dict = {"calls": []}
    if CALLS_LOG_PATH.exists():
        try:
            with open(CALLS_LOG_PATH, "r", encoding="utf-8") as f:
                calls_data = json.load(f)
        except Exception as exc:
            logger.warning("Could not read calls-log.json: %s", exc)

    # Avoid duplicate on same day
    today = datetime.now(timezone.utc).isoformat().split("T")[0]
    for c in calls_data.get("calls", []):
        c_date = c.get("timestamp", "").split("T")[0]
        if (
            c.get("asset") == ticker
            and c_date == today
            and c.get("direction") == setup["direction"]
            and c.get("source") == "chart_analysis"
        ):
            logger.info("Chart call for %s already logged today. Skipping.", ticker)
            return

    new_call = {
        "id":                    str(uuid.uuid4()),
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "asset":                 ticker,
        "asset_class":           setup.get("asset_class", "stock"),
        "direction":             setup["direction"],
        "score":                 setup["score"],
        "score_breakdown":       {"chart_confidence": setup["score"]},
        "price_at_call":         setup["price_at_call"],
        "key_level_support":     setup["support_resistance"]["support"][0] if setup["support_resistance"]["support"] else None,
        "key_level_resistance":  setup["support_resistance"]["resistance"][0] if setup["support_resistance"]["resistance"] else None,
        "regime_at_call":        regime,
        "source":                "chart_analysis",
        "outcome_3d":            None,
        "outcome_7d":            None,
        "price_3d":              None,
        "price_7d":              None,
        "result_3d":             None,
        "result_7d":             None,
        "checked_3d":            False,
        "checked_7d":            False,
    }

    calls_data.setdefault("calls", []).append(new_call)

    try:
        with open(CALLS_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(calls_data, f, indent=2, ensure_ascii=False)
        logger.info(
            "Chart setup logged to calls-log.json: %s %s at $%.4f",
            setup["direction"].upper(), ticker, setup["price_at_call"]
        )
    except Exception as exc:
        logger.error("Failed to save calls-log.json: %s", exc)


# ---------------------------------------------------------------------------
# Rich terminal output
# ---------------------------------------------------------------------------

def print_chart_analysis(analysis_text: str, report_path: Path) -> None:
    """Display the chart analysis in the terminal with rich formatting."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from rich.rule import Rule
        from rich import box

        console = Console()
        console.print()
        console.print(Rule("[bold magenta]🌙 LUNA CHART ANALYSIS ENGINE[/bold magenta]", style="magenta"))
        console.print()

        # Colour-code key sections
        lines = analysis_text.split("\n")
        formatted_lines: list[str] = []
        for line in lines:
            if line.strip().startswith("━"):
                formatted_lines.append(f"[dim]{line}[/dim]")
            elif line.startswith("SETUP:"):
                if "BUY" in line.upper() or "LONG" in line.upper():
                    formatted_lines.append(f"[bold green]{line}[/bold green]")
                elif "SELL" in line.upper() or "SHORT" in line.upper():
                    formatted_lines.append(f"[bold red]{line}[/bold red]")
                else:
                    formatted_lines.append(f"[bold yellow]{line}[/bold yellow]")
            elif line.startswith("CONFIDENCE:"):
                conf_match = re.search(r"(\d+)/10", line)
                if conf_match:
                    score = int(conf_match.group(1))
                    color = "green" if score >= 7 else "yellow" if score >= 5 else "red"
                    formatted_lines.append(f"[bold {color}]{line}[/bold {color}]")
                else:
                    formatted_lines.append(line)
            elif line.startswith("TREND:"):
                if "BULLISH" in line.upper():
                    formatted_lines.append(f"[green]{line}[/green]")
                elif "BEARISH" in line.upper():
                    formatted_lines.append(f"[red]{line}[/red]")
                else:
                    formatted_lines.append(f"[yellow]{line}[/yellow]")
            elif line.startswith("✅"):
                formatted_lines.append(f"[green]{line}[/green]")
            elif line.startswith("❌"):
                formatted_lines.append(f"[red]{line}[/red]")
            elif line.startswith("⚠️"):
                formatted_lines.append(f"[yellow]{line}[/yellow]")
            elif "LUNA DATA CONFIRMS" in line:
                formatted_lines.append(f"[bold green]{line}[/bold green]")
            elif "LUNA DATA CONFLICTS" in line:
                formatted_lines.append(f"[bold yellow]{line}[/bold yellow]")
            elif line.startswith("ENTRY:"):
                formatted_lines.append(f"[bold cyan]{line}[/bold cyan]")
            elif line.startswith("STOP LOSS:"):
                formatted_lines.append(f"[bold red]{line}[/bold red]")
            elif line.startswith("TP"):
                formatted_lines.append(f"[bold green]{line}[/bold green]")
            elif line.startswith("OVERALL R:R:"):
                formatted_lines.append(f"[bold magenta]{line}[/bold magenta]")
            elif line.startswith("ASSET:") or line.startswith("TIMEFRAME:") or line.startswith("CURRENT PRICE:"):
                formatted_lines.append(f"[bold white]{line}[/bold white]")
            else:
                formatted_lines.append(line)

        console.print("\n".join(formatted_lines))
        console.print()
        console.print(Rule(style="dim"))
        console.print(
            f"[dim]📁 Report saved: [bold]{report_path}[/bold][/dim]"
        )
        console.print()

    except ImportError:
        # Fallback to plain print if rich not available
        print("\n" + "=" * 60)
        print("🌙 LUNA CHART ANALYSIS")
        print("=" * 60)
        print(analysis_text)
        print(f"\nReport saved: {report_path}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Main orchestrator for --chart mode
# ---------------------------------------------------------------------------

def run_chart_mode(image_path: str, config: dict) -> None:
    """
    Full chart analysis pipeline:
      1. Validate image path
      2. Run analysis via vision AI
      3. Save report to reports/chart-analyses/
      4. Print rich terminal output
      5. Log setup to calls-log.json
    """
    logger.info("═" * 60)
    logger.info("CHART ANALYSIS MODE — %s", image_path)
    logger.info("═" * 60)

    # Validate path
    if not Path(image_path).exists():
        logger.error("Image file not found: %s", image_path)
        print(f"\n❌ Error: Image file not found: {image_path}")
        sys.exit(1)

    ticker = _infer_asset_from_path(image_path)

    try:
        # Run analysis
        logger.info("Running chart analysis for %s...", ticker)
        analysis_text = run_chart_analysis(image_path, config)

        # Save report
        report_path = save_chart_analysis_report(analysis_text, image_path, ticker)

        # Print to terminal
        print_chart_analysis(analysis_text, report_path)

        # Log to calls-log.json
        macro_regime = "UNKNOWN"
        try:
            if LAST_RUN_PATH.exists():
                with open(LAST_RUN_PATH, "r", encoding="utf-8") as f:
                    state = json.load(f)
                macro_regime = state.get("macro_state", {}).get("regime", "UNKNOWN")
        except Exception:
            pass

        log_chart_setup_to_calls(analysis_text, ticker, macro_regime)

        logger.info("Chart analysis complete for %s.", ticker)

    except RuntimeError as exc:
        logger.error("Chart analysis failed: %s", exc)
        print(f"\n❌ {exc}\n")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error in chart analysis: %s", exc)
        sys.exit(1)
