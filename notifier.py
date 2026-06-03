"""
notifier.py
-----------
Handles all notifications for the trading research agent.
Supports Telegram (via requests POST to Bot API) and ANSI-colored console output.
"""

import logging
import sys
import textwrap
from datetime import datetime
from typing import Optional

import requests

# Reconfigure stdout/stderr to UTF-8 for reliable console print on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSI Color Codes
# ---------------------------------------------------------------------------

class _C:
    RESET       = "\033[0m"
    BOLD        = "\033[1m"
    DIM         = "\033[2m"

    # Regime colors
    RISK_ON     = "\033[92m"   # bright green
    RISK_OFF    = "\033[91m"   # bright red
    TRANSITION  = "\033[93m"   # yellow

    # Price direction
    POSITIVE    = "\033[92m"   # bright green
    NEGATIVE    = "\033[91m"   # bright red

    # Alert
    ALERT       = "\033[33m"   # orange / dark yellow
    ALERT_HIGH  = "\033[91m"   # bright red
    ALERT_MED   = "\033[93m"   # yellow
    ALERT_LOW   = "\033[36m"   # cyan

    # General
    HEADER      = "\033[95m"   # magenta
    INFO        = "\033[94m"   # blue
    WHITE       = "\033[97m"   # bright white
    GREY        = "\033[90m"   # dark grey


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _regime_color(regime: str) -> str:
    """Return ANSI color code for a given macro regime string."""
    r = regime.upper()
    if "RISK-ON" in r or "RISK_ON" in r:
        return _C.RISK_ON
    if "RISK-OFF" in r or "RISK_OFF" in r:
        return _C.RISK_OFF
    return _C.TRANSITION


def _pct_color(value: float) -> str:
    """Return ANSI color for a percentage value."""
    return _C.POSITIVE if value >= 0 else _C.NEGATIVE


def _pct_str(value: float, decimals: int = 2) -> str:
    """Format a percentage with sign and color."""
    sign = "+" if value >= 0 else ""
    color = _pct_color(value)
    return f"{color}{sign}{value:.{decimals}f}%{_C.RESET}"


def _severity_color(severity: str) -> str:
    s = severity.upper()
    if s == "HIGH":
        return _C.ALERT_HIGH
    if s == "MEDIUM":
        return _C.ALERT_MED
    return _C.ALERT_LOW


def _truncate(text: str, max_len: int = 4096) -> str:
    """Truncate text to max_len characters, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    cutoff = max_len - 30
    return text[:cutoff] + "\n\n… _(message truncated)_"


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (for Telegram messages)."""
    import re
    ansi_escape = re.compile(r'\033\[[0-9;]*m')
    return ansi_escape.sub('', text)


# ---------------------------------------------------------------------------
# Priority Levels & Deduplication Helpers
# ---------------------------------------------------------------------------
import json
import time
from pathlib import Path

ALERTS_SENT_PATH = Path("state/alerts-sent.json")

def should_suppress_alert(alert: dict, record: bool = True) -> bool:
    """
    Check if the alert is a duplicate sent within the last 4 hours.
    Auto-cleans alerts older than 24 hours on start.
    """
    try:
        ALERTS_SENT_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing alerts
        sent_alerts = {}
        if ALERTS_SENT_PATH.exists():
            try:
                with open(ALERTS_SENT_PATH, "r", encoding="utf-8") as f:
                    sent_alerts = json.load(f)
            except Exception:
                pass
                
        now = time.time()
        
        # Clean up alerts older than 24 hours (86400 seconds)
        cleaned_alerts = {}
        for key, timestamp in sent_alerts.items():
            if now - timestamp < 86400:
                cleaned_alerts[key] = timestamp
                
        # Generate unique key for current alert
        asset = alert.get("asset", "GLOBAL")
        message = alert.get("message") or alert.get("reason", "")
        # Use first 50 chars of message for key stability
        msg_stub = message.strip()[:50]
        alert_key = f"{asset}:{msg_stub}"
        
        # Check if already sent in last 4 hours (14400 seconds)
        is_duplicate = False
        if alert_key in cleaned_alerts:
            last_sent = cleaned_alerts[alert_key]
            if now - last_sent < 14400:
                is_duplicate = True
                
        # Update timestamp if we are sending it
        if not is_duplicate and record:
            cleaned_alerts[alert_key] = now
            # Save state
            try:
                with open(ALERTS_SENT_PATH, "w", encoding="utf-8") as f:
                    json.dump(cleaned_alerts, f, indent=2)
            except Exception:
                pass
            
        return is_duplicate
    except Exception as e:
        logger.error("Deduplication error: %s", e)
        return False


def get_priority_details(alert: dict) -> tuple[str, str, str]:
    """
    Returns (priority_label, emoji, color_code).
    Priority levels:
      - P0 CRITICAL (SL/TP hit, VIX > 35, circuit breakers)
      - P1 HIGH (>5% move, RSI cross, Golden/Death cross)
      - P2 MEDIUM (>3% move, setup score >= 7, correlation anomaly)
      - P3 LOW (daily brief, events, weight changes)
    """
    severity = str(alert.get("severity", "")).upper()
    priority = str(alert.get("priority", "")).upper()
    message = str(alert.get("message") or alert.get("reason", "")).upper()
    
    # 1. P0 checks
    if (priority == "P0" or severity == "CRITICAL" or "STOP LOSS" in message or 
        "TAKE PROFIT" in message or "VIX > 35" in message or "CIRCUIT BREAKER" in message):
        return "P0 CRITICAL", "🚨", _C.ALERT_HIGH
        
    # 2. P1 checks
    if (priority == "P1" or severity == "HIGH" or ">5%" in message or "RSI CROSS" in message or 
        "GOLDEN CROSS" in message or "DEATH CROSS" in message):
        return "P1 HIGH", "⚠️", _C.ALERT_HIGH
        
    # 3. P2 checks
    if (priority == "P2" or severity == "MEDIUM" or ">3%" in message or "SCORE" in message or 
        "CORRELATION" in message or "ANOMALY" in message):
        return "P2 MEDIUM", "⚡", _C.ALERT_MED
        
    # 4. P3/LOW checks (default)
    return "P3 LOW", "ℹ️", _C.ALERT_LOW



# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram(message: str, bot_token: str, chat_id: str) -> bool:
    """
    Send a Markdown-formatted message to a Telegram chat via the Bot API.

    Parameters
    ----------
    message   : Text in Telegram Markdown format (use *bold*, _italic_, `code`).
    bot_token : Telegram Bot API token.
    chat_id   : Target chat / channel ID.

    Returns
    -------
    bool: True if the message was sent successfully, False otherwise.
    """
    if not bot_token or not chat_id:
        logger.warning("Telegram credentials not provided — skipping Telegram notification.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": _truncate(message, max_len=4096),
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        if not result.get("ok"):
            logger.error(
                "Telegram API returned ok=false: %s", result.get("description", "unknown error")
            )
            return False
        logger.debug("Telegram message sent successfully (chat_id=%s).", chat_id)
        return True

    except requests.exceptions.ConnectionError as exc:
        logger.error("Telegram connection error: %s", exc)
    except requests.exceptions.Timeout:
        logger.error("Telegram request timed out after 15 seconds.")
    except requests.exceptions.HTTPError as exc:
        logger.error("Telegram HTTP error: %s — response: %s", exc, response.text[:200])
    except requests.exceptions.RequestException as exc:
        logger.error("Telegram request failed: %s", exc)
    except Exception as exc:
        logger.exception("Unexpected error sending Telegram message: %s", exc)

    return False


# ---------------------------------------------------------------------------
# Console formatters
# ---------------------------------------------------------------------------

def format_console_brief(macro_state: dict, opportunities: list, alerts: list) -> str:
    """
    Build a one-paragraph ANSI-colored console summary.

    Parameters
    ----------
    macro_state   : Output of macro.analyze_macro() — expects keys like 'regime',
                    'top_mover', 'top_mover_pct', 'key_risk', etc.
    opportunities : Ranked list of opportunity dicts from scanner.rank_opportunities().
    alerts        : List of active alert dicts.

    Returns
    -------
    str: Formatted, colorized single-paragraph summary string.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Regime ---
    regime = macro_state.get("regime", "UNKNOWN")
    rc = _regime_color(regime)
    regime_str = f"{rc}{_C.BOLD}{regime}{_C.RESET}"

    # --- Top mover ---
    top_mover = macro_state.get("top_mover", "N/A")
    top_mover_pct = macro_state.get("top_mover_pct", 0.0)
    mover_str = f"{_C.WHITE}{top_mover}{_C.RESET} {_pct_str(top_mover_pct)}"

    # --- Top opportunity ---
    if opportunities:
        top_opp = opportunities[0]
        opp_asset = top_opp.get("asset", "N/A")
        opp_score = top_opp.get("score", 0.0)
        opp_bias = top_opp.get("bias", "NEUTRAL")
        opp_str = (
            f"{_C.POSITIVE if opp_bias.upper() == 'BULLISH' else _C.NEGATIVE}"
            f"{opp_asset}{_C.RESET} (score={opp_score:.1f}, bias={opp_bias})"
        )
    else:
        opp_str = f"{_C.GREY}None identified{_C.RESET}"

    # --- Key risk ---
    key_risk = macro_state.get("key_risk", "No significant risks flagged")
    risk_str = f"{_C.ALERT}{key_risk}{_C.RESET}"

    # --- Alert count ---
    high_alerts = [a for a in alerts if a.get("severity", "").upper() == "HIGH"]
    alert_note = ""
    if high_alerts:
        alert_note = (
            f"  {_C.ALERT_HIGH}{_C.BOLD}⚠ {len(high_alerts)} HIGH ALERT(S) ACTIVE!{_C.RESET}"
        )
    elif alerts:
        alert_note = f"  {_C.ALERT_MED}{len(alerts)} alert(s) pending.{_C.RESET}"

    lines = [
        f"{_C.HEADER}{_C.BOLD}{'─' * 70}{_C.RESET}",
        f"  {_C.GREY}[{ts}]{_C.RESET}  {_C.BOLD}LUNS — CYCLE COMPLETE{_C.RESET}",
        f"{_C.HEADER}{'─' * 70}{_C.RESET}",
        (
            f"  Regime: {regime_str}  │  "
            f"Top Mover: {mover_str}  │  "
            f"Top Opp: {opp_str}"
        ),
        f"  Key Risk: {risk_str}",
    ]
    if alert_note:
        lines.append(alert_note)
    lines.append(f"{_C.HEADER}{'─' * 70}{_C.RESET}")

    return "\n".join(lines)


def format_alert_message(alert: dict) -> str:
    """
    Format a single alert dict into a colored console string with prioritization.
    """
    asset    = alert.get("asset", "UNKNOWN")
    reason   = alert.get("message") or alert.get("reason", "No reason provided")
    context  = alert.get("context", "")
    atype    = alert.get("type", "")

    label, emoji, color = get_priority_details(alert)

    header = f"{color}{_C.BOLD}{emoji}  ALERT [{label}]: {asset}{_C.RESET} — {reason}"

    lines = [header]

    if atype:
        lines.append(f"   {_C.GREY}Type: {atype}{_C.RESET}")

    if context:
        # Wrap context to 2 visible lines max
        ctx_lines = str(context).strip().splitlines()[:2]
        for cl in ctx_lines:
            lines.append(f"   {_C.DIM}{cl}{_C.RESET}")

    return "\n".join(lines)


def format_daily_summary(
    market_data: dict,
    opportunities: list,
    macro_state: dict,
    calendar: list,
) -> str:
    """
    Build a full daily digest string for console output.

    Parameters
    ----------
    market_data   : Dict keyed by asset symbol, each value is a dict with
                    price, pct_change, volume, etc.
    opportunities : Ranked opportunity list.
    macro_state   : Dict from macro.analyze_macro().
    calendar      : List of upcoming economic calendar events (dicts with
                    'time', 'event', 'impact' keys).

    Returns
    -------
    str: Multi-section colored daily digest.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    sep = f"{_C.HEADER}{_C.BOLD}{'═' * 70}{_C.RESET}"
    thin = f"{_C.GREY}{'─' * 70}{_C.RESET}"

    # Header
    lines.append(sep)
    lines.append(
        f"  {_C.BOLD}{_C.WHITE}📊 LUNS — DAILY BRIEF  │  {ts}{_C.RESET}"
    )
    lines.append(sep)

    # ---- Macro Regime ----
    regime = macro_state.get("regime", "UNKNOWN")
    rc = _regime_color(regime)
    vix    = macro_state.get("vix", {})
    dxy    = macro_state.get("dxy", {})
    yields = macro_state.get("yields", {})

    lines.append(f"\n  {_C.BOLD}MACRO REGIME{_C.RESET}")
    lines.append(thin)
    lines.append(f"  Regime   : {rc}{_C.BOLD}{regime}{_C.RESET}")

    vix_val = vix.get("value", "N/A")
    vix_chg = vix.get("pct_change", None)
    vix_str = f"VIX {vix_val}"
    if vix_chg is not None:
        vix_str += f" ({_pct_str(vix_chg)})"
    lines.append(f"  Risk Idx : {vix_str}")

    dxy_val = dxy.get("value", "N/A")
    dxy_chg = dxy.get("pct_change", None)
    dxy_str = f"DXY {dxy_val}"
    if dxy_chg is not None:
        dxy_str += f" ({_pct_str(dxy_chg)})"
    lines.append(f"  DXY      : {dxy_str}")

    us10y = yields.get("US10Y", "N/A")
    lines.append(f"  US10Y    : {us10y}")

    key_risk = macro_state.get("key_risk", "None flagged")
    lines.append(f"  Key Risk : {_C.ALERT}{key_risk}{_C.RESET}")

    # ---- Overnight Movers ----
    lines.append(f"\n  {_C.BOLD}OVERNIGHT MOVERS{_C.RESET}")
    lines.append(thin)

    if market_data:
        # Sort by absolute pct_change descending
        sorted_assets = sorted(
            market_data.items(),
            key=lambda kv: abs(kv[1].get("pct_change", 0.0)),
            reverse=True,
        )
        shown = 0
        for symbol, data in sorted_assets:
            if shown >= 8:
                break
            price   = data.get("price", "N/A")
            pct_chg = data.get("pct_change", 0.0)
            vol     = data.get("volume", None)
            line = (
                f"  {_C.WHITE}{symbol:<12}{_C.RESET}"
                f"  {_C.BOLD}{price:>10}{_C.RESET}"
                f"  {_pct_str(pct_chg):>20}"
            )
            if vol is not None:
                line += f"  {_C.GREY}Vol: {vol}{_C.RESET}"
            lines.append(line)
            shown += 1
    else:
        lines.append(f"  {_C.GREY}No market data available.{_C.RESET}")

    # ---- Watchlist Highlights / Opportunities ----
    lines.append(f"\n  {_C.BOLD}TODAY'S WATCHLIST HIGHLIGHTS{_C.RESET}")
    lines.append(thin)

    if opportunities:
        for i, opp in enumerate(opportunities[:5], start=1):
            asset  = opp.get("asset", "N/A")
            score  = opp.get("score", 0.0)
            bias   = opp.get("bias", "NEUTRAL")
            reason = opp.get("reason", opp.get("summary", ""))
            bc = _C.POSITIVE if bias.upper() == "BULLISH" else (
                 _C.NEGATIVE if bias.upper() == "BEARISH" else _C.GREY)
            lines.append(
                f"  {i}. {_C.WHITE}{asset:<12}{_C.RESET}"
                f"  {bc}{bias:<8}{_C.RESET}"
                f"  score={score:.1f}"
                + (f"  {_C.DIM}{reason[:50]}{_C.RESET}" if reason else "")
            )
    else:
        lines.append(f"  {_C.GREY}No ranked opportunities at this time.{_C.RESET}")

    # ---- Upcoming Events ----
    lines.append(f"\n  {_C.BOLD}UPCOMING ECONOMIC EVENTS{_C.RESET}")
    lines.append(thin)

    if calendar:
        for event in calendar[:6]:
            ev_time   = event.get("time", "TBD")
            ev_name   = event.get("event", "Unknown event")
            ev_impact = event.get("impact", "").upper()
            impact_c  = (
                _C.ALERT_HIGH if ev_impact == "HIGH" else
                _C.ALERT_MED  if ev_impact == "MEDIUM" else
                _C.GREY
            )
            lines.append(
                f"  {_C.GREY}{ev_time:<12}{_C.RESET}"
                f"  {ev_name:<40}"
                f"  {impact_c}[{ev_impact}]{_C.RESET}"
            )
    else:
        lines.append(f"  {_C.GREY}No upcoming events in calendar.{_C.RESET}")

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Telegram message formatters (Markdown, no ANSI)
# ---------------------------------------------------------------------------

def _telegram_brief(macro_state: dict, opportunities: list, alerts: list) -> str:
    """Build a Telegram Markdown brief from cycle results."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    regime = macro_state.get("regime", "UNKNOWN")
    top_mover = macro_state.get("top_mover", "N/A")
    top_mover_pct = macro_state.get("top_mover_pct", 0.0)
    key_risk = macro_state.get("key_risk", "None flagged")

    sign = "+" if top_mover_pct >= 0 else ""
    mover_str = f"{top_mover} {sign}{top_mover_pct:.2f}%"

    opp_line = "None"
    if opportunities:
        top = opportunities[0]
        opp_line = (
            f"{top.get('asset','N/A')} "
            f"({top.get('bias','NEUTRAL')}, score={top.get('score',0):.1f})"
        )

    alert_section = ""
    if alerts:
        high = [a for a in alerts if a.get("severity","").upper() == "HIGH"]
        alert_section = f"\n⚠️ *Alerts:* {len(alerts)} total"
        if high:
            alert_section += f", *{len(high)} HIGH*"

    lines = [
        f"📊 *LUNS — CYCLE COMPLETE*",
        f"_{ts}_",
        "",
        f"🌐 *Regime:* `{regime}`",
        f"📈 *Top Mover:* `{mover_str}`",
        f"🎯 *Top Opportunity:* {opp_line}",
        f"⚡ *Key Risk:* {key_risk}",
    ]
    if alert_section:
        lines.append(alert_section)

    return "\n".join(lines)


def _telegram_alert(alert: dict) -> str:
    """Build a Telegram Markdown alert message with priority templates."""
    asset    = alert.get("asset", "UNKNOWN")
    reason   = alert.get("message") or alert.get("reason", "No reason provided")
    atype    = alert.get("type", "")
    context  = alert.get("context", "")

    label, emoji, _ = get_priority_details(alert)

    lines = [
        f"{emoji} *ALERT [{label}]: {asset}*",
        f"_{reason}_",
    ]
    if atype:
        lines.append(f"Type: `{atype}`")
    if context:
        ctx_lines = str(context).strip().splitlines()[:2]
        for cl in ctx_lines:
            lines.append(cl)

    return "\n".join(lines)


def _telegram_daily_summary(
    market_data: dict,
    opportunities: list,
    macro_state: dict,
    calendar: list,
) -> str:
    """Build a Telegram Markdown daily digest."""
    ts = datetime.now().strftime("%Y-%m-%d")

    regime   = macro_state.get("regime", "UNKNOWN")
    key_risk = macro_state.get("key_risk", "None flagged")
    vix      = macro_state.get("vix", {})
    dxy      = macro_state.get("dxy", {})
    yields   = macro_state.get("yields", {})

    vix_val = vix.get("value", "N/A")
    dxy_val = dxy.get("value", "N/A")
    us10y   = yields.get("US10Y", "N/A")

    lines = [
        f"📊 *LUNS — DAILY BRIEF*",
        f"_{ts}_",
        "",
        f"🌐 *Regime:* `{regime}`",
        f"📉 VIX: `{vix_val}`  │  DXY: `{dxy_val}`  │  US10Y: `{us10y}`",
        f"⚡ *Key Risk:* {key_risk}",
        "",
        "📈 *Top Movers:*",
    ]

    if market_data:
        sorted_assets = sorted(
            market_data.items(),
            key=lambda kv: abs(kv[1].get("pct_change", 0.0)),
            reverse=True,
        )
        for symbol, data in sorted_assets[:5]:
            pct = data.get("pct_change", 0.0)
            price = data.get("price", "N/A")
            sign = "+" if pct >= 0 else ""
            arrow = "🟢" if pct >= 0 else "🔴"
            lines.append(f"  {arrow} `{symbol}` {price}  ({sign}{pct:.2f}%)")
    else:
        lines.append("  _No data available_")

    lines += ["", "🎯 *Watchlist Opportunities:*"]
    if opportunities:
        for i, opp in enumerate(opportunities[:5], 1):
            asset  = opp.get("asset", "N/A")
            score  = opp.get("score", 0.0)
            bias   = opp.get("bias", "NEUTRAL")
            reason = opp.get("reason", opp.get("summary", ""))[:40]
            lines.append(f"  {i}. *{asset}* — {bias} (score={score:.1f})  _{reason}_")
    else:
        lines.append("  _None at this time_")

    lines += ["", "📅 *Upcoming Events:*"]
    if calendar:
        for event in calendar[:5]:
            ev_time   = event.get("time", "TBD")
            ev_name   = event.get("event", "Unknown")
            ev_impact = event.get("impact", "")
            impact_emoji = "🔴" if ev_impact.upper() == "HIGH" else (
                           "🟡" if ev_impact.upper() == "MEDIUM" else "⚪")
            lines.append(f"  {impact_emoji} `{ev_time}` — {ev_name}")
    else:
        lines.append("  _No events in calendar_")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bloomberg Terminal CLI Renderer (Primary console output using rich)
# ---------------------------------------------------------------------------

def render_bloomberg_terminal(macro_state: dict, opportunities: list, alerts: list, config: dict, portfolio_status: dict | None = None) -> None:
    """
    Render a premium Bloomberg Terminal-style CLI dashboard using the rich library.
    Inspired by modern dark financial dashboard design with structured panels,
    colored stat badges, two-column layouts, and glowing accent borders.
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich.align import Align
        from rich.columns import Columns
        from rich.rule import Rule
        from rich.padding import Padding
        import json
        import os
        from datetime import datetime
        import numpy as np

        console = Console()
        now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        WIDTH = console.width or 120

        # ── MASTER HEADER ──────────────────────────────────────────────────────
        console.print()
        header = Text()
        header.append("  ██╗     ██╗   ██╗███╗   ██╗ █████╗  ", style="bold bright_blue")
        console.print(Align.center(header))
        header2 = Text()
        header2.append("  ██║     ██║   ██║████╗  ██║██╔══██╗ ", style="bold bright_blue")
        console.print(Align.center(header2))
        header3 = Text()
        header3.append("  ██║     ██║   ██║██╔██╗ ██║███████║ ", style="bold bright_blue")
        console.print(Align.center(header3))
        header4 = Text()
        header4.append("  ██║     ██║   ██║██║╚██╗██║██╔══██║ ", style="bold cyan")
        console.print(Align.center(header4))
        header5 = Text()
        header5.append("  ███████╗╚██████╔╝██║ ╚████║██║  ██║ ", style="bold cyan")
        console.print(Align.center(header5))
        header6 = Text()
        header6.append("  ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝ ", style="bold cyan")
        console.print(Align.center(header6))
        console.print()

        subtitle = Text(justify="center")
        subtitle.append("  INTERMARKET INTELLIGENCE & PORTFOLIO SUITE  ", style="bold white on dark_blue")
        subtitle.append(f"   {now_str} IST  ", style="dim white")
        console.print(Align.center(subtitle))
        console.print(Rule(style="bright_blue"))
        console.print()

        # ── SECTION 1: MACRO REGIME ─────────────────────────────────────────
        regime = macro_state.get("regime", "UNKNOWN").upper()
        regime_color = "yellow"
        regime_icon = "🟡"
        if "RISK-ON" in regime or "RISK_ON" in regime:
            regime_color = "bright_green"
            regime_icon = "🟢"
        elif "RISK-OFF" in regime or "RISK_OFF" in regime:
            regime_color = "bright_red"
            regime_icon = "🔴"

        risk_score = macro_state.get("risk_score") or macro_state.get("risk_on_score", 0.0)
        vix_val    = macro_state.get("vix")
        dxy_val    = macro_state.get("dxy")
        rates_trend = macro_state.get("rates_trend", "flat")

        vix_is_stale = macro_state.get("vix_is_stale", False)
        vix_str = f"{vix_val:.2f}" if (vix_val is not None and not np.isnan(vix_val)) else "N/A"
        if vix_is_stale and vix_str != "N/A":
            vix_str += " [STALE]"
        dxy_str = f"{dxy_val:.2f}" if (dxy_val is not None and not np.isnan(dxy_val)) else "N/A"
        vix_color = "bold bright_red" if (vix_val or 0) > 30 else ("bold yellow" if (vix_val or 0) > 20 else "bold bright_green")
        rates_color = "bright_red" if rates_trend == "rising" else ("bright_green" if rates_trend == "falling" else "yellow")

        # Regime badge
        regime_badge = Text()
        regime_badge.append(f" {regime_icon} {regime} ", style=f"bold {regime_color} on grey11")

        # Build stat pills row
        macro_grid = Table.grid(expand=True, padding=(0, 2))
        macro_grid.add_column(ratio=2)
        macro_grid.add_column(ratio=1, justify="center")
        macro_grid.add_column(ratio=1, justify="center")
        macro_grid.add_column(ratio=1, justify="center")
        macro_grid.add_column(ratio=1, justify="center")

        regime_cell = Text()
        regime_cell.append("REGIME\n", style="dim white")
        regime_cell.append(f" {regime_icon} {regime} ", style=f"bold {regime_color}")

        score_cell = Text(justify="center")
        score_cell.append("RISK SCORE\n", style="dim white")
        score_cell.append(f" {risk_score:.1f}/10 " if risk_score is not None else " N/A ", style="bold yellow")

        vix_cell = Text(justify="center")
        vix_cell.append("VIX\n", style="dim white")
        vix_cell.append(f" {vix_str} ", style=vix_color)

        dxy_cell = Text(justify="center")
        dxy_cell.append("DXY\n", style="dim white")
        dxy_cell.append(f" {dxy_str} ", style="bold cyan")

        rates_cell = Text(justify="center")
        rates_cell.append("RATES TREND\n", style="dim white")
        rates_cell.append(f" {rates_trend.upper()} ", style=rates_color)

        macro_grid.add_row(regime_cell, score_cell, vix_cell, dxy_cell, rates_cell)

        console.print(Panel(
            macro_grid,
            title="[bold cyan]🌐  GLOBAL MACRO REGIME[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        ))

        # VIX high-volatility critical banner
        if vix_val is not None and not np.isnan(vix_val) and vix_val > 30:
            warn = Text(justify="center")
            warn.append(f"  ⚠  HIGH VOLATILITY REGIME — VIX {vix_val:.2f}  ⚠  REDUCE POSITION SIZES & INCREASE HEDGE RATIOS  ", style="bold white on red")
            console.print(Align.center(warn))
            console.print()

        console.print()

        # ── SECTION 2: TWO-COLUMN — OPPORTUNITIES + MOVERS ─────────────────
        console.print(Rule("[bold white]  MARKET INTELLIGENCE  [/bold white]", style="bright_blue"))
        console.print()

        # Opportunities table
        opps_table = Table(
            title="[bold bright_green]🎯  TOP SCORING OPPORTUNITIES[/bold bright_green]",
            border_style="dark_green",
            header_style="bold bright_green",
            expand=True,
            show_lines=True,
        )
        opps_table.add_column("#", style="dim white", width=3, justify="center")
        opps_table.add_column("Asset", style="bold white", min_width=8)
        opps_table.add_column("Score", justify="center", min_width=7)
        opps_table.add_column("Bias", justify="center", min_width=9)
        opps_table.add_column("Setup & Invalidation", min_width=30)

        valid_opps = sorted(opportunities, key=lambda x: abs(x.get("score", 0.0)), reverse=True)
        for i, opp in enumerate(valid_opps[:5], 1):
            score = opp.get("score", 0.0)
            score_color = "bright_green" if score >= 6 else ("bright_red" if score <= -6 else "yellow")
            bias = str(opp.get("bias", "NEUTRAL")).upper()
            bias_color = "bright_green" if bias == "BULLISH" else ("bright_red" if bias == "BEARISH" else "yellow")
            bias_bg    = "on dark_green" if bias == "BULLISH" else ("on dark_red" if bias == "BEARISH" else "on dark_orange3")
            reasoning  = opp.get("reasoning") or opp.get("reason") or "—"
            invalid    = opp.get("invalidation_level")
            invalid_str = f"\n[dim]⛔ Invalidation: {invalid}[/dim]" if invalid else ""

            opps_table.add_row(
                f"[dim]{i}[/dim]",
                f"[bold white]{opp.get('ticker', '?')}[/bold white]",
                f"[{score_color}]{score:+.1f}[/{score_color}]",
                f"[bold {bias_color} {bias_bg}] {bias} [/bold {bias_color} {bias_bg}]",
                f"{reasoning}{invalid_str}"
            )

        if not valid_opps:
            opps_table.add_row("—", "—", "—", "—", "[dim]No setups qualified this cycle.[/dim]")

        # Movers table
        gainers, losers = [], []
        if os.path.exists("state/last-run.json"):
            try:
                with open("state/last-run.json", "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                    movers  = state_data.get("market_data", {}).get("top_movers", {})
                    gainers = movers.get("gainers", [])
                    losers  = movers.get("losers", [])
            except Exception:
                pass

        movers_table = Table(
            title="[bold yellow]⚡  TOP MOVERS (24H)[/bold yellow]",
            border_style="dark_orange3",
            header_style="bold yellow",
            expand=True,
            show_lines=True,
        )
        movers_table.add_column("Ticker", style="bold white", min_width=10)
        movers_table.add_column("Direction", justify="center", min_width=9)
        movers_table.add_column("Change", justify="right", min_width=10)
        movers_table.add_column("Bar", min_width=12)

        has_movers = False
        for g in gainers[:3]:
            chg = g.get("change_24h")
            chg_str = f"+{chg:.2f}%" if (chg is not None and not (isinstance(chg, float) and np.isnan(chg))) else "N/A"
            bar_len = min(int(abs(chg or 0) * 1.5), 12)
            bar = "[bright_green]" + "█" * bar_len + "[/bright_green]"
            movers_table.add_row(
                g.get("ticker", "?"),
                "[bold bright_green on dark_green] ▲ GAIN [/bold bright_green on dark_green]",
                f"[bright_green]{chg_str}[/bright_green]",
                bar
            )
            has_movers = True

        for l in losers[:3]:
            chg = l.get("change_24h")
            chg_str = f"{chg:.2f}%" if (chg is not None and not (isinstance(chg, float) and np.isnan(chg))) else "N/A"
            bar_len = min(int(abs(chg or 0) * 1.5), 12)
            bar = "[bright_red]" + "█" * bar_len + "[/bright_red]"
            movers_table.add_row(
                l.get("ticker", "?"),
                "[bold bright_red on dark_red] ▼ LOSS [/bold bright_red on dark_red]",
                f"[bright_red]{chg_str}[/bright_red]",
                bar
            )
            has_movers = True

        if not has_movers:
            movers_table.add_row("—", "—", "[dim]No movers this cycle[/dim]", "")

        console.print(Columns([opps_table, movers_table], expand=True, equal=True))
        console.print()

        # ── SECTION 3: PORTFOLIO SUMMARY ────────────────────────────────────
        p_stats = None
        portfolio_is_empty = False
        if portfolio_status is not None:
            p_stats = portfolio_status
        elif os.path.exists("portfolio.json"):
            try:
                with open("portfolio.json", "r", encoding="utf-8") as f:
                    port_data = json.load(f)
                open_pos = [p for p in port_data.get("positions", []) if p.get("status", "open").lower() == "open"]
                if open_pos:
                    from portfolio import calculate_portfolio_status
                    p_stats = calculate_portfolio_status(silent=True)
                else:
                    portfolio_is_empty = True
            except Exception:
                p_stats = None

        console.print(Rule("[bold white]  PORTFOLIO INTELLIGENCE  [/bold white]", style="magenta"))
        console.print()

        if portfolio_is_empty and p_stats is None:
            empty_text = Text(justify="center")
            empty_text.append(
                "  No positions tracked yet. Add your positions to portfolio.json to begin tracking.  ",
                style="bold yellow"
            )
            console.print(Panel(
                Align.center(empty_text),
                title="[bold magenta]💼  INVESTMENT PORTFOLIO SUMMARY[/bold magenta]",
                border_style="magenta",
                padding=(1, 3)
            ))
            console.print()
        elif p_stats:
            total_val = p_stats.get("total_portfolio_value", 0.0)
            cash      = p_stats.get("cash_balance", 0.0)
            pnl       = p_stats.get("total_unrealized_pnl", 0.0)
            pnl_pct   = p_stats.get("unrealized_pnl_pct", 0.0)
            heat      = p_stats.get("portfolio_heat", 0.0)
            win_rate  = p_stats.get("win_rate", 0.0)
            avg_rr    = p_stats.get("avg_rr", 0.0)

            pnl_color = "bright_green" if pnl >= 0 else "bright_red"
            pnl_sign  = "+" if pnl >= 0 else ""
            heat_color = "bright_green" if heat < 10 else ("yellow" if heat <= 20 else "bright_red")
            heat_icon  = "🟢" if heat < 10 else ("🟡" if heat <= 20 else "🔴")

            # Stat tiles — two columns of two metrics each
            left_col = Table.grid(expand=True, padding=(0, 2))
            left_col.add_column(ratio=1)
            left_col.add_column(ratio=1)

            def stat_tile(label: str, value: str, value_style: str, detail: str = "") -> Text:
                t = Text()
                t.append(f"{label}\n", style="dim white")
                t.append(value, style=f"bold {value_style}")
                if detail:
                    t.append(f"\n{detail}", style="dim")
                return t

            left_col.add_row(
                stat_tile("TOTAL VALUE", f"${total_val:,.2f}", "bright_white", "Cash + Open Holdings"),
                stat_tile("CASH BALANCE", f"${cash:,.2f}", "cyan", "Available Buying Power"),
            )
            left_col.add_row(
                stat_tile("UNREALISED P&L", f"{pnl_sign}${pnl:,.2f}  ({pnl_sign}{pnl_pct:.2f}%)", pnl_color, "Open position return"),
                stat_tile(f"PORTFOLIO HEAT  {heat_icon}", f"{heat:.2f}%", heat_color, "Capital at risk"),
            )
            left_col.add_row(
                stat_tile("WIN RATE", f"{win_rate:.1f}%", "bright_white", "Closed trades"),
                stat_tile("AVG RISK:REWARD", f"{avg_rr:.2f}:1", "cyan", "Open positions"),
            )

            console.print(Panel(
                left_col,
                title="[bold magenta]💼  INVESTMENT PORTFOLIO SUMMARY[/bold magenta]",
                border_style="magenta",
                padding=(1, 3)
            ))
            console.print()

        # ── SECTION 4: CORRELATION ANOMALIES ───────────────────────────────
        correlations_data = macro_state.get("correlations", {})
        pairs_data        = correlations_data.get("pairs", {})
        anomaly_rows      = []

        nice_names = {
            "GC=F_DX-Y.NYB":  "Gold vs DXY",
            "^TNX_^GSPC":     "10Y Yield vs S&P 500",
            "CL=F_^GSPC":     "Oil vs S&P 500",
            "BTC_^NDX":       "Bitcoin vs NASDAQ",
            "GC=F_^VIX":      "Gold vs VIX",
            "HYG_^GSPC":      "HY Bonds vs S&P 500",
            "EURUSD=X_GC=F":  "EUR/USD vs Gold",
            "^TNX_DX-Y.NYB":  "10Y Yield vs DXY",
            "CL=F_USDINR=X":  "Oil vs USD/INR"
        }

        for pk, pinfo in pairs_data.items():
            if pinfo.get("anomaly", False):
                anomaly_rows.append((pk, pinfo))

        console.print(Rule("[bold white]  CROSS-ASSET CORRELATION RADAR  [/bold white]", style="red"))
        console.print()

        corr_table = Table(
            border_style="dark_red",
            header_style="bold red",
            expand=True,
            show_lines=True,
        )
        corr_table.add_column("Asset Pair", style="bold white", min_width=18)
        corr_table.add_column("Expected", justify="center", min_width=10)
        corr_table.add_column("30D Corr", justify="center", min_width=9)
        corr_table.add_column("7D Corr", justify="center", min_width=9)
        corr_table.add_column("Δ Status", justify="center", min_width=13)
        corr_table.add_column("Strategic Interpretation", min_width=32)

        if anomaly_rows:
            for pk, pinfo in anomaly_rows:
                name = nice_names.get(pk, pk.replace("_", " vs "))
                c30  = pinfo.get("correlation_30d")
                c7   = pinfo.get("correlation_7d")
                c30_str = f"{c30:+.2f}" if c30 is not None else "N/A"
                c7_str  = f"{c7:+.2f}"  if c7  is not None else "N/A"
                c30_col = "bright_green" if (c30 or 0) > 0 else "bright_red"
                c7_col  = "bright_green" if (c7  or 0) > 0 else "bright_red"
                corr_table.add_row(
                    name,
                    f"[dim]{pinfo.get('expected', 'N/A')}[/dim]",
                    f"[{c30_col}]{c30_str}[/{c30_col}]",
                    f"[{c7_col}]{c7_str}[/{c7_col}]",
                    "[bold yellow on dark_orange3] ⚡ DECOUPLED [/bold yellow on dark_orange3]",
                    f"[yellow]{pinfo.get('interpretation', 'Decoupled')}[/yellow]"
                )
        else:
            corr_table.add_row(
                "All Pairs", "—", "—", "—",
                "[bold bright_green on dark_green] ✓ ALIGNED [/bold bright_green on dark_green]",
                "[bright_green]All intermarket relationships are historically aligned.[/bright_green]"
            )

        console.print(corr_table)
        console.print()

        # ── SECTION 5: SYSTEMIC RISK DASHBOARD ──────────────────────────────
        console.print(Rule("[bold white]  SYSTEMIC RISK MONITOR  [/bold white]", style="red"))
        console.print()

        risk_table = Table(
            border_style="dark_red",
            header_style="bold red",
            expand=True,
            show_lines=True,
        )
        risk_table.add_column("Risk Category", style="bold white", min_width=22)
        risk_table.add_column("Level", justify="center", min_width=20)
        risk_table.add_column("Guard / Remediation Action", min_width=38)

        yc_inverted = macro_state.get("yield_curve_inverted", False)
        if yc_inverted:
            yc_level  = "[bold white on red] 🔴 INVERTED [/bold white on red]"
            yc_guard  = "Hedge via gold/TLT; recession probability elevated."
        else:
            yc_level  = "[bold bright_green on dark_green] 🟢  NORMAL  [/bold bright_green on dark_green]"
            yc_guard  = "Standard sector weightings active."

        if vix_val is not None and not np.isnan(vix_val) and vix_val > 30:
            vix_level = "[bold white on red] 🔴 HIGH VOL [/bold white on red]"
            vix_guard = "Reduce size 30–50%, increase hedge ratios."
        elif vix_val is not None and not np.isnan(vix_val) and vix_val > 20:
            vix_level = "[bold black on yellow] 🟡 ELEVATED [/bold black on yellow]"
            vix_guard = "Tighten stops; monitor closely."
        else:
            vix_level = "[bold bright_green on dark_green] 🟢  LOW VOL  [/bold bright_green on dark_green]"
            vix_guard = "Normal positioning parameters active."

        regime_risk = "[bold white on red] 🔴 RISK-OFF [/bold white on red]" if ("RISK-OFF" in regime or "RISK_OFF" in regime) else (
                      "[bold bright_green on dark_green] 🟢  RISK-ON  [/bold bright_green on dark_green]" if ("RISK-ON" in regime or "RISK_ON" in regime) else
                      "[bold black on yellow] 🟡  MIXED   [/bold black on yellow]")
        regime_guard = "De-risk, reduce beta exposure." if ("RISK-OFF" in regime or "RISK_OFF" in regime) else "Favour growth assets and cyclicals."

        risk_table.add_row("Yield Curve Spread", yc_level, yc_guard)
        risk_table.add_row("Market Volatility (VIX)", vix_level, vix_guard)
        risk_table.add_row("Macro Regime Bias", regime_risk, regime_guard)

        console.print(risk_table)

        # ── FOOTER ──────────────────────────────────────────────────────────
        console.print()
        console.print(Rule(style="bright_blue"))
        footer = Text(justify="center")
        footer.append("  LUNA v1.0.0  ", style="bold bright_blue")
        footer.append("│  Autonomous Intermarket Intelligence Suite  │  ", style="dim white")
        footer.append("All outputs are for informational purposes only. Not financial advice.  ", style="dim red")
        console.print(Align.center(footer))
        console.print()

    except Exception as e:
        logger.error("Failed to render Bloomberg Terminal CLI output: %s", e)




# ---------------------------------------------------------------------------
def notify_run_complete(
    macro_state: dict,
    opportunities: list,
    alerts: list,
    config: dict,
    portfolio_status: dict | None = None,
) -> None:
    """
    Send end-of-cycle notifications.

    Always prints a colored brief to the console.
    If Telegram credentials are present in config, also sends to Telegram.
    """
    # --- Console ---
    render_bloomberg_terminal(macro_state, opportunities, alerts, config, portfolio_status=portfolio_status)

    # --- Individual alert lines to console (with duplicate tag check) ---
    for alert in alerts:
        try:
            is_dup = should_suppress_alert(alert, record=False)
            dup_tag = f" {_C.DIM}[SUPPRESSED DUPLICATE]{_C.RESET}" if is_dup else ""
            print(format_alert_message(alert) + dup_tag)
        except Exception as exc:
            logger.error("Failed to format alert for console: %s", exc)

    # --- Telegram ---
    bot_token = config.get("TELEGRAM_BOT_TOKEN")
    chat_id   = config.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return

    # Check and record duplicate status of alerts
    non_duplicate_alerts = []
    for alert in alerts:
        if not should_suppress_alert(alert, record=True):
            non_duplicate_alerts.append(alert)

    try:
        # Pass non-duplicate list to Telegram summary brief to keep it uncluttered
        tg_msg = _telegram_brief(macro_state, opportunities, non_duplicate_alerts)
        success = send_telegram(tg_msg, bot_token, chat_id)
        if not success:
            logger.warning("Telegram brief not delivered (run_complete).")

        # Send each CRITICAL or HIGH alert separately for immediate visibility
        for alert in non_duplicate_alerts:
            label, _, _ = get_priority_details(alert)
            if "CRITICAL" in label or "HIGH" in label:
                try:
                    a_msg = _telegram_alert(alert)
                    send_telegram(a_msg, bot_token, chat_id)
                except Exception as exc:
                    logger.error("Failed to send Telegram prioritized alert: %s", exc)
    except Exception as exc:
        logger.error("Unexpected error in notify_run_complete (Telegram): %s", exc)


def notify_alert(alert: dict, config: dict) -> None:
    """
    Immediately push a single triggered alert to console and Telegram.
    Suppress duplicates to Telegram using 4-hour duplicate filter.
    """
    # Check deduplication without recording for console print
    is_duplicate = should_suppress_alert(alert, record=False)
    
    # --- Console ---
    try:
        dup_tag = f" {_C.DIM}[SUPPRESSED DUPLICATE]{_C.RESET}" if is_duplicate else ""
        print(format_alert_message(alert) + dup_tag)
    except Exception as exc:
        logger.error("Failed to format alert for console output: %s", exc)

    # Check and record duplicate status for Telegram dispatch
    is_duplicate_real = should_suppress_alert(alert, record=True)
    if is_duplicate_real:
        return

    # --- Telegram ---
    bot_token = config.get("TELEGRAM_BOT_TOKEN")
    chat_id   = config.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return

    try:
        tg_msg = _telegram_alert(alert)
        success = send_telegram(tg_msg, bot_token, chat_id)
        if not success:
            logger.warning(
                "Telegram alert for %s not delivered.", alert.get("asset", "UNKNOWN")
            )
    except Exception as exc:
        logger.error("Unexpected error in notify_alert (Telegram): %s", exc)


def notify_daily_summary(
    market_data: dict,
    opportunities: list,
    macro_state: dict,
    calendar: list,
    config: dict,
) -> None:
    """
    Send the 7am daily digest to console and (optionally) Telegram.

    Parameters
    ----------
    market_data   : Dict keyed by symbol with price / pct_change / volume.
    opportunities : Ranked opportunity list.
    macro_state   : Macro analysis result dict.
    calendar      : Upcoming economic events list.
    config        : Config dict.
    """
    # --- Console ---
    try:
        console_msg = format_daily_summary(market_data, opportunities, macro_state, calendar)
        print(console_msg)
    except Exception as exc:
        logger.error("Failed to format daily summary for console: %s", exc)

    # --- Telegram ---
    bot_token = config.get("TELEGRAM_BOT_TOKEN")
    chat_id   = config.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return

    try:
        tg_msg = _telegram_daily_summary(market_data, opportunities, macro_state, calendar)
        success = send_telegram(tg_msg, bot_token, chat_id)
        if not success:
            logger.warning("Telegram daily summary not delivered.")
    except Exception as exc:
        logger.error("Unexpected error in notify_daily_summary (Telegram): %s", exc)
