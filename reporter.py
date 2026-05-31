"""
reporter.py — Autonomous Trading Research Agent
Generates all markdown reports and persists state.
"""

import json
import os
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
REPORTS_DIR = Path("reports")
STATE_DIR = Path("state")

DAILY_BRIEF_PATH       = REPORTS_DIR / "daily-brief.md"
OPPORTUNITIES_PATH     = REPORTS_DIR / "opportunities.md"
WATCHLIST_STATUS_PATH  = REPORTS_DIR / "watchlist-status.md"
MACRO_DASHBOARD_PATH   = REPORTS_DIR / "macro-dashboard.md"
ECONOMIC_CALENDAR_PATH = REPORTS_DIR / "economic-calendar.md"
STATE_PATH             = STATE_DIR / "last-run.json"

IST = timezone(timedelta(hours=5, minutes=30))


def _ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_pct(value: float) -> str:
    """
    Format a percentage value with sign and colour emoji.
    e.g. '+3.42% 🟢' or '-1.23% 🔴'
    Returns 'N/A' when value is None or non-finite.
    """
    try:
        if value is None or value != value:  # catches None and NaN
            return "N/A"
        sign = "+" if value >= 0 else ""
        emoji = "🟢" if value >= 0 else "🔴"
        return f"{sign}{value:.2f}% {emoji}"
    except (TypeError, ValueError):
        return "N/A"


def format_price(value: float) -> str:
    """
    Smart price formatting:
    - 6 decimal places for prices < $1
    - 2 decimal places for prices ≥ $1
    Returns 'N/A' when value is None or invalid.
    """
    try:
        if value is None or value != value:
            return "N/A"
        if abs(value) < 1.0:
            return f"${value:.6f}"
        return f"${value:,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def format_stars(rating: float) -> str:
    """
    Convert a 0–10 rating to a ★/☆ display.
    0–3  → ★☆☆
    4–6  → ★★☆
    7–10 → ★★★
    """
    try:
        if rating is None:
            return "☆☆☆"
        rating = float(rating)
        if rating >= 7:
            return "★★★"
        if rating >= 4:
            return "★★☆"
        return "★☆☆"
    except (TypeError, ValueError):
        return "☆☆☆"


def check_stale(fetch_time_str: str) -> tuple[str, str]:
    """
    Check if the fetch time is older than 2 hours.
    Returns (display_time, status_tag).
    """
    if not fetch_time_str:
        return "N/A", " [STALE]"
    try:
        # Standardise Z to +00:00 for ISO parsing in Python < 3.11
        dt = datetime.fromisoformat(fetch_time_str.replace("Z", "+00:00"))
        dt_ist = dt.astimezone(IST)
        time_display = dt_ist.strftime("%Y-%m-%d %H:%M:%S")
        
        diff = (datetime.now(timezone.utc) - dt).total_seconds()
        if diff > 7200:  # 2 hours
            return time_display, " [STALE]"
        return time_display, ""
    except Exception:
        return "N/A", " [STALE]"


def build_asset_table(assets: list) -> str:
    """
    Build a markdown table for a list of asset dicts.

    Expected keys per asset (all optional — falls back to N/A):
        ticker, name, price, change_24h, change_7d,
        rsi, macd_signal, ema_stack, bb_position, score, alerts
    """
    if not assets:
        return "_No data available._\n"

    header = (
        "| Asset | Price | 24h | 7d | RSI | MACD | EMA Stack | BB Pos | Score | Fetch Time | Status |\n"
        "|-------|-------|-----|----|-----|------|-----------|--------|-------|------------|--------|\n"
    )
    rows = []
    for a in assets:
        ticker      = a.get("ticker", "?")
        name        = a.get("name", ticker)
        price       = format_price(a.get("price"))
        chg_24h     = format_pct(a.get("change_24h"))
        chg_7d      = format_pct(a.get("change_7d"))
        rsi_val     = a.get("rsi")
        rsi_str     = f"{rsi_val:.1f}" if rsi_val is not None else "N/A"
        macd        = a.get("macd_signal", "N/A") or "N/A"
        ema_stack   = a.get("ema_stack", "N/A") or "N/A"
        bb_pos      = a.get("bb_position", "N/A") or "N/A"
        score_val   = a.get("score")
        score_str   = (
            f"{score_val:+.1f}" if score_val is not None else "N/A"
        )
        alert_list  = a.get("alerts", [])
        alert_str   = ", ".join(alert_list) if alert_list else "—"
        
        fetch_time_str = a.get("fetch_time")
        time_disp, stale_tag = check_stale(fetch_time_str)
        status_disp = f"🔴 {stale_tag.strip()}" if stale_tag else "🟢 Active"
        
        rows.append(
            f"| **{name}** ({ticker}) | {price} | {chg_24h} | {chg_7d} "
            f"| {rsi_str} | {macd} | {ema_stack} | {bb_pos} | {score_str} | {time_disp} | {status_disp} |"
        )

    return header + "\n".join(rows) + "\n"


def build_opportunity_block(opp: dict) -> str:
    """
    Build a formatted markdown block for a single opportunity.

    Expected keys:
        name, ticker, asset_class, price, change_24h, score,
        score_breakdown, support, resistance, reasoning,
        catalyst, invalidation_level
    """
    name        = opp.get("name", opp.get("ticker", "Unknown"))
    ticker      = opp.get("ticker", "?")
    asset_cls   = opp.get("asset_class", "N/A")
    price       = format_price(opp.get("price"))
    chg_24h     = format_pct(opp.get("change_24h"))
    score       = opp.get("score", 0)
    breakdown   = opp.get("score_breakdown", "N/A")
    support     = format_price(opp.get("support"))
    resistance  = format_price(opp.get("resistance"))
    reasoning   = opp.get("reasoning", "No reasoning provided.")
    catalyst    = opp.get("catalyst", "None identified")
    invalid_lvl = format_price(opp.get("invalidation_level"))
    
    fetch_time_str = opp.get("fetch_time")
    time_disp, stale_tag = check_stale(fetch_time_str)

    score_str = f"{score:+.1f}" if score is not None else "N/A"

    block = (
        f"### {name} ({ticker}) — Score: {score_str}/10\n"
        f"- **Asset Class:** {asset_cls}\n"
        f"- **Price:** {price} | **24h:** {chg_24h}{stale_tag} (Fetched: {time_disp})\n"
        f"- **Score Breakdown:** {breakdown}\n"
        f"- **Key Levels:** Support {support}, Resistance {resistance}\n"
        f"- **Why Now:** {reasoning}\n"
        f"- **Catalyst:** {catalyst}\n"
        f"- **Invalidation:** Below {invalid_lvl} this setup fails\n"
    )
    return block


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------

def generate_daily_brief(
    market_data: dict,
    macro_state: dict,
    opportunities: list,
    research: dict,
    alerts: list,
    config: dict,
) -> str:
    """
    Generate the Daily Market Brief markdown string.

    Parameters
    ----------
    market_data   : dict  — raw market snapshot; keys: global_snapshot, top_movers, etc.
    macro_state   : dict  — regime, VIX, DXY, yield_curve_inverted, etc.
    opportunities : list  — list of opportunity dicts
    research      : dict  — headlines, risks, events
    alerts        : list  — list of alert strings
    config        : dict  — agent config (unused in formatting, reserved)
    """
    now_ist = datetime.now(IST)
    date_str  = now_ist.strftime("%A, %d %B %Y")
    time_str  = now_ist.strftime("%H:%M")
    ts_str    = now_ist.strftime("%Y-%m-%d %H:%M:%S")

    regime  = macro_state.get("regime", "Unknown")
    vix_val = macro_state.get("vix")
    dxy_val = macro_state.get("dxy")
    vix_str = f"{vix_val:.2f}" if vix_val is not None else "N/A"
    dxy_str = f"{dxy_val:.2f}" if dxy_val is not None else "N/A"
    yc_inv  = macro_state.get("yield_curve_inverted", False)

    # --- Banners ---
    banners = ""
    if vix_val is not None and vix_val > 30:
        banners += "\n> ⚠️ **HIGH VOLATILITY REGIME** — VIX above 30. Reduce position sizes. Expect wide intraday swings.\n"
    if yc_inv:
        banners += "\n> 🔔 **YIELD CURVE INVERTED** — 2Y/10Y spread negative. Recession probability elevated.\n"

    # --- Global Snapshot table ---
    snapshot_assets = market_data.get("global_snapshot", [])
    snapshot_header = (
        "| Asset | Price | 24h | 7d |\n"
        "|-------|-------|-----|----|"
    )
    snapshot_rows = []
    for a in snapshot_assets:
        name    = a.get("name", a.get("ticker", "?"))
        price   = format_price(a.get("price"))
        chg_24h = format_pct(a.get("change_24h"))
        chg_7d  = format_pct(a.get("change_7d"))
        snapshot_rows.append(f"| {name} | {price} | {chg_24h} | {chg_7d} |")
    snapshot_table = snapshot_header + "\n" + "\n".join(snapshot_rows) if snapshot_rows else "_No data._"

    # --- Headlines ---
    headlines = research.get("headlines", [])
    headlines_md = ""
    for i, h in enumerate(headlines[:5], 1):
        title  = h.get("title", "Untitled")
        source = h.get("source", "Unknown")
        impact = h.get("impact", "No impact summary available.")
        headlines_md += f"{i}. **{title}** *(via {source})*\n   > {impact}\n\n"
    if not headlines_md:
        headlines_md = "_No headlines available._\n"

    # --- Top Movers ---
    movers = market_data.get("top_movers", {})
    gainers = movers.get("gainers", [])
    losers  = movers.get("losers", [])

    def _fmt_movers(lst: list) -> str:
        if not lst:
            return "_None_"
        parts = []
        for m in lst:
            ticker  = m.get("ticker", "?")
            chg     = format_pct(m.get("change_24h"))
            parts.append(f"**{ticker}** {chg}")
        return " · ".join(parts)

    gainers_str = _fmt_movers(gainers)
    losers_str  = _fmt_movers(losers)

    # --- Risks ---
    risks = research.get("risks", [])
    risks_md = ""
    for r in risks:
        if isinstance(r, str):
            risks_md += f"- {r}\n"
        elif isinstance(r, dict):
            risks_md += f"- **{r.get('title', 'Risk')}**: {r.get('detail', '')}\n"
    if not risks_md:
        risks_md = "- No specific risks flagged.\n"

    # --- Events This Week ---
    events = research.get("events_this_week", [])
    events_md = ""
    for ev in events:
        if isinstance(ev, str):
            events_md += f"- {ev}\n"
        elif isinstance(ev, dict):
            ev_date  = ev.get("date", "TBD")
            ev_name  = ev.get("event", "?")
            ev_curr  = ev.get("currency", "")
            events_md += f"- `{ev_date}` — **{ev_name}** {ev_curr}\n"
    if not events_md:
        events_md = "- No events listed.\n"

    # --- Active Alerts ---
    alerts_md = ""
    for alert in alerts:
        if isinstance(alert, str):
            alerts_md += f"- 🚨 {alert}\n"
        elif isinstance(alert, dict):
            alerts_md += f"- 🚨 **{alert.get('ticker', '?')}**: {alert.get('message', '')}\n"
    if not alerts_md:
        alerts_md = "- No active alerts.\n"

    # --- Assemble ---
    report = f"""# Daily Market Brief — {date_str} {time_str} IST

> 🌍 **Market Regime:** {regime} | **VIX:** {vix_str} | **DXY:** {dxy_str}
{banners}
## 📊 Global Snapshot

{snapshot_table}

## 📰 Top Headlines

{headlines_md.rstrip()}

## 🔥 Top Movers (24h)

**Gainers:** {gainers_str}
**Losers:** {losers_str}

## ⚠️ Key Risks

{risks_md.rstrip()}

## 🗓️ Events This Week

{events_md.rstrip()}

## 🚨 Active Alerts

{alerts_md.rstrip()}

---
*Generated: {ts_str} IST | This is automated research output. Not financial advice. Always do your own research.*
"""
    return report


def generate_opportunities_report(
    opportunities: list,
    market_data: dict,
    calendar: list,
    timestamp: datetime,
) -> str:
    """
    Generate the Opportunities & Setups markdown report.

    Parameters
    ----------
    opportunities : list  — full list of opportunity dicts with 'score' field
    market_data   : dict  — reserved (can be used for price cross-reference)
    calendar      : list  — economic calendar events (used for upcoming catalysts)
    timestamp     : datetime
    """
    ts_ist  = timestamp.astimezone(IST) if timestamp.tzinfo else timestamp.replace(tzinfo=IST)
    date_str = ts_ist.strftime("%A, %d %B %Y")

    bullish  = [o for o in opportunities if (o.get("score") or 0) >= 6]
    bearish  = [o for o in opportunities if (o.get("score") or 0) <= -6]
    neutral  = [o for o in opportunities if -6 < (o.get("score") or 0) < 6]

    # --- Correlation breaks ---
    corr_breaks = [
        o for o in neutral
        if o.get("correlation_break") or o.get("tag") == "correlation_break"
    ]

    # --- Momentum plays (new highs) ---
    momentum = [
        o for o in opportunities
        if o.get("new_high") or o.get("tag") == "momentum"
    ]

    # --- BB Squeeze breakout candidates ---
    squeeze = [
        o for o in opportunities
        if o.get("bb_squeeze") or o.get("tag") == "bb_squeeze"
    ]

    def _section(opp_list: list) -> str:
        if not opp_list:
            return "_None identified._\n"
        blocks = []
        for o in sorted(opp_list, key=lambda x: abs(x.get("score", 0)), reverse=True):
            blocks.append(build_opportunity_block(o))
        return "\n".join(blocks)

    # Correlation breaks section
    corr_section = ""
    if corr_breaks:
        for o in corr_breaks:
            name   = o.get("name", o.get("ticker", "?"))
            detail = o.get("correlation_note", "Unusual divergence detected from typical correlation.")
            corr_section += f"- **{name}**: {detail}\n"
    else:
        corr_section = "_No correlation breaks detected._\n"

    # Momentum section
    momentum_section = ""
    if momentum:
        for o in momentum:
            name  = o.get("name", o.get("ticker", "?"))
            price = format_price(o.get("price"))
            chg   = format_pct(o.get("change_24h"))
            momentum_section += f"- **{name}** at {price} ({chg}) — Breaking to new highs\n"
    else:
        momentum_section = "_No momentum plays (new highs) detected._\n"

    # Squeeze section
    squeeze_section = ""
    if squeeze:
        for o in squeeze:
            name   = o.get("name", o.get("ticker", "?"))
            price  = format_price(o.get("price"))
            detail = o.get("squeeze_note", "Bollinger Band squeeze detected. Breakout imminent.")
            squeeze_section += f"- **{name}** at {price} — {detail}\n"
    else:
        squeeze_section = "_No BB squeeze candidates._\n"

    report = f"""# Opportunities & Setups — {date_str}

## 🟢 Bullish Setups (Score ≥ 6)

{_section(bullish).rstrip()}

## 🔴 Bearish / Risk Setups (Score ≤ -6)

{_section(bearish).rstrip()}

## 🔄 Correlation Breaks

{corr_section.rstrip()}

## 💥 Momentum Plays (New Highs)

{momentum_section.rstrip()}

## 🎯 BB Squeeze Breakout Candidates

{squeeze_section.rstrip()}

---
*Not financial advice. Automated research only.*
"""
    return report


def generate_watchlist_status(
    enriched_data: dict,
    macro_state: dict,
    timestamp: datetime,
) -> str:
    """
    Generate the Watchlist Status markdown report.

    Parameters
    ----------
    enriched_data : dict  — keys are asset categories (e.g. 'crypto', 'stocks',
                            'forex', 'commodities'); values are lists of asset dicts
    macro_state   : dict  — macro context (for header)
    timestamp     : datetime
    """
    ts_ist   = timestamp.astimezone(IST) if timestamp.tzinfo else timestamp.replace(tzinfo=IST)
    date_str = ts_ist.strftime("%A, %d %B %Y")
    ts_str   = ts_ist.strftime("%Y-%m-%d %H:%M:%S")

    regime  = macro_state.get("regime", "Unknown")
    vix_val = macro_state.get("vix")
    vix_str = f"{vix_val:.2f}" if vix_val is not None else "N/A"

    category_icons = {
        "crypto":      "📊 Crypto",
        "stocks":      "📈 Stocks",
        "forex":       "💱 Forex",
        "commodities": "🏗️ Commodities",
        "etfs":        "📦 ETFs",
        "indices":     "🌐 Indices",
    }

    sections = []
    for category, assets in enriched_data.items():
        if not assets:
            continue
        icon_label = category_icons.get(category.lower(), f"📌 {category.title()}")
        table = build_asset_table(assets)
        sections.append(f"## {icon_label}\n\n{table}")

    if not sections:
        sections_md = "_No watchlist data available._\n"
    else:
        sections_md = "\n\n".join(sections)

    report = f"""# Watchlist Status — {date_str}

> **Regime:** {regime} | **VIX:** {vix_str}

{sections_md}

---
*Generated: {ts_str} IST | Automated research output. Not financial advice.*
"""
    return report


def generate_macro_dashboard(
    macro_state: dict,
    treasury_data: dict,
    fred_data: dict,
    timestamp: datetime,
) -> str:
    """
    Generate the full Macro Dashboard markdown report.

    Parameters
    ----------
    macro_state   : dict  — regime, vix, dxy, yield_curve, etc.
    treasury_data : dict  — treasury yield data keyed by maturity ('2y', '5y', '10y', '30y')
    fred_data     : dict  — FRED macro series: cpi, pce, unemployment, gdp, m2, fed_funds
    timestamp     : datetime
    """
    ts_ist   = timestamp.astimezone(IST) if timestamp.tzinfo else timestamp.replace(tzinfo=IST)
    date_str = ts_ist.strftime("%A, %d %B %Y")
    ts_str   = ts_ist.strftime("%Y-%m-%d %H:%M:%S")

    regime  = macro_state.get("regime", "Unknown")
    vix_val = macro_state.get("vix")
    dxy_val = macro_state.get("dxy")
    vix_str = f"{vix_val:.2f}" if vix_val is not None else "N/A"
    dxy_str = f"{dxy_val:.2f}" if dxy_val is not None else "N/A"
    yc_inv  = macro_state.get("yield_curve_inverted", False)
    yc_spread = macro_state.get("yield_curve_spread")
    spread_str = f"{yc_spread:+.2f} bps" if yc_spread is not None else "N/A"
    yc_label   = "🔴 INVERTED" if yc_inv else "🟢 Normal"

    # --- Volatility banners ---
    vix_banner = ""
    if vix_val is not None and vix_val > 30.0:
        vix_banner = f"\n> 🔴 **CRITICAL WARNING: HIGH VOLATILITY REGIME (VIX > 30)** — VIX is currently **{vix_val:.2f}**. Extreme market stress detected. Protect capital and minimize risk exposure.\n"

    # --- Sector Rotation ---
    import numpy as np
    sector_rotation = macro_state.get("sector_rotation", {})
    sector_section = ""
    if sector_rotation:
        rotation_rows = []
        for sec_1m in sector_rotation.get("ranked_1m", []):
            ticker = sec_1m["ticker"]
            name = sec_1m["name"]
            rel_perf_1m = sec_1m["rel_perf"]
            
            # Find 1w relative performance
            rel_perf_1w = 0.0
            for sec_1w in sector_rotation.get("ranked_1w", []):
                if sec_1w["ticker"] == ticker:
                    rel_perf_1w = sec_1w["rel_perf"]
                    break
                    
            emoji_1m = "🟢" if rel_perf_1m >= 0 else "🔴"
            emoji_1w = "🟢" if rel_perf_1w >= 0 else "🔴"
            
            rel_perf_1w_str = f"+{rel_perf_1w:.2f}% {emoji_1w}" if rel_perf_1w >= 0 else f"{rel_perf_1w:.2f}% {emoji_1w}"
            rel_perf_1m_str = f"+{rel_perf_1m:.2f}% {emoji_1m}" if rel_perf_1m >= 0 else f"{rel_perf_1m:.2f}% {emoji_1m}"
            
            rotation_rows.append(f"| **{name} ({ticker})** | {rel_perf_1w_str} | {rel_perf_1m_str} |")
            
        rotation_table = (
            "| Sector ETF | 1W Rel. Performance vs SPY | 1M Rel. Performance vs SPY |\n"
            "| :--- | :--- | :--- |\n"
            + "\n".join(rotation_rows)
        )
        
        signals_list = []
        for sig in sector_rotation.get("rotation_signals", []):
            sig_type = sig["type"]
            emoji = "🚀" if sig_type == "BULLISH_ROTATION" else "⚠️"
            signals_list.append(f"- {emoji} **{sig['details']}**")
            
        signals_md = "\n".join(signals_list) if signals_list else "_No significant bottom-to-top sector rotation signals detected this cycle._"
        
        sector_section = f"""
---

## 🔄 Sector Rotation Dashboard

Performance relative to SPY (S&P 500) over 1-Week and 1-Month periods.

{rotation_table}

### 🎯 Sector Rotation Signals
{signals_md}
"""
    else:
        sector_section = "\n---\n\n## 🔄 Sector Rotation Dashboard\n\n_Sector rotation analysis is currently unavailable._\n"

    # --- Regime classification ---
    regime_detail = macro_state.get("regime_detail", "")
    risk_score    = macro_state.get("risk_score")
    risk_str      = f"{risk_score:.1f}/10" if risk_score is not None else "N/A"
    stars_disp    = format_stars((risk_score or 0))

    # --- Treasury yields table ---
    yield_maturities = ["1m", "3m", "6m", "1y", "2y", "5y", "10y", "20y", "30y"]
    yield_rows = []
    for mat in yield_maturities:
        val = treasury_data.get(mat)
        if val is not None:
            try:
                yield_rows.append(f"| {mat.upper()} | {float(val):.3f}% |")
            except (TypeError, ValueError):
                yield_rows.append(f"| {mat.upper()} | N/A |")

    if yield_rows:
        yield_table = (
            "| Maturity | Yield |\n"
            "|----------|-------|\n"
            + "\n".join(yield_rows)
        )
    else:
        yield_table = "_Treasury data unavailable._"

    # --- FRED macro data ---
    def _fred_val(key: str, fmt: str = ".2f", suffix: str = "") -> str:
        val = fred_data.get(key)
        if val is None:
            return "N/A"
        try:
            return f"{float(val):{fmt}}{suffix}"
        except (TypeError, ValueError):
            return str(val)

    cpi_val        = _fred_val("cpi", ".2f", "%")
    pce_val        = _fred_val("pce", ".2f", "%")
    unemployment   = _fred_val("unemployment", ".2f", "%")
    gdp_growth     = _fred_val("gdp", ".2f", "%")
    m2_growth      = _fred_val("m2", ".2f", "%")
    fed_funds_rate = _fred_val("fed_funds", ".2f", "%")

    # --- Fed meeting / stance ---
    fed_stance     = macro_state.get("fed_stance", "Unknown")
    next_meeting   = macro_state.get("next_fed_meeting", "TBD")
    hike_prob      = macro_state.get("hike_probability")
    cut_prob       = macro_state.get("cut_probability")
    hike_str       = f"{hike_prob:.1f}%" if hike_prob is not None else "N/A"
    cut_str        = f"{cut_prob:.1f}%"  if cut_prob is not None else "N/A"

    # --- Fear & Greed / Sentiment ---
    fear_greed     = macro_state.get("fear_greed_index")
    fg_label       = macro_state.get("fear_greed_label", "Unknown")
    fg_str         = f"{fear_greed:.0f}" if fear_greed is not None else "N/A"
    btc_dominance  = macro_state.get("btc_dominance")
    btc_dom_str    = f"{btc_dominance:.2f}%" if btc_dominance is not None else "N/A"
    alt_season     = macro_state.get("alt_season_index")
    alt_str        = f"{alt_season:.0f}/100" if alt_season is not None else "N/A"

    # --- Global equity mood ---
    equity_mood    = macro_state.get("equity_mood", "Neutral")
    spx_trend      = macro_state.get("spx_trend", "N/A")
    global_risk    = macro_state.get("global_risk_on", None)
    risk_label     = "🟢 Risk-ON" if global_risk else ("🔴 Risk-OFF" if global_risk is False else "⚪ Neutral")

    # --- Cross-Asset Correlation Anomalies ---
    corr_section = ""
    correlations_data = macro_state.get("correlations", {})
    pairs_data = correlations_data.get("pairs", {})
    
    corr_rows = []
    has_anomalies = False
    
    nice_names = {
        "GC=F_DX-Y.NYB": "Gold vs DXY",
        "^TNX_^GSPC": "10Y Yield vs S&P 500",
        "CL=F_^GSPC": "Oil vs S&P 500",
        "BTC_^NDX": "Bitcoin vs NASDAQ",
        "GC=F_^VIX": "Gold vs VIX",
        "HYG_^GSPC": "HY Bonds vs S&P 500",
        "EURUSD=X_GC=F": "EUR/USD vs Gold",
        "^TNX_DX-Y.NYB": "10Y Yield vs DXY",
        "CL=F_USDINR=X": "Oil vs USD/INR"
    }
    
    for pk, pinfo in pairs_data.items():
        name = nice_names.get(pk, pk.replace("_", " vs "))
        expected = pinfo.get("expected", "N/A")
        c30 = pinfo.get("correlation_30d")
        c7 = pinfo.get("correlation_7d")
        c30_str = f"{c30:+.2f}" if c30 is not None else "N/A"
        c7_str = f"{c7:+.2f}" if c7 is not None else "N/A"
        
        anomaly = pinfo.get("anomaly", False)
        status = "🟢 Aligned"
        if anomaly:
            has_anomalies = True
            since = pinfo.get("anomaly_since")
            since_str = f" since {since}" if since else ""
            status = f"⚠️ **ANOMALY**{since_str}"
            
        interpretation = pinfo.get("interpretation", "")
        corr_rows.append(f"| {name} | {expected} | {c30_str} | {c7_str} | {status} | {interpretation} |")

    if corr_rows:
        corr_rows_joined = "\n".join(corr_rows)
        corr_section = f"""
## 🔄 Cross-Asset Correlation Anomalies

| Pair | Normal Relationship | 30D Corr | 7D Corr | Status | Interpretation / Impending Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
{corr_rows_joined}

"""
        if has_anomalies:
            corr_section += "> ⚠️ **Warning**: Significant cross-asset correlation anomalies detected. Normal macro intermarket relationships have decoupled. This typically precedes regime shifts or indicates high systemic stress.\n"
        else:
            corr_section += "> 🟢 **Status**: All key cross-asset relationships are aligned within standard historical expectations.\n"
    else:
        corr_section = "\n## 🔄 Cross-Asset Correlation Anomalies\n\n_Cross-asset correlation data is currently unavailable._\n"

    report = f"""# Macro Dashboard — {date_str}

> 🌍 **Regime:** {regime} | **Risk Score:** {risk_str} {stars_disp} | **Global Stance:** {risk_label}

{f"> ⚠️ **YIELD CURVE INVERTED** — Spread: {spread_str}" if yc_inv else f"> 📈 Yield Curve: {yc_label} | Spread: {spread_str}"}
{vix_banner}
---

## 🏦 Federal Reserve

| Metric | Value |
|--------|-------|
| Fed Funds Rate | {fed_funds_rate} |
| Fed Stance | {fed_stance} |
| Next FOMC Meeting | {next_meeting} |
| Hike Probability | {hike_str} |
| Cut Probability | {cut_str} |

---

## 📊 Key Economic Indicators (FRED)

| Indicator | Latest |
|-----------|--------|
| CPI (YoY) | {cpi_val} |
| PCE (YoY) | {pce_val} |
| Unemployment Rate | {unemployment} |
| GDP Growth (QoQ) | {gdp_growth} |
| M2 Money Supply Growth | {m2_growth} |

---

## 📉 Treasury Yield Curve

{yield_table}

> **2Y/10Y Spread:** {spread_str} — {yc_label}

---

## 😨 Market Sentiment

| Metric | Value |
|--------|-------|
| Fear & Greed Index | {fg_str} ({fg_label}) |
| BTC Dominance | {btc_dom_str} |
| Alt Season Index | {alt_str} |
| Equity Mood | {equity_mood} |
| SPX Trend | {spx_trend} |

---

## 💱 Currency & Volatility

| Metric | Value |
|--------|-------|
| DXY (USD Index) | {dxy_str} |
| VIX | {vix_str} |

---
{corr_section}
{sector_section}
---

## 🔍 Regime Analysis

**Current Regime:** {regime}
{f"**Detail:** {regime_detail}" if regime_detail else ""}

| Signal | Reading |
|--------|---------|
| Risk Appetite | {risk_label} |
| Yield Curve | {yc_label} |
| Fed Stance | {fed_stance} |
| Volatility (VIX) | {vix_str} |
| DXY | {dxy_str} |

---
*Generated: {ts_str} IST | Automated macro analysis. Not financial advice.*
"""
    return report


def generate_economic_calendar(calendar: list, timestamp: datetime) -> str:
    """
    Generate the Economic Calendar markdown report.

    Parameters
    ----------
    calendar  : list of dict — each with keys:
                    date, time, event, currency, impact,
                    forecast, previous, expected_market_impact
    timestamp : datetime
    """
    ts_ist   = timestamp.astimezone(IST) if timestamp.tzinfo else timestamp.replace(tzinfo=IST)
    date_str = ts_ist.strftime("%A, %d %B %Y")
    ts_str   = ts_ist.strftime("%Y-%m-%d %H:%M:%S")

    impact_emoji = {
        "high":   "🔴",
        "medium": "🟡",
        "low":    "🟢",
        "": "⚪",
    }

    header = (
        "| Date | Time | Event | Currency | Impact | Forecast | Previous | Expected Market Impact |\n"
        "|------|------|-------|----------|--------|----------|----------|------------------------|\n"
    )
    rows = []
    for ev in sorted(calendar, key=lambda x: (x.get("date", ""), x.get("time", ""))):
        ev_date   = ev.get("date", "TBD")
        ev_time   = ev.get("time", "TBD")
        ev_name   = ev.get("event", "Unknown Event")
        currency  = ev.get("currency", "—")
        impact    = ev.get("impact", "").lower()
        emoji     = impact_emoji.get(impact, "⚪")
        impact_lbl = f"{emoji} {impact.title()}" if impact else "⚪ N/A"
        forecast  = ev.get("forecast", "N/A") or "N/A"
        previous  = ev.get("previous", "N/A") or "N/A"
        mkt_impact = ev.get("expected_market_impact", "—") or "—"
        rows.append(
            f"| {ev_date} | {ev_time} | **{ev_name}** | {currency} "
            f"| {impact_lbl} | {forecast} | {previous} | {mkt_impact} |"
        )

    if not rows:
        calendar_table = "_No economic events found for the next 7 days._"
    else:
        calendar_table = header + "\n".join(rows)

    # --- Highlight high-impact events ---
    high_impact = [
        ev for ev in calendar
        if ev.get("impact", "").lower() == "high"
    ]
    highlights_md = ""
    if high_impact:
        for ev in high_impact:
            ev_date  = ev.get("date", "TBD")
            ev_name  = ev.get("event", "?")
            currency = ev.get("currency", "")
            mkt_imp  = ev.get("expected_market_impact", "Watch closely.")
            highlights_md += f"- 🔴 `{ev_date}` **{ev_name}** ({currency}) — {mkt_imp}\n"
    else:
        highlights_md = "_No high-impact events this week._\n"

    report = f"""# Economic Calendar — Next 7 Days

> Generated for week of: **{date_str}**

## 📅 Full Schedule

{calendar_table}

## 🔴 High-Impact Events to Watch

{highlights_md.rstrip()}

---
*Generated: {ts_str} IST | Sources: FRED, Investing.com, ForexFactory. Not financial advice.*
"""
    return report


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _make_serializable(obj):
    """
    Recursively convert obj to a JSON-serializable form.
    - DataFrames → None
    - datetime → ISO string
    - numpy types → Python native
    - Sets → lists
    """
    # Attempt pandas DataFrame detection without hard import
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return None
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
    except ImportError:
        pass

    # numpy scalar detection without hard import
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass

    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(i) for i in obj]
    if isinstance(obj, set):
        return [_make_serializable(i) for i in obj]
    return obj


def save_state(
    market_data: dict,
    macro_state: dict,
    opportunities: list,
    timestamp: datetime,
    alerts: list = None,
) -> None:
    """
    Serialize all trading state to state/last-run.json.

    DataFrames are converted to None to ensure JSON-serializability.
    All timestamps are stored as ISO-8601 strings.

    Parameters
    ----------
    market_data   : dict     — raw market snapshot
    macro_state   : dict     — macro regime data
    opportunities : list     — scored opportunity dicts
    timestamp     : datetime — run timestamp
    alerts        : list     — list of alert strings/dicts (optional)
    """
    _ensure_dirs()

    ts_str = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)

    payload = {
        "timestamp": ts_str,
        "market_data": _make_serializable(market_data),
        "macro_state": _make_serializable(macro_state),
        "opportunities": _make_serializable(opportunities),
        "alerts": _make_serializable(alerts or []),
        "meta": {
            "agent": "trading-research-agent",
            "version": "1.0.0",
            "generated_at": datetime.now(IST).isoformat(),
        },
    }

    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info("State saved to %s", STATE_PATH)
    except (OSError, IOError) as exc:
        logger.error("Failed to save state: %s", exc)
    except (TypeError, ValueError) as exc:
        logger.error("State serialization error: %s", exc)


def load_state() -> dict | None:
    """
    Load state from state/last-run.json.

    Returns
    -------
    dict  — previously saved state
    None  — if file not found or JSON parse error
    """
    if not STATE_PATH.exists():
        logger.info("No previous state found at %s", STATE_PATH)
        return None

    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        logger.info("State loaded from %s (timestamp: %s)", STATE_PATH, state.get("timestamp"))
        return state
    except json.JSONDecodeError as exc:
        logger.warning("State file is corrupted (JSON error): %s", exc)
        return None
    except (OSError, IOError) as exc:
        logger.warning("Could not read state file: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error loading state: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Master orchestrator
# ---------------------------------------------------------------------------

def generate_all_reports(
    market_data: dict,
    enriched_data: dict,
    opportunities: list,
    macro_state: dict,
    research_data: dict,
    alerts: list,
    config: dict,
) -> None:
    """
    Orchestrate generation of all reports and state persistence.

    Writes:
      reports/daily-brief.md
      reports/opportunities.md
      reports/watchlist-status.md
      reports/macro-dashboard.md
      reports/economic-calendar.md
      state/last-run.json

    Parameters
    ----------
    market_data   : dict  — raw market snapshot (global_snapshot, top_movers, …)
    enriched_data : dict  — category → list[asset dicts] with indicators
    opportunities : list  — scored opportunity dicts
    macro_state   : dict  — macro regime & sentiment data
    research_data : dict  — headlines, risks, events, treasury_data, fred_data, calendar
    alerts        : list  — list of alert strings or dicts
    config        : dict  — agent configuration
    """
    _ensure_dirs()
    timestamp = datetime.now(IST)

    logger.info("=== Report generation started at %s ===", timestamp.isoformat())

    # ---- Shared sub-structures from research_data ----
    treasury_data = research_data.get("treasury_data", {})
    fred_data     = research_data.get("fred_data", {})
    calendar      = research_data.get("calendar", [])

    # ---- 1. Daily Brief ----
    try:
        logger.info("Generating daily brief…")
        if not market_data and DAILY_BRIEF_PATH.exists():
            logger.warning("Market data is empty. Preserving last good daily brief report.")
        else:
            daily_brief_md = generate_daily_brief(
                market_data  = market_data,
                macro_state  = macro_state,
                opportunities= opportunities,
                research     = research_data,
                alerts       = alerts,
                config       = config,
            )
            if len(daily_brief_md.strip()) < 100 and DAILY_BRIEF_PATH.exists():
                logger.warning("Generated daily brief is too short. Preserving last good report.")
            else:
                DAILY_BRIEF_PATH.write_text(daily_brief_md, encoding="utf-8")
                logger.info("Written: %s", DAILY_BRIEF_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to generate daily brief: %s", exc)

    # ---- 2. Opportunities ----
    try:
        logger.info("Generating opportunities report…")
        if not opportunities and OPPORTUNITIES_PATH.exists():
            logger.warning("Opportunities list is empty. Preserving last good opportunities report.")
        else:
            opp_md = generate_opportunities_report(
                opportunities = opportunities,
                market_data   = market_data,
                calendar      = calendar,
                timestamp     = timestamp,
            )
            if len(opp_md.strip()) < 100 and OPPORTUNITIES_PATH.exists():
                logger.warning("Generated opportunities report is too short. Preserving last good report.")
            else:
                OPPORTUNITIES_PATH.write_text(opp_md, encoding="utf-8")
                logger.info("Written: %s", OPPORTUNITIES_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to generate opportunities report: %s", exc)

    # ---- 3. Watchlist Status ----
    try:
        logger.info("Generating watchlist status…")
        if not enriched_data and WATCHLIST_STATUS_PATH.exists():
            logger.warning("Enriched data is empty. Preserving last good watchlist status report.")
        else:
            watchlist_md = generate_watchlist_status(
                enriched_data = enriched_data,
                macro_state   = macro_state,
                timestamp     = timestamp,
            )
            if len(watchlist_md.strip()) < 100 and WATCHLIST_STATUS_PATH.exists():
                logger.warning("Generated watchlist status report is too short. Preserving last good report.")
            else:
                WATCHLIST_STATUS_PATH.write_text(watchlist_md, encoding="utf-8")
                logger.info("Written: %s", WATCHLIST_STATUS_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to generate watchlist status: %s", exc)

    # ---- 4. Macro Dashboard ----
    try:
        logger.info("Generating macro dashboard…")
        if not macro_state and MACRO_DASHBOARD_PATH.exists():
            logger.warning("Macro state is empty. Preserving last good macro dashboard report.")
        else:
            macro_md = generate_macro_dashboard(
                macro_state   = macro_state,
                treasury_data = treasury_data,
                fred_data     = fred_data,
                timestamp     = timestamp,
            )
            if len(macro_md.strip()) < 100 and MACRO_DASHBOARD_PATH.exists():
                logger.warning("Generated macro dashboard is too short. Preserving last good report.")
            else:
                MACRO_DASHBOARD_PATH.write_text(macro_md, encoding="utf-8")
                logger.info("Written: %s", MACRO_DASHBOARD_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to generate macro dashboard: %s", exc)

    # ---- 5. Economic Calendar ----
    try:
        logger.info("Generating economic calendar…")
        if not calendar and ECONOMIC_CALENDAR_PATH.exists():
            logger.warning("Economic calendar is empty. Preserving last good economic calendar report.")
        else:
            cal_md = generate_economic_calendar(
                calendar  = calendar,
                timestamp = timestamp,
            )
            if len(cal_md.strip()) < 100 and ECONOMIC_CALENDAR_PATH.exists():
                logger.warning("Generated economic calendar is too short. Preserving last good report.")
            else:
                ECONOMIC_CALENDAR_PATH.write_text(cal_md, encoding="utf-8")
                logger.info("Written: %s", ECONOMIC_CALENDAR_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to generate economic calendar: %s", exc)

    # ---- 6. Save State ----
    try:
        logger.info("Saving state…")
        save_state(
            market_data   = market_data,
            macro_state   = macro_state,
            opportunities = opportunities,
            timestamp     = timestamp,
            alerts        = alerts,
        )
        logger.info("Written: %s", STATE_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save state: %s", exc)

    logger.info("=== Report generation complete ===")


AGENT_PERFORMANCE_PATH = REPORTS_DIR / "agent-performance.md"

def generate_performance_report() -> None:
    """
    Generate reports/agent-performance.md containing overall accuracy,
    signal effectiveness, regime performance, recent weight changes,
    and worst calls learning log.
    """
    logger.info("Generating agent performance report...")
    _ensure_dirs()
    
    # 1. Load calls log
    from calls_tracker import load_calls, GLOBAL_WEIGHTS_PATH
    calls_data = load_calls()
    calls = calls_data.get("calls", [])
    
    total_calls = len(calls)
    completed_calls = [c for c in calls if c.get("checked_3d") and c.get("checked_7d")]
    completed_count = len(completed_calls)
    
    # Calculate 3-day win rate
    calls_3d = [c for c in calls if c.get("checked_3d")]
    wins_3d = sum(1 for c in calls_3d if c.get("result_3d") == "WIN")
    win_rate_3d = (wins_3d / len(calls_3d)) * 100 if calls_3d else 0.0
    
    # Calculate 7-day win rate
    calls_7d = [c for c in calls if c.get("checked_7d")]
    wins_7d = sum(1 for c in calls_7d if c.get("result_7d") == "WIN")
    win_rate_7d = (wins_7d / len(calls_7d)) * 100 if calls_7d else 0.0
    
    # Calculate win rate per asset class
    asset_classes = ["stock", "crypto", "forex", "commodity", "bond"]
    class_labels = {
        "stock": "Stocks",
        "crypto": "Crypto",
        "forex": "Forex",
        "commodity": "Commodities",
        "bond": "Bonds"
    }
    
    class_stats = {}
    best_class = "N/A"
    best_class_rate = -1.0
    worst_class = "N/A"
    worst_class_rate = 101.0
    
    for ac in asset_classes:
        ac_calls = [c for c in completed_calls if c.get("asset_class", "").lower() == ac]
        ac_count = len(ac_calls)
        
        ac_wins_3d = sum(1 for c in ac_calls if c.get("result_3d") == "WIN")
        ac_rate_3d = (ac_wins_3d / ac_count) * 100 if ac_count > 0 else 0.0
        
        ac_wins_7d = sum(1 for c in ac_calls if c.get("result_7d") == "WIN")
        ac_rate_7d = (ac_wins_7d / ac_count) * 100 if ac_count > 0 else 0.0
        
        class_stats[ac] = {
            "count": ac_count,
            "rate_3d": ac_rate_3d,
            "rate_7d": ac_rate_7d
        }
        
        if ac_count > 0:
            # use 7d win rate for ranking
            if ac_rate_7d > best_class_rate:
                best_class_rate = ac_rate_7d
                best_class = class_labels[ac]
            if ac_rate_7d < worst_class_rate:
                worst_class_rate = ac_rate_7d
                worst_class = class_labels[ac]
                
    if best_class == "N/A":
        best_class = "None"
        best_class_rate = 0.0
    if worst_class == "N/A":
        worst_class = "None"
        worst_class_rate = 0.0
        
    # 2. Load dynamic global weights
    gw_weights = {}
    gw_win_rates = {}
    gw_history = []
    if os.path.exists(GLOBAL_WEIGHTS_PATH):
        try:
            with open(GLOBAL_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                gw_data = json.load(f)
                gw_weights = gw_data.get("weights", {})
                gw_win_rates = gw_data.get("win_rates", {})
                gw_history = gw_data.get("history", [])
        except Exception:
            pass
            
    # Calculate Signal Effectiveness rows
    signals_list = [
        ("RSI", "rsi"),
        ("MACD", "macd"),
        ("EMA Stack", "ema_stack"),
        ("Volume", "volume"),
        ("BB Position", "bb_position"),
        ("Sentiment", "sentiment"),
        ("Catalyst", "catalyst")
    ]
    
    signals_rows = []
    for label, key in signals_list:
        weight = gw_weights.get(key, 1.0)
        win_rate = gw_win_rates.get(key, 0.5) * 100
        
        # Trend indicator based on weight vs default
        default_w = 2.0 if key in ["rsi", "macd", "ema_stack"] else 1.0
        if weight > default_w:
            trend = "↑"
        elif weight < default_w:
            trend = "↓"
        else:
            trend = "→"
            
        signals_rows.append(f"| {label} | {weight:.1f} | {win_rate:.1f}% | {trend} |")
        
    # Calculate Regime Performance
    regimes = ["RISK-ON", "RISK-OFF", "TRANSITIONING"]
    regime_rows = []
    for r in regimes:
        r_calls = [c for c in completed_calls if c.get("regime_at_call", "").upper() == r]
        r_count = len(r_calls)
        r_wins = sum(1 for c in r_calls if c.get("result_7d") == "WIN")
        r_rate = (r_wins / r_count) * 100 if r_count > 0 else 0.0
        regime_rows.append(f"| {r} | {r_count} | {r_rate:.1f}% |")
        
    # Asset Class Performance Table
    class_rows = []
    for ac in asset_classes:
        stats = class_stats[ac]
        class_rows.append(
            f"| {class_labels[ac]} | {stats['count']} | {stats['rate_3d']:.1f}% | {stats['rate_7d']:.1f}% |"
        )
        
    # Recent Weight Changes
    recent_changes = []
    for h in reversed(gw_history[-5:]):
        ts = h.get("timestamp", "").split("T")[0]
        recent_changes.append(f"- `[{ts}]` {h.get('change', 'No details available')} (basis: {h.get('calls_basis', 0)} calls)")
    if not recent_changes:
        recent_changes.append("- No weight adjustments logged yet.")
        
    # Worst Calls (Learning Log) - Bottom 5 completed calls by outcome
    def get_outcome_pct(c):
        p_at_call = c.get("price_at_call", 1.0)
        p_7d = c.get("price_7d") or c.get("price_3d") or p_at_call
        pct = ((p_7d - p_at_call) / p_at_call) * 100
        # If bearish, invert the move pct
        if c.get("direction") == "bearish":
            pct = -pct
        return pct
        
    worst_calls = sorted(completed_calls, key=get_outcome_pct)[:5]
    worst_calls_rows = []
    for c in worst_calls:
        pct_move = get_outcome_pct(c)
        direction = c.get("direction", "bullish")
        asset = c.get("asset", "")
        price_at_call = c.get("price_at_call", 0.0)
        price_end = c.get("price_7d") or c.get("price_3d") or 0.0
        
        # Smart post-mortem reasoning
        breakdown = c.get("score_breakdown", {})
        strongest_indicator = "None"
        max_val = 0
        for ik, iv in breakdown.items():
            if abs(iv) > max_val:
                max_val = abs(iv)
                strongest_indicator = ik.upper()
                
        reasoning = (
            f"Asset logged a {pct_move:+.2f}% underperformance from call price ${price_at_call:,.2f} to ${price_end:,.2f}. "
            f"The call relied on strong {strongest_indicator} alignment in a {c.get('regime_at_call')} regime, "
            f"but opposing macro/flow headwinds led to invalidation."
        )
        
        worst_calls_rows.append(
            f"#### {asset} ({direction.upper()} call on {c.get('timestamp', '').split('T')[0]})\n"
            f"- **Result:** LOSS ({pct_move:+.2f}% return)\n"
            f"- **Post-Mortem:** {reasoning}\n"
        )
    if not worst_calls_rows:
        worst_calls_rows.append("_No completed losses logged yet._")
        
    # Assemble markdown report
    date_now = datetime.now(IST).strftime("%B %d, %Y")
    
    signals_rows_joined = "\n".join(signals_rows)
    regime_rows_joined = "\n".join(regime_rows)
    class_rows_joined = "\n".join(class_rows)
    recent_changes_joined = "\n".join(recent_changes)
    worst_calls_rows_joined = "\n".join(worst_calls_rows)

    report_md = f"""# Agent Performance Report — {date_now}

## Overall Accuracy

- **Total calls made:** {total_calls}
- **Calls with outcomes:** {completed_count}
- **3-day win rate:** {win_rate_3d:.2f}%
- **7-day win rate:** {win_rate_7d:.2f}%
- **Best performing asset class:** {best_class} ({best_class_rate:.2f}% win rate)
- **Worst performing asset class:** {worst_class} ({worst_class_rate:.2f}% win rate)

## Signal Effectiveness

| Signal | Weight | Win Rate | Trend |
| :--- | :--- | :--- | :--- |
{signals_rows_joined}

## Regime Performance

| Regime | Calls | Win Rate |
| :--- | :--- | :--- |
{regime_rows_joined}

## Asset Class Performance

| Class | Calls | 3d Win Rate | 7d Win Rate |
| :--- | :--- | :--- | :--- |
{class_rows_joined}

## Recent Weight Changes

{recent_changes_joined}

## Worst Calls (Learning Log)

{worst_calls_rows_joined}

---
*Not financial advice. Past agent accuracy does not guarantee future performance.*
"""

    try:
        AGENT_PERFORMANCE_PATH.write_text(report_md, encoding="utf-8")
        logger.info("Written performance report to %s", AGENT_PERFORMANCE_PATH)
    except Exception as exc:
        logger.error("Failed to generate performance report: %s", exc)


# ---------------------------------------------------------------------------
# Module self-test (run directly for smoke test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # --- Minimal smoke-test data ---
    _now = datetime.now(IST)

    _market_data = {
        "global_snapshot": [
            {"ticker": "BTC", "name": "Bitcoin",  "price": 67_450.22, "change_24h": 2.31,  "change_7d": -1.55},
            {"ticker": "ETH", "name": "Ethereum", "price": 3_512.80,  "change_24h": -0.87, "change_7d": 4.21},
            {"ticker": "SPX", "name": "S&P 500",  "price": 5_304.72,  "change_24h": 0.44,  "change_7d": 1.10},
            {"ticker": "DXY", "name": "DXY",      "price": 104.82,    "change_24h": -0.12, "change_7d": 0.33},
            {"ticker": "GLD", "name": "Gold",     "price": 2_315.50,  "change_24h": 0.65,  "change_7d": 2.02},
        ],
        "top_movers": {
            "gainers": [
                {"ticker": "SOL",  "change_24h": 8.45},
                {"ticker": "AVAX", "change_24h": 6.12},
            ],
            "losers": [
                {"ticker": "DOGE", "change_24h": -5.22},
                {"ticker": "XRP",  "change_24h": -3.10},
            ],
        },
    }

    _macro_state = {
        "regime": "Risk-ON / Moderate Inflation",
        "vix": 17.32,
        "dxy": 104.82,
        "yield_curve_inverted": False,
        "yield_curve_spread": 18.5,
        "risk_score": 6.5,
        "fear_greed_index": 62,
        "fear_greed_label": "Greed",
        "btc_dominance": 54.2,
        "alt_season_index": 41,
        "equity_mood": "Bullish",
        "spx_trend": "Uptrend",
        "global_risk_on": True,
        "fed_stance": "Hold",
        "next_fed_meeting": "2025-06-12",
        "hike_probability": 3.5,
        "cut_probability": 28.7,
    }

    _treasury_data = {
        "2y": 4.82, "5y": 4.41, "10y": 4.52, "30y": 4.68,
    }

    _fred_data = {
        "cpi": 3.4, "pce": 2.7, "unemployment": 3.9,
        "gdp": 1.6, "m2": 2.1, "fed_funds": 5.33,
    }

    _calendar = [
        {
            "date": "2025-06-04", "time": "20:30", "event": "JOLTS Job Openings",
            "currency": "USD", "impact": "high",
            "forecast": "8.35M", "previous": "8.48M",
            "expected_market_impact": "Lower reading → dovish Fed narrative → risk-on",
        },
        {
            "date": "2025-06-06", "time": "18:00", "event": "US Non-Farm Payrolls",
            "currency": "USD", "impact": "high",
            "forecast": "185K", "previous": "175K",
            "expected_market_impact": "Beat → USD strength, equity sell-off; miss → rate cut hopes",
        },
        {
            "date": "2025-06-05", "time": "13:30", "event": "ECB Rate Decision",
            "currency": "EUR", "impact": "high",
            "forecast": "4.25%", "previous": "4.50%",
            "expected_market_impact": "Cut expected; EUR sell-off on announcement",
        },
    ]

    _research_data = {
        "headlines": [
            {
                "title": "Fed signals patience on rate cuts amid sticky inflation",
                "source": "Reuters",
                "impact": "Reduces near-term cut expectations; USD supportive.",
            },
            {
                "title": "Bitcoin ETF inflows hit weekly record of $1.2B",
                "source": "Bloomberg",
                "impact": "Strong institutional demand; bullish for BTC short-term.",
            },
            {
                "title": "China PMI data disappoints, stoking global growth fears",
                "source": "WSJ",
                "impact": "Risk-off pressure on commodity-linked assets.",
            },
        ],
        "risks": [
            "Sticky US core inflation may delay Fed cuts beyond Q3 2025",
            "Geopolitical tensions in Middle East elevating oil supply risk",
            "China property sector stress continues to weigh on EM sentiment",
        ],
        "events_this_week": [
            {"date": "2025-06-04", "event": "JOLTS Job Openings",   "currency": "USD"},
            {"date": "2025-06-05", "event": "ECB Rate Decision",     "currency": "EUR"},
            {"date": "2025-06-06", "event": "Non-Farm Payrolls",     "currency": "USD"},
        ],
        "treasury_data": _treasury_data,
        "fred_data":     _fred_data,
        "calendar":      _calendar,
    }

    _enriched_data = {
        "crypto": [
            {
                "ticker": "BTC",  "name": "Bitcoin",
                "price": 67_450.22, "change_24h": 2.31, "change_7d": -1.55,
                "rsi": 58.4, "macd_signal": "Bullish", "ema_stack": "Bullish",
                "bb_position": "Upper", "score": 7.2,
                "alerts": ["Near ATH resistance"],
            },
            {
                "ticker": "ETH", "name": "Ethereum",
                "price": 3_512.80, "change_24h": -0.87, "change_7d": 4.21,
                "rsi": 52.1, "macd_signal": "Neutral", "ema_stack": "Bullish",
                "bb_position": "Mid", "score": 5.5,
                "alerts": [],
            },
        ],
        "stocks": [
            {
                "ticker": "NVDA", "name": "NVIDIA Corp",
                "price": 874.50, "change_24h": 3.22, "change_7d": 11.5,
                "rsi": 71.2, "macd_signal": "Bullish", "ema_stack": "Bullish",
                "bb_position": "Upper", "score": 8.1,
                "alerts": ["Overbought RSI", "Earnings beat"],
            },
        ],
    }

    _opportunities = [
        {
            "ticker": "BTC", "name": "Bitcoin", "asset_class": "Crypto",
            "price": 67_450.22, "change_24h": 2.31,
            "score": 7.2,
            "score_breakdown": "Trend +2, Momentum +2, Macro +1.5, Sentiment +1.7",
            "support": 64_000.00, "resistance": 72_000.00,
            "reasoning": (
                "BTC broke above the 200-day EMA with strong volume. "
                "ETF inflows are accelerating and the halving supply shock is being priced in. "
                "Macro conditions remain supportive with a weakening DXY."
            ),
            "catalyst": "Bitcoin halving supply squeeze + ETF demand surge",
            "invalidation_level": 61_500.00,
            "new_high": False,
            "bb_squeeze": False,
        },
        {
            "ticker": "NVDA", "name": "NVIDIA Corp", "asset_class": "US Equities",
            "price": 874.50, "change_24h": 3.22,
            "score": 8.1,
            "score_breakdown": "Trend +3, Momentum +2, Earnings +2, Sector +1.1",
            "support": 820.00, "resistance": 950.00,
            "reasoning": (
                "NVIDIA reported a massive earnings beat driven by data center demand. "
                "AI capex spend from hyperscalers remains at record levels with no signs of slowdown. "
                "Technicals show a clean cup-and-handle breakout on the weekly chart."
            ),
            "catalyst": "AI infrastructure spending cycle; upcoming GTC developer conference",
            "invalidation_level": 810.00,
            "new_high": True,
            "bb_squeeze": False,
        },
        {
            "ticker": "XRP", "name": "Ripple XRP", "asset_class": "Crypto",
            "price": 0.521345, "change_24h": -3.10,
            "score": -6.5,
            "score_breakdown": "Trend -2, Momentum -2, Legal -1.5, Sentiment -1",
            "support": 0.48, "resistance": 0.58,
            "reasoning": (
                "SEC litigation overhang continues to suppress institutional adoption. "
                "Technicals show a bear flag forming below the 50-day EMA. "
                "Volume on down-days significantly outpaces up-days."
            ),
            "catalyst": "Pending SEC ruling; BTC correlation risk",
            "invalidation_level": 0.59,
            "new_high": False,
            "bb_squeeze": False,
        },
    ]

    _alerts = [
        "BTC: Near ATH resistance zone ($72K). Watch for breakout or rejection.",
        "NVDA: RSI overbought (71.2). Consider scaling out 25% of position.",
        "VIX: Elevated spike risk into NFP Friday — reduce leverage.",
    ]

    generate_all_reports(
        market_data   = _market_data,
        enriched_data = _enriched_data,
        opportunities = _opportunities,
        macro_state   = _macro_state,
        research_data = _research_data,
        alerts        = _alerts,
        config        = {},
    )

    print("\n✅ Smoke test complete. Check reports/ and state/ directories.")
    print(f"   format_pct(3.42)  → {format_pct(3.42)}")
    print(f"   format_pct(-1.23) → {format_pct(-1.23)}")
    print(f"   format_price(67450.22) → {format_price(67450.22)}")
    print(f"   format_price(0.000531) → {format_price(0.000531)}")
    print(f"   format_stars(8.5) → {format_stars(8.5)}")
    print(f"   format_stars(5.0) → {format_stars(5.0)}")
    print(f"   format_stars(2.0) → {format_stars(2.0)}")
