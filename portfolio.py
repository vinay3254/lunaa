"""
portfolio.py
============
Autonomous Trading Research Agent — Portfolio Tracking Module

Calculates:
  - Position P&L ($ and %)
  - Proximity to Stop Loss (flags RED warning if within 2%)
  - Proximity to Take Profit (flags GREEN if within 2%)
  - Days held
  - Max drawdown from peak since entry (using OHLCV if available)
  - Risk-to-Reward (R:R) ratio: (Take Profit - Entry) / (Entry - Stop Loss)
  - Total portfolio value, unrealized P&L ($ and %)
  - Average R:R of open positions
  - Win rate of closed positions
  - Portfolio Heat (capital-at-risk percentage)
  - P0 Critical alerts (SL/TP hit)
  - Warnings (Stale trades > 30 days, High Portfolio Heat > 20%)
  - VIX banner if VIX > 30

Generates: reports/portfolio-status.md
"""

from __future__ import annotations

import json
import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

# Path configurations
BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
STATE_DIR = BASE_DIR / "state"
PORTFOLIO_PATH = BASE_DIR / "portfolio.json"
LAST_RUN_PATH = STATE_DIR / "last-run.json"
PORTFOLIO_REPORT_PATH = REPORTS_DIR / "portfolio-status.md"

logger = logging.getLogger("portfolio")
logger.setLevel(logging.INFO)

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def load_portfolio() -> dict:
    """Load portfolio register from portfolio.json."""
    if not PORTFOLIO_PATH.exists():
        logger.warning("Portfolio file not found at %s. Creating empty template.", PORTFOLIO_PATH)
        default_portfolio = {
            "positions": [],
            "closed_positions": [],
            "cash_balance": 0.00,
            "currency": "USD"
        }
        try:
            PORTFOLIO_PATH.write_text(json.dumps(default_portfolio, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error("Failed to write default portfolio: %s", e)
        return default_portfolio

    try:
        with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load portfolio.json: %s", e)
        return {"cash_balance": 0.0, "currency": "USD", "positions": []}


def save_portfolio(portfolio: dict) -> None:
    """Save portfolio register to portfolio.json."""
    try:
        with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
            json.dump(portfolio, f, indent=2)
    except Exception as e:
        logger.error("Failed to save portfolio.json: %s", e)


def get_current_price(asset: str, market_data: dict) -> float:
    """Robustly extract current asset price from market data snapshot."""
    # 1. Search in traditional
    traditional = market_data.get("traditional", {})
    if asset in traditional:
        price = traditional[asset].get("price")
        if price is not None and not np.isnan(price):
            return float(price)

    # 2. Search in crypto
    crypto = market_data.get("crypto", {})
    if asset in crypto:
        price = crypto[asset].get("price")
        if price is not None and not np.isnan(price):
            return float(price)

    # 3. Check case-insensitive
    asset_upper = asset.upper()
    if asset_upper in traditional:
        price = traditional[asset_upper].get("price")
        if price is not None and not np.isnan(price):
            return float(price)
    if asset_upper in crypto:
        price = crypto[asset_upper].get("price")
        if price is not None and not np.isnan(price):
            return float(price)

    # 4. Strip -USD for crypto checks
    if asset_upper.endswith("-USD"):
        base_crypto = asset_upper[:-4]
        if base_crypto in crypto:
            price = crypto[base_crypto].get("price")
            if price is not None and not np.isnan(price):
                return float(price)

    # 5. Add -USD for traditional checks
    if f"{asset_upper}-USD" in traditional:
        price = traditional[f"{asset_upper}-USD"].get("price")
        if price is not None and not np.isnan(price):
            return float(price)

    # 6. Fallback: Check global_snapshot list
    global_snap = market_data.get("global_snapshot", [])
    if isinstance(global_snap, list):
        for item in global_snap:
            if isinstance(item, dict) and item.get("ticker") in (asset, asset_upper, f"{asset_upper}-USD"):
                price = item.get("price")
                if price is not None and not np.isnan(price):
                    return float(price)

    return float("nan")


def fetch_price_fallback(asset: str, silent: bool = False) -> float:
    """Fetch price in real-time as a fallback if not found in state."""
    try:
        import yfinance as yf
        if not silent:
            logger.info("Price for %s not found in state, performing real-time fallback fetch...", asset)
        # Attempt yfinance
        ticker = yf.Ticker(asset)
        history = ticker.history(period="1d")
        if not history.empty:
            return float(history["Close"].iloc[-1])

        # Try appending -USD for crypto
        ticker_crypto = yf.Ticker(f"{asset}-USD")
        history_crypto = ticker_crypto.history(period="1d")
        if not history_crypto.empty:
            return float(history_crypto["Close"].iloc[-1])
    except Exception as e:
        if not silent:
            logger.error("Fallback fetch failed for %s: %s", asset, e)
    return float("nan")


def calculate_max_drawdown_ohlcv(asset: str, entry_date_str: str, current_price: float, market_data: dict) -> float:
    """
    Compute max drawdown from peak price since entry using OHLCV history.
    """
    try:
        entry_dt = datetime.fromisoformat(entry_date_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        try:
            entry_dt = datetime.strptime(entry_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            return 0.0

    # 1. Locate OHLCV in market data
    ohlcv = None
    traditional = market_data.get("traditional", {})
    crypto = market_data.get("crypto", {})

    asset_upper = asset.upper()
    asset_keys = [asset, asset_upper, asset_upper.replace("-USD", ""), f"{asset_upper}-USD"]

    for key in asset_keys:
        if key in traditional and "ohlcv" in traditional[key]:
            ohlcv = traditional[key]["ohlcv"]
            break
        if key in crypto and "ohlcv" in crypto[key]:
            ohlcv = crypto[key]["ohlcv"]
            break

    # If OHLCV is a standard pandas DataFrame, we can use it directly
    if ohlcv is not None and isinstance(ohlcv, pd.DataFrame) and not ohlcv.empty:
        try:
            # Filter index since entry date
            ohlcv_filtered = ohlcv[ohlcv.index >= entry_dt]
            if not ohlcv_filtered.empty:
                peak_price = float(ohlcv_filtered["high"].max())
                if np.isnan(peak_price) or peak_price <= 0:
                    peak_price = float(ohlcv_filtered["close"].max())
                
                # Check if current price is higher than peak
                if current_price > peak_price:
                    peak_price = current_price
                
                if peak_price > 0:
                    return float((peak_price - current_price) / peak_price * 100)
        except Exception as e:
            logger.debug("Failed to calculate drawdown using OHLCV df for %s: %s", asset, e)

    # 2. Fallback to yfinance history if not found in state
    try:
        import yfinance as yf
        ticker_symbol = asset
        if asset_upper.endswith("-USD") or asset_upper in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "LINK", "DOT", "UNI"]:
            # Ensure yfinance-compatible crypto symbol
            ticker_symbol = asset_upper if asset_upper.endswith("-USD") else f"{asset_upper}-USD"
            
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(start=entry_dt.strftime("%Y-%m-%d"))
        if not df.empty:
            peak = float(df["High"].max())
            if current_price > peak:
                peak = current_price
            if peak > 0:
                return float((peak - current_price) / peak * 100)
    except Exception as e:
        logger.debug("Fallback drawdown history fetch failed for %s: %s", asset, e)

    # 3. Simple drawdown from current vs entry if no history
    if current_price < 0:
        return 0.0
    return 0.0


def calculate_portfolio_status(market_data: dict | None = None, silent: bool = False) -> dict:
    """
    Main calculation engine for portfolio tracker.
    Returns calculated metrics and updates report.
    """
    portfolio = load_portfolio()
    cash = float(portfolio.get("cash_balance", 0.0))
    currency = portfolio.get("currency", "USD")
    positions = portfolio.get("positions", [])

    # Load market data if not provided
    if market_data is None:
        market_data = {}
        if LAST_RUN_PATH.exists():
            try:
                with open(LAST_RUN_PATH, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                    market_data = state_data.get("market_data", {})
            except Exception as e:
                if not silent:
                    logger.error("Failed to load last-run.json state: %s", e)

    # 1. Fetch VIX for risk banners
    vix_val = float("nan")
    # Search VIX in state
    global_snap = market_data.get("global_snapshot", [])
    if isinstance(global_snap, list):
        for item in global_snap:
            if isinstance(item, dict) and item.get("ticker") == "^VIX":
                vix_val = float(item.get("price", float("nan")))
                break

    # If VIX is NaN, try real-time fetch fallback
    if np.isnan(vix_val):
        try:
            from market_data import fetch_vix_data_with_fallbacks
            vix_df = fetch_vix_data_with_fallbacks()
            vix_val = float(vix_df["Close"].iloc[-1])
        except Exception:
            pass

    open_positions_results = []
    closed_positions_results = []

    total_unrealized_pnl = 0.0
    total_cost_basis = 0.0
    open_positions_value = 0.0
    total_capital_at_risk = 0.0
    total_rr_open = 0.0
    rr_count = 0

    closed_wins = 0
    closed_losses = 0

    alerts = []
    warnings = []

    now = datetime.now(timezone.utc)

    for pos in positions:
        asset = pos.get("asset", "Unknown")
        qty = float(pos.get("quantity", 0.0))
        entry_price = float(pos.get("entry_price", 0.0))
        status = pos.get("status", "open").lower()
        uuid = pos.get("uuid", "")

        # Days held
        entry_date_str = pos.get("entry_date", "")
        try:
            entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_held = (now - entry_date).days
        except Exception:
            days_held = 0

        if status == "open":
            # Current Price lookup
            curr_price = get_current_price(asset, market_data)
            is_stale = False
            if np.isnan(curr_price):
                curr_price = fetch_price_fallback(asset, silent=silent)
                if np.isnan(curr_price):
                    curr_price = entry_price  # Fallback to entry
                    is_stale = True

            # Position calculations
            cost_basis = entry_price * qty
            market_value = curr_price * qty
            pnl_dollars = market_value - cost_basis
            pnl_pct = (pnl_dollars / cost_basis * 100) if cost_basis > 0 else 0.0

            total_unrealized_pnl += pnl_dollars
            total_cost_basis += cost_basis
            open_positions_value += market_value

            # Risk parameters
            sl = pos.get("stop_loss")
            tp = pos.get("take_profit")
            sl_val = float(sl) if sl is not None else float("nan")
            tp_val = float(tp) if tp is not None else float("nan")

            dist_sl_pct = float("nan")
            dist_tp_pct = float("nan")

            if not np.isnan(sl_val) and curr_price > 0:
                dist_sl_pct = ((curr_price - sl_val) / curr_price) * 100
                capital_risk = (entry_price - sl_val) * qty
                if capital_risk > 0:
                    total_capital_at_risk += capital_risk

            if not np.isnan(tp_val) and curr_price > 0:
                dist_tp_pct = ((tp_val - curr_price) / curr_price) * 100

            # Proximity flags
            is_near_sl = False
            is_near_tp = False
            if not np.isnan(dist_sl_pct) and dist_sl_pct <= 2.0:
                is_near_sl = True
            if not np.isnan(dist_tp_pct) and dist_tp_pct <= 2.0:
                is_near_tp = True

            # Stop loss / Take profit hit check
            if not np.isnan(sl_val) and curr_price <= sl_val:
                alerts.append(f"🔴 **CRITICAL (P0): Stop Loss Hit** for {asset}. Current Price {curr_price:.2f} <= SL {sl_val:.2f}!")
            elif not np.isnan(tp_val) and curr_price >= tp_val:
                alerts.append(f"🟢 **CRITICAL (P0): Take Profit Hit** for {asset}. Current Price {curr_price:.2f} >= TP {tp_val:.2f}!")

            # Risk-to-Reward (R:R) ratio
            rr_ratio = float("nan")
            if not np.isnan(sl_val) and not np.isnan(tp_val):
                risk_amt = entry_price - sl_val
                reward_amt = tp_val - entry_price
                if risk_amt > 0:
                    rr_ratio = reward_amt / risk_amt
                    total_rr_open += rr_ratio
                    rr_count += 1

            # Max Drawdown from peak since entry
            max_dd = calculate_max_drawdown_ohlcv(asset, entry_date_str, curr_price, market_data)

            # Stale Trade Check (>30 days held and within +/- 5% range)
            if days_held > 30 and abs(pnl_pct) <= 5.0:
                warnings.append(f"⏳ **Stale Trade Warning:** {asset} held for {days_held} days with low volatility ({pnl_pct:+.2f}%). Consider rotation.")

            open_positions_results.append({
                "uuid": uuid,
                "asset": asset,
                "qty": qty,
                "entry_price": entry_price,
                "current_price": curr_price,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "pnl_dollars": pnl_dollars,
                "pnl_pct": pnl_pct,
                "days_held": days_held,
                "max_drawdown": max_dd,
                "stop_loss": sl_val,
                "distance_to_sl": dist_sl_pct,
                "is_near_sl": is_near_sl,
                "take_profit": tp_val,
                "distance_to_tp": dist_tp_pct,
                "is_near_tp": is_near_tp,
                "rr_ratio": rr_ratio,
                "is_stale": is_stale
            })

        elif status == "closed":
            exit_price = float(pos.get("exit_price", 0.0))
            cost_basis = entry_price * qty
            exit_basis = exit_price * qty
            pnl_dollars = exit_basis - cost_basis
            pnl_pct = (pnl_dollars / cost_basis * 100) if cost_basis > 0 else 0.0

            if pnl_dollars > 0:
                closed_wins += 1
            else:
                closed_losses += 1

            exit_date_str = pos.get("exit_date", "")
            try:
                entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                exit_date = datetime.strptime(exit_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                days_held = (exit_date - entry_date).days
            except Exception:
                days_held = 0

            closed_positions_results.append({
                "uuid": uuid,
                "asset": asset,
                "qty": qty,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_dollars": pnl_dollars,
                "pnl_pct": pnl_pct,
                "days_held": days_held,
                "entry_date": entry_date_str,
                "exit_date": exit_date_str
            })

    # Portfolio level aggregates
    total_portfolio_value = cash + open_positions_value
    unrealized_pnl_pct = (total_unrealized_pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0.0

    avg_rr = (total_rr_open / rr_count) if rr_count > 0 else 0.0
    total_closed = closed_wins + closed_losses
    win_rate = (closed_wins / total_closed * 100) if total_closed > 0 else 0.0

    portfolio_heat = (total_capital_at_risk / total_portfolio_value * 100) if total_portfolio_value > 0 else 0.0

    # Heat Warning
    if portfolio_heat > 20.0:
        warnings.append(f"🔥 **HIGH PORTFOLIO HEAT ALERT:** Capital at risk is {portfolio_heat:.2f}%, exceeding the maximum 20.00% safety threshold. Reduce position sizes or tighten stop losses!")

    # Format Markdown
    reports_dir_created = False
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        reports_dir_created = True
    except Exception:
        pass

    # Build Markdown string
    md = []
    md.append(f"# LUNA Investment Portfolio Tracker — Status Report")
    md.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Local*")
    md.append("")

    # --- Volatility regimes and banners ---
    if not np.isnan(vix_val) and vix_val > 30.0:
        md.append(f"> 🔴 **CRITICAL WARNING: HIGH VOLATILITY REGIME (VIX > 30)** — VIX is currently **{vix_val:.2f}**. Extreme market stress detected. Protect capital and minimize risk exposure.")
        md.append("")

    # --- Warnings & Alerts ---
    if alerts or warnings:
        md.append("## ⚠️ System Alerts & Risk Warnings")
        md.append("")
        for a in alerts:
            md.append(f"- {a}")
        for w in warnings:
            md.append(f"- {w}")
        md.append("")

    # --- Summary metrics ---
    md.append("## 📊 Portfolio Summary")
    md.append("")
    md.append(f"| Metric | Value | Details / Status |")
    md.append(f"| :--- | :--- | :--- |")
    md.append(f"| **Total Portfolio Value** | **${total_portfolio_value:,.2f}** | Cash + Open Market Value |")
    md.append(f"| **Cash Balance** | ${cash:,.2f} | Available Buying Power |")
    md.append(f"| **Total Cost Basis** | ${total_cost_basis:,.2f} | Principal Invested |")
    
    pnl_emoji = "🟢" if total_unrealized_pnl >= 0 else "🔴"
    pnl_sign = "+" if total_unrealized_pnl >= 0 else ""
    md.append(f"| **Unrealized P&L ($)** | {pnl_emoji} **{pnl_sign}${total_unrealized_pnl:,.2f}** | Cumulative Open Position Return |")
    md.append(f"| **Unrealized P&L (%)** | {pnl_emoji} **{pnl_sign}{unrealized_pnl_pct:.2f}%** | Percentage Return on Cost |")
    md.append(f"| **Average Risk-to-Reward (R:R)** | **{avg_rr:.2f}:1** | Targets vs Stop-Loss ratios |")
    
    heat_status = "🟢 Safe (<10%)" if portfolio_heat < 10 else ("🟡 Moderate (10-20%)" if portfolio_heat <= 20 else "🔴 High (>20%)")
    md.append(f"| **Portfolio Heat (Risk %)** | **{portfolio_heat:.2f}%** | {heat_status} |")
    md.append(f"| **Closed Trades Win Rate** | **{win_rate:.2f}%** | Basis: {total_closed} settled positions |")
    md.append("")

    # --- Open Positions Table ---
    md.append("## 💼 Active Holdings & Open Positions")
    md.append("")
    if not open_positions_results:
        md.append("_No active holdings found in portfolio register._")
        md.append("")
    else:
        md.append("| Asset | Price (Current / Entry) | Qty | Cost Basis | Market Value | P&L ($ / %) | Days Held | Drawdown | Proximity to SL / TP | Status |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for o in open_positions_results:
            pnl_col_emoji = "🟢" if o["pnl_dollars"] >= 0 else "🔴"
            pnl_col_sign = "+" if o["pnl_dollars"] >= 0 else ""
            
            pnl_cell = f"{pnl_col_emoji} {pnl_col_sign}${o['pnl_dollars']:,.2f}<br>({pnl_col_sign}{o['pnl_pct']:.2f}%)"
            price_cell = f"**${o['current_price']:,.2f}**<br>(${o['entry_price']:,.2f})"
            
            sl_cell = f"SL: ${o['stop_loss']:,.2f}" if not np.isnan(o["stop_loss"]) else "SL: None"
            tp_cell = f"TP: ${o['take_profit']:,.2f}" if not np.isnan(o["take_profit"]) else "TP: None"
            
            # SL TP highlights
            sl_display = sl_cell
            if o["is_near_sl"]:
                sl_display = f"⚠️ <font color='red'>**{sl_cell}** (within 2%)</font>"
                
            tp_display = tp_cell
            if o["is_near_tp"]:
                tp_display = f"🎉 <font color='green'>**{tp_cell}** (within 2%)</font>"
                
            prox_cell = f"{sl_display}<br>{tp_display}"
            
            status_cell = "🔴 Stale" if o["is_stale"] else "🟢 Active"
            
            md.append(f"| **{o['asset']}** | {price_cell} | {o['qty']:,} | ${o['cost_basis']:,.2f} | ${o['market_value']:,.2f} | {pnl_cell} | {o['days_held']}d | {o['max_drawdown']:.2f}% | {prox_cell} | {status_cell} |")
        md.append("")

    # --- Closed Positions Table ---
    md.append("## 📜 Settled & Closed Trades")
    md.append("")
    if not closed_positions_results:
        md.append("_No settled trades recorded in historical register._")
        md.append("")
    else:
        md.append("| Asset | Quantity | Entry Price | Exit Price | P&L ($ / %) | Days Held | Period |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for c in closed_positions_results:
            pnl_col_emoji = "🟢" if c["pnl_dollars"] >= 0 else "🔴"
            pnl_col_sign = "+" if c["pnl_dollars"] >= 0 else ""
            
            pnl_cell = f"{pnl_col_emoji} {pnl_col_sign}${c['pnl_dollars']:,.2f} ({pnl_col_sign}{c['pnl_pct']:.2f}%)"
            period_cell = f"{c['entry_date']} to {c['exit_date']}"
            
            md.append(f"| **{c['asset']}** | {c['qty']:,} | ${c['entry_price']:,.2f} | ${c['exit_price']:,.2f} | {pnl_cell} | {c['days_held']}d | {period_cell} |")
        md.append("")

    md.append("---")
    md.append("*Disclaimer: All valuations and P&L figures are simulated based on pricing datasets. Past metrics are not indicative of future performance.*")

    report_content = "\n".join(md)

    if reports_dir_created:
        try:
            PORTFOLIO_REPORT_PATH.write_text(report_content, encoding="utf-8")
            if not silent:
                logger.info("Written portfolio status report to %s", PORTFOLIO_REPORT_PATH)
        except Exception as e:
            if not silent:
                logger.error("Failed to write portfolio status report: %s", e)
    
    return {
        "total_portfolio_value": total_portfolio_value,
        "cash_balance": cash,
        "total_unrealized_pnl": total_unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "avg_rr": avg_rr,
        "portfolio_heat": portfolio_heat,
        "win_rate": win_rate,
        "alerts_count": len(alerts),
        "warnings_count": len(warnings)
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    logger.info("Starting standalone portfolio analysis pass...")
    results = calculate_portfolio_status()
    print("\n--- Portfolio Analysis Complete ---")

    # Empty-state guard: show friendly message instead of zeroed-out fake numbers
    portfolio_raw = load_portfolio()
    has_open = any(
        p.get("status", "open").lower() == "open"
        for p in portfolio_raw.get("positions", [])
    )
    if not has_open and results['total_portfolio_value'] == 0.0:
        print("\n  No positions tracked yet. Add your positions to portfolio.json to begin tracking.")
        print(f"  Report written to: {PORTFOLIO_REPORT_PATH}")
    else:
        print(f"Total Value:     ${results['total_portfolio_value']:,.2f}")
        print(f"Unrealized P&L:  ${results['total_unrealized_pnl']:,.2f} ({results['unrealized_pnl_pct']:.2f}%)")
        print(f"Portfolio Heat:  {results['portfolio_heat']:.2f}%")
        print(f"Win Rate:        {results['win_rate']:.2f}%")
        print(f"P0 Alerts count: {results['alerts_count']}")
        print(f"Warnings count:  {results['warnings_count']}")
        print(f"Report written to: {PORTFOLIO_REPORT_PATH}")
