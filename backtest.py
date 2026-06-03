"""
backtest.py
===========
LUNA Autonomous Trading Agent — High-Fidelity Backtesting & Simulation Engine

Validates signal performance historically over 90 to 180 days.
Calculates win rates, average returns, Sharpe ratios, and alpha vs Buy-and-Hold.
Persists performance records to state/backtest-performance.json.
Provides an auto-suppression mechanism to block signals with < 45% win rate.
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import indicators
import scanner

logger = logging.getLogger("backtest")
logger.setLevel(logging.INFO)

console = Console()

PERFORMANCE_PATH = "state/backtest-performance.json"

# ---------------------------------------------------------------------------
# Load & Save Backtest Performance Cache
# ---------------------------------------------------------------------------

def load_backtest_performance() -> dict:
    """Load backtest performance records from state/backtest-performance.json."""
    if not os.path.exists(PERFORMANCE_PATH):
        return {}
    try:
        with open(PERFORMANCE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load backtest performance cache: %s", exc)
        return {}


def save_backtest_performance(perf: dict) -> None:
    """Save backtest performance records to state/backtest-performance.json."""
    os.makedirs(os.path.dirname(PERFORMANCE_PATH), exist_ok=True)
    try:
        with open(PERFORMANCE_PATH, "w", encoding="utf-8") as f:
            json.dump(perf, f, indent=2)
    except Exception as exc:
        logger.error("Failed to save backtest performance cache: %s", exc)


def is_signal_suppressed(ticker: str, direction: str) -> bool:
    """Return True if the ticker has a historical win rate < 45% for the given direction."""
    perf = load_backtest_performance()
    ticker_perf = perf.get(ticker, {})
    if not ticker_perf:
        return False
        
    stats = ticker_perf.get(direction, {})
    win_rate = stats.get("win_rate", 1.0)
    total_signals = stats.get("total_signals", 0)
    
    # Only suppress if we have at least 3 historical signals to avoid premature suppression
    if total_signals >= 3 and win_rate < 0.45:
        logger.warning("AUTO-SUPPRESSION: %s %s signal blocked. Historical win rate is %.2f%% (< 45%%).", 
                       ticker, direction.upper(), win_rate * 100)
        return True
    return False


# ---------------------------------------------------------------------------
# Backtest Run Engine
# ---------------------------------------------------------------------------

def run_backtest(target_asset: str, days: int = 90) -> dict | None:
    """Simulate scans historically over *days* for the target asset."""
    console.print(f"\n[bold cyan]=== RUNNING LUNA HISTORICAL BACKTEST: {target_asset} ({days} days) ===[/bold cyan]")
    
    # Download extra history buffer (e.g. +70 trading days) to ensure EMAs and MACD calculate correctly
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days + 110)
    
    try:
        df = yf.download(target_asset, start=start_date, end=end_date, interval="1d", progress=False, auto_adjust=True)
    except Exception as exc:
        console.print(f"[bold red]Failed to download yfinance data for {target_asset}: {exc}[/bold red]")
        return None

    if df is None or df.empty or len(df) < 60:
        console.print(f"[bold red]Insufficient daily price candles downloaded for {target_asset} ({len(df) if df is not None else 0} rows).[/bold red]")
        return None

    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"adj close": "close"})
    
    # Ensure volume and all columns exist
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = float("nan")
            
    df = df[["open", "high", "low", "close", "volume"]].dropna().copy()
    
    # Locate starting index corresponding to approx *days* ago
    # Find the row index closest to start of backtest window
    cutoff_date = (end_date - timedelta(days=days)).date()
    start_idx = 50 # absolute minimum buffer
    for idx, dt in enumerate(df.index):
        if dt.date() >= cutoff_date:
            start_idx = max(50, idx)
            break

    signals = []
    
    # Loop historically through backtest dates
    for idx in range(start_idx, len(df) - 7):
        hist_slice = df.iloc[:idx+1].copy()
        date_t = hist_slice.index[-1]
        price_t = hist_slice["close"].iloc[-1]
        
        # Build mock macro state
        macro_state = {
            "regime": "RISK-ON",
            "vix": 15.0,
            "dxy_change_7d_pct": 0.0,
            "btc_dominance": 52.0,
            "spy_7d_return": 0.0,
            "yield_10y": 4.0
        }

        # Calculate standard indicators
        try:
            asset_indicator_data = indicators.calculate_all_indicators({
                "symbol": target_asset,
                "ohlcv": hist_slice,
                "price": price_t
            })
            asset_indicator_data["asset_class"] = "stock" if "-USD" not in target_asset else "crypto"
            asset_indicator_data["ticker"] = target_asset
            asset_indicator_data["price"] = price_t
        except Exception:
            continue

        # Score the asset
        # Note: we bypass model recording during backtesting to keep feature store clean
        scored = scanner.score_asset(asset_indicator_data, [], macro_state=macro_state)
        score = scored["score"]
        
        # Bullish signal (Score >= 6), Bearish signal (Score <= -6)
        direction = None
        if score >= 6:
            direction = "bullish"
        elif score <= -6:
            direction = "bearish"
            
        if direction:
            # Look ahead for outcomes
            # Price 3 days later
            price_3d = float(df["close"].iloc[idx+3])
            ret_3d = (price_3d - price_t) / price_t * 100.0
            
            # Price 7 days later
            price_7d = float(df["close"].iloc[idx+7])
            ret_7d = (price_7d - price_t) / price_t * 100.0
            
            signals.append({
                "date": date_t.strftime("%Y-%m-%d"),
                "direction": direction,
                "score": score,
                "entry_price": price_t,
                "price_3d": price_3d,
                "price_7d": price_7d,
                "return_3d": ret_3d,
                "return_7d": ret_7d,
                "win_3d": ret_3d > 2.0 if direction == "bullish" else ret_3d < -2.0,
                "win_7d": ret_7d > 2.0 if direction == "bullish" else ret_7d < -2.0,
            })

    # Calculate Benchmark Buy & Hold performance
    bh_start_price = float(df["close"].iloc[start_idx])
    bh_end_price = float(df["close"].iloc[-1])
    bh_return = (bh_end_price - bh_start_price) / bh_start_price * 100.0

    # Consolidate backtest metrics
    bull_signals = [s for s in signals if s["direction"] == "bullish"]
    bear_signals = [s for s in signals if s["direction"] == "bearish"]
    
    def _stats_for_direction(sig_list, is_bullish):
        if not sig_list:
            return {"total_signals": 0, "win_rate": 0.0, "avg_return": 0.0, "wins": 0, "losses": 0}
        
        wins = sum(1 for s in sig_list if s["win_7d"])
        losses = len(sig_list) - wins
        win_rate = wins / len(sig_list)
        avg_ret = np.mean([s["return_7d"] for s in sig_list])
        
        return {
            "total_signals": len(sig_list),
            "win_rate": win_rate,
            "avg_return": avg_ret,
            "wins": wins,
            "losses": losses
        }

    bull_stats = _stats_for_direction(bull_signals, True)
    bear_stats = _stats_for_direction(bear_signals, False)

    # Best and Worst signal
    best_sig = None
    worst_sig = None
    if signals:
        best_sig = max(signals, key=lambda s: s["return_7d"] if s["direction"] == "bullish" else -s["return_7d"])
        worst_sig = min(signals, key=lambda s: s["return_7d"] if s["direction"] == "bullish" else -s["return_7d"])

    # Sharpe ratio proxy on signal returns
    returns = [s["return_7d"] if s["direction"] == "bullish" else -s["return_7d"] for s in signals]
    sharpe = 0.0
    if len(returns) > 1:
        std = np.std(returns)
        sharpe = (np.mean(returns) / std * np.sqrt(252 / 7)) if std != 0 else 0.0

    # Alpha generation
    avg_fwd_ret = np.mean(returns) if returns else 0.0
    alpha = avg_fwd_ret - bh_return if returns else 0.0

    # Save to local performance cache
    perf_cache = load_backtest_performance()
    perf_cache[target_asset] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "days": days,
        "buy_and_hold_return": bh_return,
        "total_signals": len(signals),
        "sharpe_ratio": sharpe,
        "alpha": alpha,
        "bullish": {
            "total_signals": bull_stats["total_signals"],
            "win_rate": bull_stats["win_rate"],
            "avg_return": bull_stats["avg_return"]
        },
        "bearish": {
            "total_signals": bear_stats["total_signals"],
            "win_rate": bear_stats["win_rate"],
            "avg_return": bear_stats["avg_return"]
        }
    }
    save_backtest_performance(perf_cache)

    # Render gorgeous rich backtest dashboard
    _render_results(target_asset, days, signals, bull_stats, bear_stats, bh_return, sharpe, alpha, best_sig, worst_sig)

    return perf_cache[target_asset]


def _render_results(ticker, days, signals, bull_stats, bear_stats, bh_return, sharpe, alpha, best, worst):
    """Render terminal report."""
    title = f"BACKTEST RESULTS — {ticker} — Last {days} Days"
    
    # 1. Summary Grid Table
    t = Table(title="[bold gold1]Key Performance Indicators[/bold gold1]", show_header=True, header_style="bold bright_cyan", expand=True)
    t.add_column("Metric", style="dim white")
    t.add_column("Bullish Signals", justify="right")
    t.add_column("Bearish Signals", justify="right")
    t.add_column("Combined / Benchmark", justify="right")

    t.add_row(
        "Total Signals Generated",
        str(bull_stats["total_signals"]),
        str(bear_stats["total_signals"]),
        str(len(signals))
    )
    t.add_row(
        "Win Rate (7d outcome)",
        f"{bull_stats['win_rate']*100:.1f}%",
        f"{bear_stats['win_rate']*100:.1f}%",
        f"{sum(1 for s in signals if s['win_7d'])/max(1, len(signals))*100:.1f}%"
    )
    t.add_row(
        "Average Fwd Return",
        f"{bull_stats['avg_return']:+.2f}%",
        f"{bear_stats['avg_return']:+.2f}%",
        f"{np.mean([s['return_7d'] for s in signals]):+.2f}%" if signals else "N/A"
    )
    t.add_row(
        "Buy & Hold Benchmark Return",
        "—",
        "—",
        f"{bh_return:+.2f}%"
    )
    t.add_row(
        "Alpha Generated vs B&H",
        "—",
        "—",
        f"[bold green]{alpha:+.2f}%[/bold green]" if alpha >= 0 else f"[bold red]{alpha:+.2f}%[/bold red]"
    )
    t.add_row(
        "Signal Sharpe Ratio",
        "—",
        "—",
        f"{sharpe:.2f}"
    )

    console.print(Panel(t, title=title, border_style="gold1"))

    # Best & Worst Signal panels
    if best:
        best_str = f"Date: {best['date']} | Entry: ${best['entry_price']:.2f} -> 7d Price: ${best['price_7d']:.2f} | Return: {best['return_7d']:+.2f}%"
        console.print(Panel(best_str, title="[bold green]✅ BEST SIGNAL SETUP[/bold green]", border_style="green"))
    if worst:
        worst_str = f"Date: {worst['date']} | Entry: ${worst['entry_price']:.2f} -> 7d Price: ${worst['price_7d']:.2f} | Return: {worst['return_7d']:+.2f}%"
        console.print(Panel(worst_str, title="[bold red]❌ WORST SIGNAL SETUP[/bold red]", border_style="red"))

    # Alert if suppressed
    if bull_stats["total_signals"] >= 3 and bull_stats["win_rate"] < 0.45:
        console.print("[bold red]⚠️  WARNING: Bullish signals for this asset will be AUTO-SUPPRESSED (win rate < 45%).[/bold red]")
    if bear_stats["total_signals"] >= 3 and bear_stats["win_rate"] < 0.45:
        console.print("[bold red]⚠️  WARNING: Bearish signals for this asset will be AUTO-SUPPRESSED (win rate < 45%).[/bold red]")


# ---------------------------------------------------------------------------
# Bulk / Watchlist Backtesting Engine
# ---------------------------------------------------------------------------

def run_backtest_all(days: int = 90) -> None:
    """Run historical backtesting on all stock/crypto assets in the watchlist."""
    console.print(f"\n[bold magenta]=== RUNNING BULK WATCHLIST BACKTEST ({days} days) ===[/bold magenta]")
    
    # Load watchlist.json
    try:
        with open("watchlist.json", "r", encoding="utf-8") as f:
            wl = json.load(f)
    except Exception as exc:
        console.print(f"[bold red]Failed to load watchlist: {exc}[/bold red]")
        return

    # Extract all stocks and crypto symbols
    symbols = []
    for cat in ["us_stocks", "indian_stocks"]:
        symbols.extend(wl.get(cat, []))
        
    for crypto in wl.get("crypto", []):
        if isinstance(crypto, dict):
            symbols.append(crypto.get("symbol", "").upper() + "-USD")
        elif isinstance(crypto, str):
            symbols.append(crypto.upper() + "-USD")

    symbols = list(dict.fromkeys([s for s in symbols if s]))
    console.print(f"Discovered {len(symbols)} tickers for backtesting.")

    summary_table = Table(title="[bold magenta]BULK BACKTEST RESULTS SUMMARY[/bold magenta]", show_header=True, header_style="bold bright_magenta")
    summary_table.add_column("Ticker", style="bold white")
    summary_table.add_column("Total Signals", justify="right")
    summary_table.add_column("Win Rate (7d)", justify="right")
    summary_table.add_column("Avg Return", justify="right")
    summary_table.add_column("B&H Return", justify="right")
    summary_table.add_column("Alpha vs B&H", justify="right")
    summary_table.add_column("Sharpe Ratio", justify="right")

    for ticker in symbols:
        try:
            res = run_backtest(ticker, days=days)
            if res:
                total_sigs = res["total_signals"]
                combined_win_rate = (res["bullish"]["win_rate"] * res["bullish"]["total_signals"] + res["bearish"]["win_rate"] * res["bearish"]["total_signals"]) / max(1, total_sigs)
                combined_avg_ret = (res["bullish"]["avg_return"] * res["bullish"]["total_signals"] + res["bearish"]["avg_return"] * res["bearish"]["total_signals"]) / max(1, total_sigs)
                
                alpha_str = f"[bold green]{res['alpha']:+.2f}%[/bold green]" if res["alpha"] >= 0 else f"[bold red]{res['alpha']:+.2f}%[/bold red]"
                
                summary_table.add_row(
                    ticker,
                    str(total_sigs),
                    f"{combined_win_rate*100:.1f}%",
                    f"{combined_avg_ret:+.2f}%",
                    f"{res['buy_and_hold_return']:+.2f}%",
                    alpha_str,
                    f"{res['sharpe_ratio']:.2f}"
                )
        except Exception as e:
            logger.error("Error backtesting %s: %s", ticker, e)

    console.print("\n")
    console.print(summary_table)
