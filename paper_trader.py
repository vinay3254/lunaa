"""
paper_trader.py
================
Autonomous Trading Research Agent — Paper-Trading Engine

Turns portfolio.json into a self-driving SIMULATED brokerage account. This
module never places a real order and never talks to any brokerage or
exchange API — there is no broker-integration code anywhere in this
codebase. Positions are opened/closed purely as JSON bookkeeping, sized
using the same risk/heat math portfolio.py already computes for manual
positions.

Entry decisions come from scanner.run_full_scan()'s bullish opportunities
(see design spec: docs/superpowers/specs/2026-07-07-paper-trading-engine-design.md).
Exit decisions are stop-loss / take-profit / max-hold-days checks against
live prices already being fetched by luna.py's existing hourly alert-check.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger("paper_trader")

# ---------------------------------------------------------------------------
# Configuration — see design spec §9 for rationale on every value below.
# ---------------------------------------------------------------------------
PAPER_TRADING = True  # No real broker integration exists in this codebase.
STARTING_CAPITAL = 100_000.0
RISK_PER_TRADE_PCT = 0.02
MAX_PORTFOLIO_HEAT_PCT = 20.0
MAX_HOLD_DAYS = 30
TAKE_PROFIT_RR_FALLBACK = 2.0
ENTRY_CONFIDENCE_TIER = "HIGH"


def bootstrap_portfolio(portfolio: dict) -> dict:
    """
    One-time initialization of virtual capital. Only touches cash_balance
    when the ledger is genuinely untouched (no positions ever opened AND
    zero cash) — never resets an account that's already in use.
    """
    has_positions = bool(portfolio.get("positions")) or bool(portfolio.get("closed_positions"))
    if not has_positions and float(portfolio.get("cash_balance", 0.0)) == 0.0:
        portfolio["cash_balance"] = STARTING_CAPITAL
        logger.info("Paper trader: bootstrapped portfolio with $%.2f starting virtual capital.", STARTING_CAPITAL)
    return portfolio


def open_new_positions(ranked: dict, portfolio: dict, market_data: dict) -> dict:
    """
    Open new simulated long positions from LUNA's bullish, HIGH-confidence
    scan opportunities. Sizes each position to risk exactly
    RISK_PER_TRADE_PCT of total portfolio value, and refuses any entry that
    would push total portfolio heat above MAX_PORTFOLIO_HEAT_PCT. Mutates
    and returns `portfolio`; caller must call portfolio.save_portfolio().
    """
    if not PAPER_TRADING:
        return portfolio

    from portfolio import calculate_portfolio_status

    portfolio = bootstrap_portfolio(portfolio)
    open_tickers = {
        p.get("asset") for p in portfolio.get("positions", [])
        if p.get("status", "open").lower() == "open"
    }

    status = calculate_portfolio_status(market_data=market_data, silent=True)
    total_portfolio_value = status.get("total_portfolio_value") or portfolio.get("cash_balance", 0.0)
    capital_at_risk = (status.get("portfolio_heat", 0.0) / 100.0) * total_portfolio_value

    candidates = [
        opp for opp in ranked.get("bullish", [])
        if opp.get("tactical_card", {}).get("confidence_tier") == ENTRY_CONFIDENCE_TIER
        and opp.get("ticker") not in open_tickers
    ]

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for opp in candidates:
        ticker = opp.get("ticker")
        price = opp.get("price")
        stop_loss = opp.get("tactical_card", {}).get("stop_loss")

        if price is None or price <= 0 or stop_loss is None or stop_loss >= price:
            logger.info("Paper trader: skipping %s — missing/invalid price or stop_loss.", ticker)
            continue

        risk_per_share = price - stop_loss
        resistance = opp.get("resistance") or []
        res_list = resistance if isinstance(resistance, list) else [resistance]
        valid_res = [r for r in res_list if r and r > price]
        take_profit = min(valid_res) if valid_res else price + TAKE_PROFIT_RR_FALLBACK * risk_per_share

        dollar_risk = total_portfolio_value * RISK_PER_TRADE_PCT
        qty = round(dollar_risk / risk_per_share, 6)
        if qty <= 0:
            continue

        candidate_risk = risk_per_share * qty
        projected_heat = (
            (capital_at_risk + candidate_risk) / total_portfolio_value * 100.0
            if total_portfolio_value > 0 else 0.0
        )
        if projected_heat > MAX_PORTFOLIO_HEAT_PCT:
            logger.info(
                "Paper trader: skipping %s — would push portfolio heat to %.1f%% (cap %.1f%%).",
                ticker, projected_heat, MAX_PORTFOLIO_HEAT_PCT,
            )
            continue

        cost = qty * price
        if cost > portfolio.get("cash_balance", 0.0):
            logger.info(
                "Paper trader: skipping %s — insufficient cash ($%.2f needed, $%.2f available).",
                ticker, cost, portfolio.get("cash_balance", 0.0),
            )
            continue

        position = {
            "asset": ticker,
            "quantity": qty,
            "entry_price": price,
            "stop_loss": stop_loss,
            "take_profit": round(take_profit, 6),
            "entry_date": today_str,
            "status": "open",
            "source": "paper_trader",
            "signal_score": opp.get("score"),
            "uuid": str(uuid.uuid4()),
        }
        portfolio["cash_balance"] = portfolio.get("cash_balance", 0.0) - cost
        portfolio.setdefault("positions", []).append(position)
        capital_at_risk += candidate_risk
        open_tickers.add(ticker)
        logger.info(
            "Paper BUY: %.6f %s @ $%.2f (SL=$%.2f TP=$%.2f)",
            qty, ticker, price, stop_loss, take_profit,
        )

    return portfolio
