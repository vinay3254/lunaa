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
