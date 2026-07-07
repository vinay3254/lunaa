# Paper-Trading Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `portfolio.json` into a self-driving simulated (paper) brokerage account that automatically opens and closes long positions based on LUNA's existing ML scoring engine, so strategy profitability can be measured before any real money is involved.

**Architecture:** One new module, `paper_trader.py`, with two functions (`open_new_positions`, `check_exits`) called from two existing hook points in `luna.py` (`run_full_cycle` for daily entries, `run_alert_check` for hourly exit checks). A small additive change to `scanner.py::generate_tactical_card()` exposes numeric fields (`stop_loss`, `confidence_tier`) that today only exist as display strings. No new services, no new scheduling, no broker integration of any kind.

**Tech Stack:** Python 3.10, pytest (already installed, not yet in `requirements.txt`), existing `portfolio.py` / `scanner.py` / `luna.py` modules.

## Global Constraints

- Starting virtual capital: **$100,000** (one-time bootstrap of `portfolio.json`, not a reset).
- Risk per trade: **2%** of total portfolio value (`cash + open_positions_value`).
- Portfolio heat cap: **20%** — matches `portfolio.py`'s existing "HIGH PORTFOLIO HEAT" threshold (portfolio.py:448).
- Entry filter: `tactical_card["confidence_tier"] == "HIGH"` only.
- Max hold period: **30 days**, matching `portfolio.py`'s existing stale-trade threshold (portfolio.py:379).
- **Long-only.** No short positions — out of scope per the approved spec.
- Position quantity is **fractional** (`round(x, 6)`), not whole-share — this is virtual money, and whole-share flooring would zero out nearly all crypto candidates.
- `portfolio.json` position fields MUST be `quantity` (not `qty`) and `entry_date`/`exit_date` as plain `YYYY-MM-DD` strings — verified directly against `portfolio.py:298-311`; any deviation silently zeroes P&L math with no error.
- No code path may place a real brokerage order. `PAPER_TRADING = True` is a module-level constant checked at the top of both public functions, purely as a guard-rail/doc marker.

Full rationale for every rule above: `docs/superpowers/specs/2026-07-07-paper-trading-engine-design.md`.

---

### Task 1: Expose numeric `stop_loss` and `confidence_tier` from `generate_tactical_card`

**Files:**
- Modify: `scanner.py:587-818` (function `generate_tactical_card`)
- Test: `test_scanner_tactical_card.py` (new file, repo root — matches existing convention of root-level test files like `test_upgrade.py`)

**Interfaces:**
- Produces: `generate_tactical_card(asset: dict, score_result: dict, macro_state: dict) -> dict` now additionally includes:
  - `"stop_loss": float | None` — the numeric stop value already computed internally, `None` when direction is `"neutral"` or `price <= 0`.
  - `"confidence_tier": "HIGH" | "MEDIUM" | "LOW"` — derived from the existing `confidence` float using the same breakpoints as the existing `stars` string (scanner.py:604-613).
- All existing keys (`confidence_stars`, `stop_invalidation`, etc.) are unchanged — this is purely additive.

- [ ] **Step 1: Write the failing tests**

Create `test_scanner_tactical_card.py`:

```python
"""
test_scanner_tactical_card.py
==============================
Verifies generate_tactical_card() exposes numeric stop_loss and a
confidence_tier alongside its existing display-string fields. These fields
are what paper_trader.py relies on for entry/exit decisions — the display
strings (stop_invalidation, confidence_stars) can't be parsed reliably.
"""

from scanner import generate_tactical_card


def _macro_state():
    return {"regime": "RISK-ON", "vix": 15.0}


def _base_asset(**overrides):
    asset = {
        "ticker": "NVDA",
        "price": 120.0,
        "rsi": 45.0,
        "macd": {},
        "ema50": 110.0,
        "support": [115.0, 108.0],
        "resistance": [130.0],
    }
    asset.update(overrides)
    return asset


def _score_result(direction, score, **overrides):
    result = {
        "direction": direction,
        "score": score,
        "breakdown": {},
        "ml_prediction": {"fallback": True},
    }
    result.update(overrides)
    return result


def test_bullish_stop_loss_uses_ema50_when_below_price():
    asset = _base_asset(ema50=110.0, support=[115.0, 108.0])
    card = generate_tactical_card(asset, _score_result("bullish", 8.0), _macro_state())
    assert card["stop_loss"] == 110.0


def test_bullish_stop_loss_falls_back_to_support_without_ema50():
    asset = _base_asset(ema50=None, support=[115.0, 108.0])
    card = generate_tactical_card(asset, _score_result("bullish", 8.0), _macro_state())
    # Nearest support below price=120.0 is 115.0
    assert card["stop_loss"] == 115.0


def test_bullish_stop_loss_default_5pct_without_ema50_or_support():
    asset = _base_asset(ema50=None, support=[])
    card = generate_tactical_card(asset, _score_result("bullish", 8.0), _macro_state())
    assert card["stop_loss"] == 120.0 * 0.95


def test_neutral_direction_has_no_stop_loss():
    asset = _base_asset()
    card = generate_tactical_card(asset, _score_result("neutral", 0.0), _macro_state())
    assert card["stop_loss"] is None


def test_high_confidence_tier_for_strong_fallback_score():
    asset = _base_asset()
    # fallback confidence = abs(score)/10 = 0.8 -> HIGH (matches existing >=0.7 star breakpoint)
    card = generate_tactical_card(asset, _score_result("bullish", 8.0), _macro_state())
    assert card["confidence_tier"] == "HIGH"


def test_medium_confidence_tier_for_moderate_fallback_score():
    asset = _base_asset()
    # confidence = 0.55 -> MEDIUM
    card = generate_tactical_card(asset, _score_result("bullish", 5.5), _macro_state())
    assert card["confidence_tier"] == "MEDIUM"


def test_low_confidence_tier_for_weak_fallback_score():
    asset = _base_asset()
    # confidence = 0.3 -> LOW
    card = generate_tactical_card(asset, _score_result("bullish", 3.0), _macro_state())
    assert card["confidence_tier"] == "LOW"


def test_existing_display_fields_still_present():
    asset = _base_asset()
    card = generate_tactical_card(asset, _score_result("bullish", 8.0), _macro_state())
    assert "confidence_stars" in card
    assert "stop_invalidation" in card
    assert card["stop_invalidation"] != ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_scanner_tactical_card.py -v`
Expected: FAIL — `KeyError: 'stop_loss'` and `KeyError: 'confidence_tier'` (fields don't exist yet).

- [ ] **Step 3: Implement the minimal change**

In `scanner.py`, the confidence tier is derived right after `stars` is set. Find this block (scanner.py:604-613):

```python
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
```

Immediately after it, add:

```python
    if confidence >= 0.7:
        confidence_tier = "HIGH"
    elif confidence >= 0.5:
        confidence_tier = "MEDIUM"
    else:
        confidence_tier = "LOW"
```

Next, initialize `stop_val = None` before the entry-zone block so it's always defined even when `price <= 0` or `direction == "neutral"`. Find (scanner.py:750-751):

```python
    # 5. Entry Zone & Stop Loss Invalidation Zone & Time Horizon
    if price > 0:
```

Change to:

```python
    # 5. Entry Zone & Stop Loss Invalidation Zone & Time Horizon
    stop_val = None
    if price > 0:
```

Finally, update the return statement (scanner.py:807-818):

```python
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
```

to:

```python
    return {
        "confidence_score": round(confidence, 2),
        "confidence_stars": stars,
        "confidence_tier": confidence_tier,
        "aligned_factors": f"{aligned_count}/{total_factors}",
        "evidence": evidence,
        "model_info": model_info,
        "accuracy_info": accuracy_info,
        "backtest_info": backtest_info,
        "entry_zone": entry_zone,
        "stop_loss": stop_val,
        "stop_invalidation": stop_invalidation,
        "time_horizon": "3-7 days"
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_scanner_tactical_card.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add scanner.py test_scanner_tactical_card.py
git commit -m "feat: expose numeric stop_loss and confidence_tier from tactical card"
```

---

### Task 2: `paper_trader.py` module skeleton + portfolio bootstrap

**Files:**
- Create: `paper_trader.py`
- Test: `test_paper_trader.py` (new file, repo root)

**Interfaces:**
- Consumes: nothing from other tasks yet.
- Produces:
  - Module constants: `STARTING_CAPITAL = 100_000.0`, `RISK_PER_TRADE_PCT = 0.02`, `MAX_PORTFOLIO_HEAT_PCT = 20.0`, `MAX_HOLD_DAYS = 30`, `TAKE_PROFIT_RR_FALLBACK = 2.0`, `PAPER_TRADING = True`.
  - `bootstrap_portfolio(portfolio: dict) -> dict`

- [ ] **Step 1: Write the failing test**

Create `test_paper_trader.py`:

```python
"""
test_paper_trader.py
=====================
Unit and integration tests for the paper-trading engine: automatic
entry/exit of simulated positions in portfolio.json, driven by LUNA's
existing scanner/ML scoring output. No real broker integration exists
or is exercised anywhere in this file.
"""

import paper_trader


def test_bootstrap_sets_starting_capital_on_untouched_portfolio():
    portfolio = {"positions": [], "closed_positions": [], "cash_balance": 0.0, "currency": "USD"}
    result = paper_trader.bootstrap_portfolio(portfolio)
    assert result["cash_balance"] == paper_trader.STARTING_CAPITAL


def test_bootstrap_leaves_existing_cash_balance_untouched():
    portfolio = {"positions": [], "closed_positions": [], "cash_balance": 87543.21, "currency": "USD"}
    result = paper_trader.bootstrap_portfolio(portfolio)
    assert result["cash_balance"] == 87543.21


def test_bootstrap_leaves_zero_cash_untouched_if_positions_exist():
    # Zero cash balance with open positions means fully invested, not untouched.
    portfolio = {
        "positions": [{"asset": "NVDA", "quantity": 1.0, "entry_price": 100.0,
                        "stop_loss": 90.0, "take_profit": 120.0,
                        "entry_date": "2026-07-01", "status": "open"}],
        "closed_positions": [],
        "cash_balance": 0.0,
        "currency": "USD",
    }
    result = paper_trader.bootstrap_portfolio(portfolio)
    assert result["cash_balance"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_paper_trader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paper_trader'`

- [ ] **Step 3: Write minimal implementation**

Create `paper_trader.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_paper_trader.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add paper_trader.py test_paper_trader.py
git commit -m "feat: add paper_trader module with portfolio bootstrap"
```

---

### Task 3: Entry logic — `open_new_positions`

**Files:**
- Modify: `paper_trader.py`
- Modify: `test_paper_trader.py`

**Interfaces:**
- Consumes: `bootstrap_portfolio` (Task 2). From `portfolio.py`: `calculate_portfolio_status(market_data: dict, silent: bool = False) -> dict` (existing, returns dict with `total_portfolio_value` and `portfolio_heat` keys).
- Produces: `open_new_positions(ranked: dict, portfolio: dict, market_data: dict) -> dict` — `ranked` is the dict returned by `scanner.run_full_scan()` (has a `"bullish"` key, a list of opportunity dicts each with `ticker`, `price`, `score`, `resistance` (list), `tactical_card` (dict with `confidence_tier`, `stop_loss`)). Returns the mutated `portfolio` dict; caller is responsible for `save_portfolio()`.

- [ ] **Step 1: Write the failing tests**

Append to `test_paper_trader.py`:

```python
def _opportunity(ticker, price, stop_loss, confidence_tier="HIGH", resistance=None, score=8.0):
    return {
        "ticker": ticker,
        "price": price,
        "score": score,
        "resistance": resistance if resistance is not None else [],
        "tactical_card": {"stop_loss": stop_loss, "confidence_tier": confidence_tier},
    }


def _empty_portfolio(cash=100_000.0):
    return {"positions": [], "closed_positions": [], "cash_balance": cash, "currency": "USD"}


def test_open_new_positions_sizes_by_2pct_risk(monkeypatch):
    import portfolio as portfolio_module

    def fake_status(market_data, silent=False):
        return {"total_portfolio_value": 100_000.0, "portfolio_heat": 0.0}

    monkeypatch.setattr(portfolio_module, "calculate_portfolio_status", fake_status)

    ranked = {"bullish": [_opportunity("NVDA", price=120.0, stop_loss=110.0, resistance=[140.0])]}
    portfolio = _empty_portfolio()

    result = paper_trader.open_new_positions(ranked, portfolio, market_data={})

    assert len(result["positions"]) == 1
    pos = result["positions"][0]
    # dollar_risk = 100_000 * 0.02 = 2000; risk_per_share = 120-110 = 10; qty = 200
    assert pos["quantity"] == 200.0
    assert pos["asset"] == "NVDA"
    assert pos["entry_price"] == 120.0
    assert pos["stop_loss"] == 110.0
    assert pos["take_profit"] == 140.0  # nearest resistance above price
    assert pos["status"] == "open"
    assert result["cash_balance"] == 100_000.0 - (200.0 * 120.0)


def test_open_new_positions_falls_back_to_2to1_rr_without_resistance(monkeypatch):
    import portfolio as portfolio_module
    monkeypatch.setattr(portfolio_module, "calculate_portfolio_status",
                         lambda market_data, silent=False: {"total_portfolio_value": 100_000.0, "portfolio_heat": 0.0})

    ranked = {"bullish": [_opportunity("NVDA", price=120.0, stop_loss=110.0, resistance=[])]}
    result = paper_trader.open_new_positions(ranked, _empty_portfolio(), market_data={})

    pos = result["positions"][0]
    # entry + 2*(entry-stop) = 120 + 2*10 = 140
    assert pos["take_profit"] == 140.0


def test_open_new_positions_skips_medium_and_low_confidence(monkeypatch):
    import portfolio as portfolio_module
    monkeypatch.setattr(portfolio_module, "calculate_portfolio_status",
                         lambda market_data, silent=False: {"total_portfolio_value": 100_000.0, "portfolio_heat": 0.0})

    ranked = {"bullish": [
        _opportunity("AAA", price=50.0, stop_loss=45.0, confidence_tier="MEDIUM"),
        _opportunity("BBB", price=50.0, stop_loss=45.0, confidence_tier="LOW"),
    ]}
    result = paper_trader.open_new_positions(ranked, _empty_portfolio(), market_data={})
    assert result["positions"] == []


def test_open_new_positions_skips_ticker_already_open(monkeypatch):
    import portfolio as portfolio_module
    monkeypatch.setattr(portfolio_module, "calculate_portfolio_status",
                         lambda market_data, silent=False: {"total_portfolio_value": 100_000.0, "portfolio_heat": 0.0})

    portfolio = _empty_portfolio()
    portfolio["positions"].append({
        "asset": "NVDA", "quantity": 10.0, "entry_price": 100.0, "stop_loss": 90.0,
        "take_profit": 120.0, "entry_date": "2026-07-01", "status": "open",
    })
    ranked = {"bullish": [_opportunity("NVDA", price=120.0, stop_loss=110.0)]}
    result = paper_trader.open_new_positions(ranked, portfolio, market_data={})
    assert len(result["positions"]) == 1  # unchanged, no duplicate


def test_open_new_positions_skips_when_stop_loss_missing_or_invalid(monkeypatch):
    import portfolio as portfolio_module
    monkeypatch.setattr(portfolio_module, "calculate_portfolio_status",
                         lambda market_data, silent=False: {"total_portfolio_value": 100_000.0, "portfolio_heat": 0.0})

    ranked = {"bullish": [
        _opportunity("NOSTOP", price=100.0, stop_loss=None),
        _opportunity("BADSTOP", price=100.0, stop_loss=105.0),  # stop above price
    ]}
    result = paper_trader.open_new_positions(ranked, _empty_portfolio(), market_data={})
    assert result["positions"] == []


def test_open_new_positions_respects_heat_cap(monkeypatch):
    import portfolio as portfolio_module
    # Portfolio already at 19% heat on a $100k account -> only ~1% of room left.
    monkeypatch.setattr(portfolio_module, "calculate_portfolio_status",
                         lambda market_data, silent=False: {"total_portfolio_value": 100_000.0, "portfolio_heat": 19.0})

    # risk_per_share=10, dollar_risk=2000 -> qty=200 -> candidate_risk=2000 (2% of 100k)
    # projected heat = (19000 + 2000) / 100000 * 100 = 21% > 20% cap -> must skip
    ranked = {"bullish": [_opportunity("NVDA", price=120.0, stop_loss=110.0, resistance=[140.0])]}
    result = paper_trader.open_new_positions(ranked, _empty_portfolio(), market_data={})
    assert result["positions"] == []


def test_open_new_positions_skips_when_cash_insufficient(monkeypatch):
    import portfolio as portfolio_module
    monkeypatch.setattr(portfolio_module, "calculate_portfolio_status",
                         lambda market_data, silent=False: {"total_portfolio_value": 100_000.0, "portfolio_heat": 0.0})

    # qty=200 @ $120 = $24,000 cost, but only $10 cash available.
    portfolio = _empty_portfolio(cash=10.0)
    ranked = {"bullish": [_opportunity("NVDA", price=120.0, stop_loss=110.0, resistance=[140.0])]}
    result = paper_trader.open_new_positions(ranked, portfolio, market_data={})
    assert result["positions"] == []
    assert result["cash_balance"] == 10.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_paper_trader.py -v -k open_new_positions`
Expected: FAIL with `AttributeError: module 'paper_trader' has no attribute 'open_new_positions'`

- [ ] **Step 3: Write minimal implementation**

Append to `paper_trader.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_paper_trader.py -v -k open_new_positions`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add paper_trader.py test_paper_trader.py
git commit -m "feat: implement paper-trading entry logic with risk-based sizing and heat cap"
```

---

### Task 4: Exit logic — `check_exits`

**Files:**
- Modify: `paper_trader.py`
- Modify: `test_paper_trader.py`

**Interfaces:**
- Consumes: from `portfolio.py`: `get_current_price(asset: str, market_data: dict) -> float` (existing, returns `float('nan')` on failure).
- Produces: `check_exits(portfolio: dict, market_data: dict) -> dict` — `market_data` is the raw dict returned by `market_data.fetch_all_market_data()` (has top-level `traditional`/`crypto`/`global_snapshot` keys, which `get_current_price` reads directly). Returns the mutated `portfolio` dict; caller must call `save_portfolio()`.

- [ ] **Step 1: Write the failing tests**

Append to `test_paper_trader.py`:

```python
def _open_position(asset="NVDA", qty=10.0, entry_price=100.0, stop_loss=90.0,
                    take_profit=120.0, entry_date="2026-07-01"):
    return {
        "asset": asset, "quantity": qty, "entry_price": entry_price,
        "stop_loss": stop_loss, "take_profit": take_profit,
        "entry_date": entry_date, "status": "open", "source": "paper_trader",
        "signal_score": 8.0, "uuid": "test-uuid",
    }


def _market_data_with_price(ticker, price):
    return {"traditional": {ticker: {"price": price}}, "crypto": {}, "global_snapshot": []}


def test_check_exits_closes_on_stop_loss_hit():
    portfolio = {"positions": [_open_position(stop_loss=90.0)], "closed_positions": [], "cash_balance": 0.0}
    result = paper_trader.check_exits(portfolio, _market_data_with_price("NVDA", 88.0))

    assert result["positions"] == []
    assert len(result["closed_positions"]) == 1
    closed = result["closed_positions"][0]
    assert closed["exit_reason"] == "stop_loss"
    assert closed["exit_price"] == 90.0
    # proceeds = 10 * 90 = 900
    assert result["cash_balance"] == 900.0


def test_check_exits_closes_on_take_profit_hit():
    portfolio = {"positions": [_open_position(take_profit=120.0)], "closed_positions": [], "cash_balance": 0.0}
    result = paper_trader.check_exits(portfolio, _market_data_with_price("NVDA", 125.0))

    closed = result["closed_positions"][0]
    assert closed["exit_reason"] == "take_profit"
    assert closed["exit_price"] == 120.0
    assert result["cash_balance"] == 1200.0


def test_check_exits_leaves_position_open_between_sl_and_tp():
    portfolio = {"positions": [_open_position(stop_loss=90.0, take_profit=120.0)],
                 "closed_positions": [], "cash_balance": 0.0}
    result = paper_trader.check_exits(portfolio, _market_data_with_price("NVDA", 105.0))

    assert len(result["positions"]) == 1
    assert result["closed_positions"] == []


def test_check_exits_closes_on_max_hold_days():
    from datetime import datetime, timezone, timedelta

    # Use a date relative to "now" so the test doesn't depend on mocking
    # datetime.now() — 45 days ago always exceeds MAX_HOLD_DAYS=30 regardless
    # of when the test suite runs.
    stale_entry_date = (datetime.now(timezone.utc) - timedelta(days=45)).strftime("%Y-%m-%d")
    portfolio = {
        "positions": [_open_position(entry_date=stale_entry_date, stop_loss=50.0, take_profit=200.0)],
        "closed_positions": [], "cash_balance": 0.0,
    }
    result = paper_trader.check_exits(portfolio, _market_data_with_price("NVDA", 105.0))

    closed = result["closed_positions"][0]
    assert closed["exit_reason"] == "max_hold"
    assert closed["exit_price"] == 105.0


def test_check_exits_skips_position_when_price_unavailable():
    portfolio = {"positions": [_open_position()], "closed_positions": [], "cash_balance": 0.0}
    # No matching ticker anywhere in market_data -> get_current_price returns NaN
    result = paper_trader.check_exits(portfolio, {"traditional": {}, "crypto": {}, "global_snapshot": []})

    assert len(result["positions"]) == 1
    assert result["closed_positions"] == []


def test_check_exits_ignores_already_closed_positions():
    closed_already = _open_position()
    closed_already["status"] = "closed"
    portfolio = {"positions": [closed_already], "closed_positions": [], "cash_balance": 500.0}
    result = paper_trader.check_exits(portfolio, _market_data_with_price("NVDA", 50.0))

    # Untouched: still in positions list, cash unchanged, nothing newly closed
    assert len(result["positions"]) == 1
    assert result["closed_positions"] == []
    assert result["cash_balance"] == 500.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_paper_trader.py -v -k check_exits`
Expected: FAIL with `AttributeError: module 'paper_trader' has no attribute 'check_exits'`

- [ ] **Step 3: Write minimal implementation**

Append to `paper_trader.py`:

```python
def check_exits(portfolio: dict, market_data: dict) -> dict:
    """
    Close simulated open positions whose live price has crossed stop-loss
    or take-profit, or that have been held past MAX_HOLD_DAYS. Mutates and
    returns `portfolio`; caller must call portfolio.save_portfolio().
    """
    if not PAPER_TRADING:
        return portfolio

    from portfolio import get_current_price

    today = datetime.now(timezone.utc).date()
    still_open = []
    closed_positions = portfolio.setdefault("closed_positions", [])

    for pos in portfolio.get("positions", []):
        if pos.get("status", "open").lower() != "open":
            still_open.append(pos)
            continue

        ticker = pos.get("asset")
        current_price = get_current_price(ticker, market_data)
        if current_price is None or np.isnan(current_price):
            still_open.append(pos)
            continue

        stop_loss = pos.get("stop_loss")
        take_profit = pos.get("take_profit")

        try:
            entry_date = datetime.strptime(pos.get("entry_date", ""), "%Y-%m-%d").date()
            days_held = (today - entry_date).days
        except (TypeError, ValueError):
            days_held = 0

        exit_reason = None
        exit_price = None
        if stop_loss is not None and current_price <= stop_loss:
            exit_reason, exit_price = "stop_loss", stop_loss
        elif take_profit is not None and current_price >= take_profit:
            exit_reason, exit_price = "take_profit", take_profit
        elif days_held > MAX_HOLD_DAYS:
            exit_reason, exit_price = "max_hold", current_price

        if exit_reason is None:
            still_open.append(pos)
            continue

        qty = pos.get("quantity", 0.0)
        proceeds = qty * exit_price
        pnl = proceeds - (qty * pos.get("entry_price", 0.0))

        closed_pos = dict(pos)
        closed_pos.update({
            "status": "closed",
            "exit_price": exit_price,
            "exit_date": today.strftime("%Y-%m-%d"),
            "exit_reason": exit_reason,
        })
        closed_positions.append(closed_pos)
        portfolio["cash_balance"] = portfolio.get("cash_balance", 0.0) + proceeds
        logger.info(
            "Paper SELL: %.6f %s @ $%.2f (%s) P&L=$%.2f",
            qty, ticker, exit_price, exit_reason, pnl,
        )

    portfolio["positions"] = still_open
    return portfolio
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_paper_trader.py -v -k check_exits`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add paper_trader.py test_paper_trader.py
git commit -m "feat: implement paper-trading exit logic (stop-loss/take-profit/max-hold)"
```

---

### Task 5: Wire into `luna.py`

**Files:**
- Modify: `luna.py:1038-1090` (inside `run_full_cycle`)
- Modify: `luna.py:1240-1289` (inside `run_alert_check`)

**Interfaces:**
- Consumes: `paper_trader.open_new_positions` and `paper_trader.check_exits` (Tasks 3 and 4); `portfolio.load_portfolio` / `portfolio.save_portfolio` (existing).
- Produces: nothing new consumed by later tasks — this is the final wiring task.

This task has no isolated unit test of its own (`run_full_cycle`/`run_alert_check` are large, already-untested integration functions with live network calls — consistent with the rest of the codebase, which has no test coverage on these either). Verification is a manual dry run in Step 3.

- [ ] **Step 1: Add the entry hook to `run_full_cycle`**

Find this block in `luna.py` (lines 1073-1091):

```python
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
```

Insert a new block immediately after `_log_done("Generate all reports", t0)` (and before the existing `# Precalculate portfolio status before notification` comment):

```python
        _log_done("Generate all reports", t0)

        # Paper-trading engine: open new simulated positions from bullish,
        # HIGH-confidence opportunities. Never touches real money or any
        # broker — see paper_trader.py module docstring.
        try:
            import paper_trader
            from portfolio import load_portfolio, save_portfolio
            paper_portfolio = load_portfolio()
            paper_portfolio = paper_trader.open_new_positions(ranked, paper_portfolio, report_market_data)
            save_portfolio(paper_portfolio)
        except Exception as exc:
            logger.error("Paper trader open_new_positions failed: %s", exc)
```

- [ ] **Step 2: Add the exit hook to `run_alert_check`**

Find this block in `luna.py` (lines 1248-1258):

```python
    try:
        watchlist  = load_watchlist()
        last_state = load_state()

        market_data = fetch_all_market_data(watchlist)
        if not market_data:
            logger.warning("No market data for alert check — skipping.")
            return

        flat_data     = _build_enriched_flat(market_data)
        enriched_data = enrich_all_assets(flat_data)
```

Insert a new block immediately after the `if not market_data:` guard and before `flat_data = _build_enriched_flat(market_data)`:

```python
        market_data = fetch_all_market_data(watchlist)
        if not market_data:
            logger.warning("No market data for alert check — skipping.")
            return

        # Paper-trading engine: check open simulated positions against live
        # prices for stop-loss/take-profit/max-hold exits.
        try:
            import paper_trader
            from portfolio import load_portfolio, save_portfolio, calculate_portfolio_status
            paper_portfolio = load_portfolio()
            paper_portfolio = paper_trader.check_exits(paper_portfolio, market_data)
            save_portfolio(paper_portfolio)
            calculate_portfolio_status(market_data=market_data, silent=True)  # refresh portfolio-status.md
        except Exception as exc:
            logger.error("Paper trader check_exits failed: %s", exc)

        flat_data     = _build_enriched_flat(market_data)
        enriched_data = enrich_all_assets(flat_data)
```

- [ ] **Step 3: Manual dry-run verification**

Run: `python luna.py --alert-check`

Expected in output: either `Paper SELL: ...` log lines (if any seeded position's stop/take-profit was crossed) or no paper-trader errors and normal alert-check completion. Then run:

Run: `python luna.py --run`

Expected: log line `Paper BUY: ...` for any bullish HIGH-confidence opportunity found (or none, if the day's scan has no HIGH-confidence bullish signals — not a failure), and no `Paper trader open_new_positions failed` error line. Confirm `portfolio.json` now has a non-zero `cash_balance` and, if any trade fired, a new entry under `positions`.

- [ ] **Step 4: Run the full test suite to confirm no regressions**

Run: `pytest test_paper_trader.py test_scanner_tactical_card.py -v`
Expected: all tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add luna.py
git commit -m "feat: wire paper-trading engine into daily scan and hourly alert-check"
```

---

### Task 6: End-to-end multi-day integration test

**Files:**
- Modify: `test_paper_trader.py`

**Interfaces:**
- Consumes: `bootstrap_portfolio`, `open_new_positions`, `check_exits` (Tasks 2-4). No new interfaces produced — this is a pure test task validating the whole module works together across a simulated multi-day sequence, per design spec §8.

- [ ] **Step 1: Write the integration test**

Append to `test_paper_trader.py`:

```python
def test_end_to_end_multi_day_simulation(monkeypatch):
    """
    Day 1: scan finds one HIGH-confidence bullish setup -> paper trader buys it.
    Day 2: price still between SL/TP -> position stays open, no change.
    Day 3: price drops through stop-loss -> position closes, cash realizes the loss.
    Asserts final cash_balance and closed_positions match hand-calculated values.
    """
    import portfolio as portfolio_module
    monkeypatch.setattr(portfolio_module, "calculate_portfolio_status",
                         lambda market_data, silent=False: {"total_portfolio_value": 100_000.0, "portfolio_heat": 0.0})

    portfolio = {"positions": [], "closed_positions": [], "cash_balance": 0.0, "currency": "USD"}

    # --- Day 1: entry ---
    ranked = {"bullish": [_opportunity("NVDA", price=120.0, stop_loss=110.0, resistance=[140.0])]}
    portfolio = paper_trader.open_new_positions(ranked, portfolio, market_data={})

    assert len(portfolio["positions"]) == 1
    # dollar_risk = 100_000 * 0.02 = 2000; risk_per_share = 10 -> qty = 200
    assert portfolio["positions"][0]["quantity"] == 200.0
    cash_after_entry = 100_000.0 - (200.0 * 120.0)  # = 76,000.0
    assert portfolio["cash_balance"] == cash_after_entry

    # --- Day 2: no exit trigger ---
    portfolio = paper_trader.check_exits(portfolio, _market_data_with_price("NVDA", 118.0))
    assert len(portfolio["positions"]) == 1
    assert portfolio["closed_positions"] == []
    assert portfolio["cash_balance"] == cash_after_entry  # unchanged

    # --- Day 3: stop-loss hit ---
    portfolio = paper_trader.check_exits(portfolio, _market_data_with_price("NVDA", 108.0))

    assert portfolio["positions"] == []
    assert len(portfolio["closed_positions"]) == 1
    closed = portfolio["closed_positions"][0]
    assert closed["exit_reason"] == "stop_loss"
    assert closed["exit_price"] == 110.0

    proceeds = 200.0 * 110.0  # = 22,000.0
    expected_final_cash = cash_after_entry + proceeds  # = 98,000.0
    assert portfolio["cash_balance"] == expected_final_cash

    # Net result: entered at 120, stopped out at 110 on 200 units -> -2000 realized loss
    realized_loss = (200.0 * 110.0) - (200.0 * 120.0)
    assert realized_loss == -2000.0
    assert expected_final_cash == 100_000.0 + realized_loss
```

- [ ] **Step 2: Run test to verify it fails first (sanity check on the test itself)**

Run: `pytest test_paper_trader.py -v -k end_to_end`
Expected: at this point in the plan the implementation already exists (Tasks 2-4 done), so this should already PASS. If it fails, the failure output tells you which assertion doesn't match — do not adjust the implementation to fit the test without first re-checking the hand-calculated numbers above are correct.

- [ ] **Step 3: Run the complete test suite**

Run: `pytest test_paper_trader.py test_scanner_tactical_card.py -v`
Expected: all tests PASS (final count: 8 scanner tests + 17 paper_trader tests).

- [ ] **Step 4: Commit**

```bash
git add test_paper_trader.py
git commit -m "test: add end-to-end multi-day paper-trading simulation"
```

---

## Post-Implementation Checklist

- [ ] `pytest test_paper_trader.py test_scanner_tactical_card.py -v` — all green.
- [ ] `python luna.py --run` produces a `Paper BUY` log line or a clean no-signal run — no `Paper trader ... failed` errors.
- [ ] `python luna.py --alert-check` runs clean.
- [ ] `reports/portfolio-status.md` reflects the current simulated positions after a real run.
- [ ] `git log --oneline` shows one commit per task above — six feature/test commits total.
