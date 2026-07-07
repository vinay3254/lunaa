# LUNA Paper-Trading Engine — Design Spec

**Date:** 2026-07-07
**Status:** Approved for implementation

## 1. Purpose

LUNA today is a research/alerting agent: it scores opportunities and tells you about
them, but a human has to manually track and manage any resulting trade in
`portfolio.json`. This project turns `portfolio.json` into a **live, self-driving
paper (simulated-money) brokerage account**: the agent opens and closes its own
positions based on its existing scoring engine, sized with real risk-management
rules, so that whether the strategy is actually profitable can be measured over
time — before any real money is ever involved.

**Non-goals (explicitly out of scope for this spec):**
- No connection to any real brokerage/exchange. No order-placement code of any kind.
- No short positions. Only long trades on bullish signals (see §4, rationale below).
- No Telegram/Reddit/FRED credential setup — that thread is dropped per the user's
  last message; this spec does not depend on any of those integrations.
- No new market-data infrastructure — reuses the existing yfinance/CoinGecko polling
  already running on the GitHub Actions cron fixed in the prior session.

## 2. Architecture Overview

One new module, `paper_trader.py`, with two public entry points wired into the
existing `luna.py` run modes. No new scheduling, no new services — this rides
entirely on the GitHub Actions cron already in `.github/workflows/scheduled-run.yml`.

```
run_full_cycle()          [luna.py --run, daily cron]
  └─ ranked = run_full_scan(...)          (scanner.py, existing)
  └─ paper_trader.open_new_positions(ranked, portfolio, enriched_data)   [NEW HOOK]

run_alert_check()          [luna.py --alert-check, hourly cron]
  └─ market_data = fetch_all_market_data(...)   (existing)
  └─ paper_trader.check_exits(portfolio, market_data)                   [NEW HOOK]
```

Both hooks load/save `portfolio.json` via the existing `portfolio.py` functions
(`load_portfolio()` / `save_portfolio()`), and both end by calling
`portfolio.calculate_portfolio_status()` so `reports/portfolio-status.md` — which
already renders P&L, win rate, and portfolio heat — is regenerated with no changes
needed to `portfolio.py`'s reporting code.

## 3. Data Model Changes

`portfolio.json` currently:
```json
{ "positions": [], "closed_positions": [], "cash_balance": 0.00, "currency": "USD" }
```

Change: on first run, if `cash_balance == 0.00` and `positions` is empty (i.e. an
untouched ledger), `paper_trader.py` initializes it to the starting virtual capital:

```json
{ "positions": [], "closed_positions": [], "cash_balance": 100000.00, "currency": "USD" }
```

This is a one-time bootstrap, not a reset-on-every-run — once `cash_balance` is
non-zero or positions exist, it's left alone.

Each position entry (matches the schema `portfolio.py` already reads):
```json
{
  "asset": "NVDA",
  "qty": 42,
  "entry_price": 118.30,
  "stop_loss": 112.00,
  "take_profit": 131.00,
  "entry_date": "2026-07-08T01:15:00+00:00",
  "status": "open",
  "source": "paper_trader",
  "signal_score": 8.4,
  "signal_confidence": "HIGH"
}
```

Closed positions append `exit_price`, `exit_date`, `exit_reason`
(`"stop_loss" | "take_profit" | "max_hold"`), matching what `portfolio.py`'s
`calculate_portfolio_status()` already expects for the closed-positions table.

## 3.1 Correction from initial spec draft — actual data shape

Tracing the real call path (`luna.py::run_full_cycle` → `scanner.run_full_scan`,
which is an alias for `run_full_scan_with_ml` → `rank_opportunities`) showed the
opportunities actually consumed by `luna.py` (`ranked["bullish"]` /
`ranked["bearish"]`) do **not** carry a top-level `model_confidence` or `stop_loss`
field — those only exist as human-readable strings buried inside a nested
`tactical_card` dict (e.g. `stop_invalidation: "Below $112.00 (closes below
support $112.00)"`, `confidence_stars: "★★★★☆ HIGH CONFIDENCE"`). Parsing display
strings for numeric decisions would be fragile.

Instead, Task 1 of the implementation plan makes a small, additive change to
`scanner.py::generate_tactical_card()` (scanner.py:807-818) to also return the raw
values it already computes internally but currently discards:
- `stop_loss` (float | None) — the exact `stop_val` already computed for
  bullish/bearish directions (support/EMA50-based logic, scanner.py:755-798).
- `confidence_tier` (`"HIGH" | "MEDIUM" | "LOW"`) — derived from the same
  `confidence` float already computed (scanner.py:604-613), using the existing
  breakpoints (`>=0.7` → HIGH, `>=0.5` → MEDIUM, else LOW).

This does not remove or rename `stop_invalidation` / `confidence_stars` — existing
consumers (report rendering) are unaffected.

Every field reference below (`opp["tactical_card"]["stop_loss"]`, etc.) reflects
this corrected, verified shape.

## 4. Entry Logic (`open_new_positions`)

Called once per day, after `run_full_scan()` in `run_full_cycle()`.

1. **Candidate filter:** from `ranked["bullish"]`, take opportunities where
   `opp["tactical_card"]["confidence_tier"] == "HIGH"`. (`ranked["bearish"]` is not
   used — see long-only rationale below.)
2. **Skip existing:** skip any ticker that already has an `open` position.
3. **Long-only rationale:** the existing `stop_loss`/entry math throughout
   `portfolio.py` assumes price falls for risk are long positions (`capital_risk =
   (entry_price - sl_val) * qty`). Supporting bearish/short calls would need new
   risk math and is scoped out of this spec — can be a fast-follow if the long-only
   version proves out.
4. **Stop-loss:** use `opp["tactical_card"]["stop_loss"]` (see §3.1). Skip the
   candidate if this is `None` or `>= price` (bad/missing data).
5. **Take-profit:** use the nearest resistance level from `opp["resistance"]`
   (a list already present on every opportunity dict) where the level is above
   entry price, if any; otherwise fall back to a fixed **2:1 reward:risk** target:
   `take_profit = entry_price + 2 * (entry_price - stop_loss)`.
6. **Position sizing — risk 2% of virtual capital per trade:**
   ```
   risk_per_share = entry_price - stop_loss
   dollar_risk    = cash_balance_at_start_of_day * 0.02
   qty            = floor(dollar_risk / risk_per_share)
   ```
   Skip the trade if `risk_per_share <= 0` (bad data) or `qty < 1` (position too
   small to size, e.g. tiny risk gap on a high-price asset).
7. **Portfolio heat cap — 20% max:** before adding a new position, compute portfolio
   heat using `portfolio.py`'s exact existing formula — `total_capital_at_risk /
   total_portfolio_value * 100`, where `total_portfolio_value = cash_balance +
   open_positions_value` (line 438/445 of `portfolio.py`) — including the
   candidate position's own risk in the numerator. If that would push heat over
   20%, skip it (and log which candidates were skipped due to the cap, so it's
   visible in run logs — no silent truncation).
8. **Cash check:** skip if `qty * entry_price` exceeds available `cash_balance`.
9. On accept: deduct `qty * entry_price` from `cash_balance`, append the position,
   log at INFO level (`Paper BUY: <qty> <ticker> @ <price>, SL=<sl> TP=<tp>`).

## 5. Exit Logic (`check_exits`)

Called every hour (`run_alert_check`), for every position with `status == "open"`.

1. Get current price from the already-fetched `market_data` for that run (no extra
   API calls — reuses the hourly alert-check's existing price fetch).
2. If `current_price <= stop_loss`: close at `stop_loss` price, `exit_reason =
   "stop_loss"`.
3. Else if `current_price >= take_profit`: close at `take_profit` price,
   `exit_reason = "take_profit"`.
4. Else if `days_held > 30` (matches `portfolio.py`'s existing "stale trade"
   threshold): close at `current_price`, `exit_reason = "max_hold"`.
5. On close: credit `qty * exit_price` back to `cash_balance`, move the entry from
   `positions` to `closed_positions`, log the realized P&L.

## 6. Reporting

No new report code. `portfolio.py::calculate_portfolio_status()` already computes
and `reports/portfolio-status.md` already renders: open positions with live P&L,
proximity to SL/TP, portfolio heat, closed-position history, and win rate. Both
new hook functions call this at the end of their run so the report reflects every
change immediately. This file is already committed back to `main` by the existing
"Commit updated reports and state" CI step — no workflow changes needed.

## 7. Safety

- `PAPER_TRADING = True` module-level constant in `paper_trader.py`, checked at the
  top of both entry points — pure documentation/guard-rail since there is no
  broker-integration code anywhere in this repo to accidentally invoke.
- All sizing math is capped by the 2%-risk and 20%-heat rules above; a single bad
  signal cannot meaningfully damage the simulated account.
- Nothing here touches real credentials, real orders, or any external write besides
  the existing git-committed `portfolio.json` / `reports/portfolio-status.md`.

## 8. Testing Plan

Following `superpowers:test-driven-development`:
- Unit tests for `open_new_positions`: position sizing math, heat-cap rejection,
  duplicate-ticker skip, cash-check skip, take-profit fallback when no resistance
  level is present.
- Unit tests for `check_exits`: stop-loss trigger, take-profit trigger, max-hold
  trigger, P&L math on close, no-op when price is between SL/TP and under 30 days.
- One integration-style test: seed a `portfolio.json`, feed a synthetic scan result
  and a synthetic price series across simulated "days," and assert the final
  `cash_balance` and `closed_positions` match hand-calculated expectations.

## 9. Open Assumptions to Confirm on Review

- Starting virtual capital: **$100,000**.
- Risk per trade: **2%** of capital.
- Portfolio heat cap: **20%**.
- Entry confidence threshold: `tactical_card["confidence_tier"] == "HIGH"` only (excludes MEDIUM/LOW).
- Max hold period: **30 days**.

If any of these numbers should be different, this is the place to say so before
the implementation plan is written.
