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


def test_open_new_positions_preserves_fractional_quantities(monkeypatch):
    import portfolio as portfolio_module
    monkeypatch.setattr(portfolio_module, "calculate_portfolio_status",
                         lambda market_data, silent=False: {"total_portfolio_value": 100_000.0, "portfolio_heat": 0.0})

    # dollar_risk = 100_000 * 0.02 = 2000; risk_per_share = 101-98 = 3; qty = round(2000/3, 6) = 666.666667
    # This test ensures that fractional quantities are preserved (not floored/truncated to integers)
    ranked = {"bullish": [_opportunity("BTC", price=101.0, stop_loss=98.0, resistance=[110.0])]}
    portfolio = _empty_portfolio()

    result = paper_trader.open_new_positions(ranked, portfolio, market_data={})

    assert len(result["positions"]) == 1
    pos = result["positions"][0]
    # The key assertion: qty must be the exact fractional value, not an integer
    assert pos["quantity"] == 666.666667
    assert pos["asset"] == "BTC"
    assert pos["entry_price"] == 101.0
    assert pos["stop_loss"] == 98.0
    assert pos["take_profit"] == 110.0


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


def test_open_new_positions_day1_bootstrap_survives_real_calculate_portfolio_status(monkeypatch, tmp_path):
    """
    Regression test for the Day-1 bootstrap staleness bug: on a brand-new
    account, bootstrap_portfolio() sets cash_balance=100_000.0 in memory,
    but portfolio.json on disk is still the pristine cash_balance=0.0
    template (the caller only calls save_portfolio() AFTER
    open_new_positions() returns). open_new_positions() sources its sizing
    baseline from portfolio.calculate_portfolio_status(), which does its
    OWN internal load_portfolio() and ignores the in-memory `portfolio`
    dict entirely -- so on this very first run it would compute
    total_portfolio_value=0.0 from the stale disk file if not for the
    `status.get("total_portfolio_value") or portfolio.get("cash_balance", 0.0)`
    fallback in paper_trader.open_new_positions().

    Deliberately does NOT monkeypatch calculate_portfolio_status, so this
    exercises the real disk-reading code path end-to-end.
    """
    import json
    import portfolio as portfolio_module

    fresh_portfolio_file = tmp_path / "portfolio.json"
    fresh_portfolio_file.write_text(json.dumps({
        "positions": [], "closed_positions": [], "cash_balance": 0.0, "currency": "USD",
    }))
    monkeypatch.setattr(portfolio_module, "PORTFOLIO_PATH", fresh_portfolio_file)

    portfolio = paper_trader.bootstrap_portfolio(
        {"positions": [], "closed_positions": [], "cash_balance": 0.0, "currency": "USD"}
    )
    assert portfolio["cash_balance"] == 100_000.0  # in-memory only; disk file is still 0.0

    ranked = {"bullish": [_opportunity("NVDA", price=120.0, stop_loss=110.0, resistance=[140.0])]}
    result = paper_trader.open_new_positions(ranked, portfolio, market_data={})

    # This is the assertion that fails today if the `or` fallback is ever
    # "cleaned up" into a stricter `is not None` check: total_portfolio_value
    # would resolve to 0.0 from the stale on-disk file, every candidate's
    # qty would round to 0, and the silent `if qty <= 0: continue` branch
    # would skip the trade with no log output.
    assert len(result["positions"]) == 1
    assert result["positions"][0]["asset"] == "NVDA"
    assert result["positions"][0]["quantity"] == 200.0
