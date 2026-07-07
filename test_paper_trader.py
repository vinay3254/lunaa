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
