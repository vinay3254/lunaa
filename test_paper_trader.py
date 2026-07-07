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
