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
