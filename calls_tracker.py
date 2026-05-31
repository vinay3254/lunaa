"""
calls_tracker.py
================
Manages flagged opportunity calls, outcome tracking (Module 6), and adaptive weights tuning (Module 7).
Tracks performance metrics and dynamically tunes scoring weights in state/scoring-weights.json and state/regime-weights.json.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
import numpy as np
import yfinance as yf

logger = logging.getLogger("calls_tracker")

CALLS_LOG_PATH = os.path.join(os.path.dirname(__file__), "state", "calls-log.json")
ARCHIVE_LOG_PATH = os.path.join(os.path.dirname(__file__), "state", "calls-archive.json")
GLOBAL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "state", "scoring-weights.json")
REGIME_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "state", "regime-weights.json")

# Default Scoring Weights
DEFAULT_GLOBAL_WEIGHTS = {
    "rsi": 2.0,
    "macd": 2.0,
    "ema_stack": 2.0,
    "volume": 1.0,
    "bb_position": 1.0,
    "sentiment": 1.0,
    "catalyst": 1.0
}

DEFAULT_REGIME_WEIGHTS = {
    "RISK-ON": {
        "rsi": 1.5,
        "macd": 2.5,
        "ema_stack": 2.5,
        "volume": 1.2,
        "bb_position": 0.8,
        "sentiment": 1.2,
        "catalyst": 1.3
    },
    "RISK-OFF": {
        "rsi": 2.5,
        "macd": 1.5,
        "ema_stack": 1.5,
        "volume": 1.5,
        "bb_position": 1.5,
        "sentiment": 0.5,
        "catalyst": 1.0
    },
    "TRANSITIONING": {
        "rsi": 2.0,
        "macd": 2.0,
        "ema_stack": 2.0,
        "volume": 1.0,
        "bb_position": 1.0,
        "sentiment": 1.0,
        "catalyst": 1.0
    }
}

def init_weights_files(overwrite: bool = False) -> None:
    """Initialise scoring-weights.json and regime-weights.json with defaults if they don't exist."""
    os.makedirs(os.path.dirname(GLOBAL_WEIGHTS_PATH), exist_ok=True)
    
    # 1. Global weights initialization
    if overwrite or not os.path.exists(GLOBAL_WEIGHTS_PATH):
        gw_data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "calls_evaluated": 0,
            "underperf_cycles_count": 0,
            "weights": DEFAULT_GLOBAL_WEIGHTS,
            "win_rates": {k: 0.5 for k in DEFAULT_GLOBAL_WEIGHTS},
            "history": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "change": "System initialisation: weights set to defaults",
                    "calls_basis": 0
                }
            ]
        }
        try:
            with open(GLOBAL_WEIGHTS_PATH, "w", encoding="utf-8") as f:
                json.dump(gw_data, f, indent=2, ensure_ascii=False)
            logger.info("Created global weights file with defaults.")
        except Exception as exc:
            logger.error("Failed to write default global weights: %s", exc)

    # 2. Regime-specific weights initialization
    if overwrite or not os.path.exists(REGIME_WEIGHTS_PATH):
        try:
            with open(REGIME_WEIGHTS_PATH, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_REGIME_WEIGHTS, f, indent=2, ensure_ascii=False)
            logger.info("Created regime weights file with defaults.")
        except Exception as exc:
            logger.error("Failed to write default regime weights: %s", exc)

def load_scoring_weights(regime: str = None) -> dict:
    """
    Load active weights mapping, supporting global and adaptive regime weights.
    Falls back to global weights if regime completed calls count is < 20.
    """
    init_weights_files()
    
    # 1. Load global weights
    global_weights = DEFAULT_GLOBAL_WEIGHTS.copy()
    try:
        if os.path.exists(GLOBAL_WEIGHTS_PATH):
            with open(GLOBAL_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                global_weights = data.get("weights", DEFAULT_GLOBAL_WEIGHTS)
    except Exception as exc:
        logger.error("Failed to read global weights: %s", exc)
        
    if not regime or regime == "UNKNOWN":
        return global_weights

    regime = regime.upper().strip()
    
    # 2. Count completed calls in this regime
    regime_completed_count = 0
    if os.path.exists(CALLS_LOG_PATH):
        try:
            with open(CALLS_LOG_PATH, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                calls = c_data.get("calls", [])
                regime_completed_count = sum(
                    1 for c in calls
                    if c.get("regime_at_call", "").upper().strip() == regime
                    and c.get("checked_7d")
                )
        except Exception as exc:
            logger.error("Failed to read calls log: %s", exc)
            
    if regime_completed_count < 20:
        logger.info(
            "Regime %s has %d completed calls (< 20). Using global weights as fallback.",
            regime, regime_completed_count
        )
        return global_weights
        
    # 3. Load regime-specific weights
    try:
        if os.path.exists(REGIME_WEIGHTS_PATH):
            with open(REGIME_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                rw_data = json.load(f)
                if regime in rw_data:
                    logger.info("Using adaptive regime-specific weights for %s.", regime)
                    return rw_data[regime]
    except Exception as exc:
        logger.error("Failed to read regime weights: %s", exc)
        
    return global_weights

def load_calls() -> dict:
    """Load calls log. Returns dict with 'calls' list."""
    os.makedirs(os.path.dirname(CALLS_LOG_PATH), exist_ok=True)
    if not os.path.exists(CALLS_LOG_PATH):
        return {"calls": []}
    try:
        with open(CALLS_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "calls" in data:
                return data
            return {"calls": []}
    except Exception as exc:
        logger.error("Failed to load calls log: %s", exc)
        return {"calls": []}

def save_calls(calls_data: dict) -> None:
    """Save calls log."""
    os.makedirs(os.path.dirname(CALLS_LOG_PATH), exist_ok=True)
    try:
        with open(CALLS_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(calls_data, f, indent=2, ensure_ascii=False)
        logger.debug("Calls log saved.")
    except Exception as exc:
        logger.error("Failed to save calls log: %s", exc)

def map_breakdown(breakdown_dict: dict) -> dict:
    """Map scanner components to database keys."""
    mapping = {
        "RSI": "rsi",
        "MACD": "macd",
        "EMA": "ema_stack",
        "Vol": "volume",
        "BB": "bb_position",
        "Sent": "sentiment",
        "Cat": "catalyst"
    }
    mapped = {}
    for scan_k, db_k in mapping.items():
        mapped[db_k] = int(breakdown_dict.get(scan_k, 0))
    return mapped

def log_new_call(opportunity: dict, current_price: float, regime: str, current_date_str: str | None = None) -> None:
    """Log a flagged setup opportunity into state/calls-log.json."""
    if not current_date_str:
        current_date_str = datetime.now(timezone.utc).isoformat()
        
    calls_data = load_calls()
    
    symbol = opportunity.get("ticker", "UNKNOWN")
    if symbol == "UNKNOWN":
        return
        
    direction = opportunity.get("direction", "neutral")
    if direction not in ["bullish", "bearish"]:
        return
        
    # Avoid duplicate logs for the same asset on the same calendar day
    today_date = current_date_str.split("T")[0]
    for c in calls_data["calls"]:
        c_date = c.get("timestamp", "").split("T")[0]
        if c.get("asset") == symbol and c_date == today_date and c.get("direction") == direction:
            logger.debug("Call for %s already logged today. Skipping.", symbol)
            return

    levels = opportunity.get("support_resistance", {})
    supports = levels.get("support", [])
    resistances = levels.get("resistance", [])
    
    score_breakdown = map_breakdown(opportunity.get("breakdown", {}))
    
    new_call = {
        "id": str(uuid.uuid4()),
        "timestamp": current_date_str,
        "asset": symbol,
        "asset_class": opportunity.get("asset_class", "stock"),
        "direction": direction,
        "score": int(opportunity.get("score", 0)),
        "score_breakdown": score_breakdown,
        "price_at_call": float(current_price),
        "key_level_support": float(supports[0]) if supports else None,
        "key_level_resistance": float(resistances[0]) if resistances else None,
        "regime_at_call": regime,
        "outcome_3d": None,
        "outcome_7d": None,
        "price_3d": None,
        "price_7d": None,
        "result_3d": None,
        "result_7d": None,
        "checked_3d": False,
        "checked_7d": False
    }
    
    calls_data["calls"].append(new_call)
    save_calls(calls_data)
    logger.info("Logged new %s call for %s at $%s.", direction, symbol, current_price)

def fetch_current_price(symbol: str) -> float | None:
    """Fetch current price of asset using yfinance with standard mappings."""
    try:
        t_symbol = symbol.strip().upper()
        # Crypto ticker translation
        if t_symbol in ["BTC", "ETH", "SOL", "ADA", "DOGE", "XRP", "BNB"]:
            t_symbol = f"{t_symbol}-USD"
            
        t = yf.Ticker(t_symbol)
        info = t.fast_info
        if info and "lastPrice" in info and not np.isnan(info["lastPrice"]):
            return float(info["lastPrice"])
            
        hist = t.history(period="1d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.error("Failed to fetch yfinance price for %s: %s", symbol, exc)
    return None

def check_pending_outcomes(current_time_str: str | None = None) -> None:
    """
    Every run, load calls-log.json and evaluate pending 3d and 7d outcomes
    relative to price_at_call.
    """
    if not current_time_str:
        current_time = datetime.now(timezone.utc)
    else:
        current_time = datetime.fromisoformat(current_time_str.replace("Z", "+00:00"))
        
    calls_data = load_calls()
    modified = False
    
    for c in calls_data["calls"]:
        call_time = datetime.fromisoformat(c["timestamp"].replace("Z", "+00:00"))
        days_passed = (current_time - call_time).days
        
        # 1. Evaluate 3d outcome (>= 3 days)
        if not c.get("checked_3d") and days_passed >= 3:
            price = fetch_current_price(c["asset"])
            if price is not None:
                move_pct = ((price - c["price_at_call"]) / c["price_at_call"]) * 100
                direction = c["direction"]
                
                if direction == "bullish":
                    if move_pct >= 1.0:
                        result = "WIN"
                    elif move_pct <= -1.0:
                        result = "LOSS"
                    else:
                        result = "NEUTRAL"
                else: # bearish
                    if move_pct <= -1.0:
                        result = "WIN"
                    elif move_pct >= 1.0:
                        result = "LOSS"
                    else:
                        result = "NEUTRAL"
                        
                c["outcome_3d"] = f"{move_pct:+.2f}%"
                c["price_3d"] = price
                c["result_3d"] = result
                c["checked_3d"] = True
                modified = True
                logger.info(
                    "Evaluated 3d outcome for %s (call date %s): %s (%+.2f%%)",
                    c["asset"], c["timestamp"].split("T")[0], result, move_pct
                )
                
        # 2. Evaluate 7d outcome (>= 7 days)
        if not c.get("checked_7d") and days_passed >= 7:
            price = fetch_current_price(c["asset"])
            if price is not None:
                move_pct = ((price - c["price_at_call"]) / c["price_at_call"]) * 100
                direction = c["direction"]
                
                if direction == "bullish":
                    if move_pct >= 1.0:
                        result = "WIN"
                    elif move_pct <= -1.0:
                        result = "LOSS"
                    else:
                        result = "NEUTRAL"
                else: # bearish
                    if move_pct <= -1.0:
                        result = "WIN"
                    elif move_pct >= 1.0:
                        result = "LOSS"
                    else:
                        result = "NEUTRAL"
                        
                c["outcome_7d"] = f"{move_pct:+.2f}%"
                c["price_7d"] = price
                c["result_7d"] = result
                c["checked_7d"] = True
                modified = True
                logger.info(
                    "Evaluated 7d outcome for %s (call date %s): %s (%+.2f%%)",
                    c["asset"], c["timestamp"].split("T")[0], result, move_pct
                )

    if modified:
        save_calls(calls_data)
        # Check if we need to adjust weights
        evaluate_and_adjust_weights()
        # Archive calls older than 90 days
        archive_old_calls()
        
        # Automatically trigger performance report regeneration when outcomes are verified
        try:
            from reporter import generate_performance_report
            generate_performance_report()
        except Exception as exc:
            logger.error("Failed to automatically regenerate performance report: %s", exc)

def _calc_win_rate_for_component(completed_calls: list[dict], component: str) -> float | None:
    """Calculate the win rate (average of 3d and 7d WINs) for completed calls where component contributed."""
    comp_calls = [
        c for c in completed_calls
        if abs(c.get("score_breakdown", {}).get(component, 0)) > 0
    ]
    if not comp_calls:
        return None
        
    total_trials = 0
    total_wins = 0
    
    for c in comp_calls:
        # 3d
        r3 = c.get("result_3d")
        if r3:
            total_trials += 1
            if r3 == "WIN":
                total_wins += 1
        # 7d
        r7 = c.get("result_7d")
        if r7:
            total_trials += 1
            if r7 == "WIN":
                total_wins += 1
                
    if total_trials == 0:
        return None
    return float(total_wins) / total_trials

def evaluate_and_adjust_weights() -> None:
    """
    Check if we hit a new block of 20 completed calls. If so, adjust weights (global & regime-specific)
    based on component accuracy and check for safety reset thresholds.
    """
    init_weights_files()
    calls_data = load_calls()
    
    completed_calls = [c for c in calls_data["calls"] if c.get("checked_3d") and c.get("checked_7d")]
    total_completed = len(completed_calls)
    
    # Load global weights config
    try:
        with open(GLOBAL_WEIGHTS_PATH, "r", encoding="utf-8") as f:
            gw_data = json.load(f)
    except Exception:
        return
        
    last_evaluated = gw_data.get("calls_evaluated", 0)
    
    # 1. Adjust weights after every 20 completed calls
    if total_completed - last_evaluated >= 20:
        logger.info(
            "New block of 20 completed calls reached (Total completed: %d, Last evaluated: %d). Adjusting weights.",
            total_completed, last_evaluated
        )
        
        # Pull last 50 completed calls overall
        last_50_completed = completed_calls[-50:]
        
        # --- GLOBAL WEIGHT TUNING ---
        current_weights = gw_data.get("weights", DEFAULT_GLOBAL_WEIGHTS).copy()
        win_rates = {}
        history_logs = []
        
        for k in DEFAULT_GLOBAL_WEIGHTS:
            win_rate = _calc_win_rate_for_component(last_50_completed, k)
            if win_rate is not None:
                win_rates[k] = round(win_rate, 4)
                
                old_w = current_weights[k]
                # Component win rate >= 70%: increase weight by 0.2 (max 3.0)
                if win_rate >= 0.70:
                    current_weights[k] = round(min(3.0, current_weights[k] + 0.2), 1)
                # Component win rate < 40%: decrease weight by 0.2 (min 0.5)
                elif win_rate < 0.40:
                    current_weights[k] = round(max(0.5, current_weights[k] - 0.2), 1)
                    
                if current_weights[k] != old_w:
                    history_logs.append(
                        f"global {k} weight {old_w} → {current_weights[k]} (win rate {win_rate*100:.1f}%)"
                    )
            else:
                win_rates[k] = gw_data.get("win_rates", {}).get(k, 0.5)
                
        # Update global weights file
        gw_data["last_updated"] = datetime.now(timezone.utc).isoformat()
        gw_data["calls_evaluated"] = total_completed
        gw_data["weights"] = current_weights
        gw_data["win_rates"] = win_rates
        
        for log_msg in history_logs:
            gw_data["history"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "change": log_msg,
                "calls_basis": total_completed
            })
            logger.info("Weight Change: %s", log_msg)

        # --- REGIME-SPECIFIC WEIGHT TUNING ---
        try:
            with open(REGIME_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                rw_data = json.load(f)
        except Exception:
            rw_data = DEFAULT_REGIME_WEIGHTS.copy()
            
        for regime in ["RISK-ON", "RISK-OFF", "TRANSITIONING"]:
            regime_completed = [
                c for c in completed_calls 
                if c.get("regime_at_call", "").upper().strip() == regime
            ]
            
            # Update regime weights if regime has >= 20 completed calls
            if len(regime_completed) >= 20:
                logger.info("Regime %s has %d completed calls. Tuning weights.", regime, len(regime_completed))
                r_last_50 = regime_completed[-50:]
                
                r_weights = rw_data.get(regime, DEFAULT_REGIME_WEIGHTS[regime]).copy()
                for k in DEFAULT_GLOBAL_WEIGHTS:
                    win_rate = _calc_win_rate_for_component(r_last_50, k)
                    if win_rate is not None:
                        old_rw = r_weights[k]
                        if win_rate >= 0.70:
                            r_weights[k] = round(min(3.0, r_weights[k] + 0.2), 1)
                        elif win_rate < 0.40:
                            r_weights[k] = round(max(0.5, r_weights[k] - 0.2), 1)
                            
                        if r_weights[k] != old_rw:
                            regime_log = f"{regime} {k} weight {old_rw} → {r_weights[k]} (win rate {win_rate*100:.1f}%)"
                            gw_data["history"].append({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "change": regime_log,
                                "calls_basis": total_completed
                            })
                            logger.info("Weight Change (%s): %s", regime, regime_log)
                rw_data[regime] = r_weights
                
        # Save regime weights
        try:
            with open(REGIME_WEIGHTS_PATH, "w", encoding="utf-8") as f:
                json.dump(rw_data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("Failed to save regime weights: %s", exc)

        # --- SAFETY MODEL RESET CHECK ---
        # Overall 7d win rate across the last 50 completed calls
        if len(last_50_completed) > 0:
            wins_7d = sum(1 for c in last_50_completed if c.get("result_7d") == "WIN")
            overall_7d_win_rate = float(wins_7d) / len(last_50_completed)
            logger.info("Overall 7-day win rate (last 50 calls): %.1f%%", overall_7d_win_rate * 100)
            
            if overall_7d_win_rate < 0.45:
                gw_data["underperf_cycles_count"] = gw_data.get("underperf_cycles_count", 0) + 1
                logger.warning(
                    "Underperformance detected (%d/3 consecutive cycles): %.1f%% < 45%%",
                    gw_data["underperf_cycles_count"], overall_7d_win_rate * 100
                )
                
                # Reached 3 consecutive cycles: trigger full reset
                if gw_data["underperf_cycles_count"] >= 3:
                    logger.critical("Underperforming for 3 consecutive cycles! Triggering Model Reset event.")
                    
                    # Reset global weights to default
                    gw_data["weights"] = DEFAULT_GLOBAL_WEIGHTS.copy()
                    gw_data["underperf_cycles_count"] = 0
                    gw_data["history"].append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "change": "MODEL RESET: overall win rate under 45% for 3 consecutive cycles. Scoring weights restored to defaults.",
                        "calls_basis": total_completed
                    })
                    
                    # Reset regime weights to default
                    try:
                        with open(REGIME_WEIGHTS_PATH, "w", encoding="utf-8") as f:
                            json.dump(DEFAULT_REGIME_WEIGHTS, f, indent=2, ensure_ascii=False)
                        logger.info("Regime-specific weights restored to defaults.")
                    except Exception as exc:
                        logger.error("Failed to reset default regime weights: %s", exc)
            else:
                gw_data["underperf_cycles_count"] = 0
                
        # Save global weights config
        try:
            with open(GLOBAL_WEIGHTS_PATH, "w", encoding="utf-8") as f:
                json.dump(gw_data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("Failed to save global weights: %s", exc)

def archive_old_calls() -> None:
    """Archive completed calls older than 90 days to state/calls-archive.json."""
    calls_data = load_calls()
    current_time = datetime.now(timezone.utc)
    
    active_calls = []
    archive_calls = []
    
    for c in calls_data["calls"]:
        call_time = datetime.fromisoformat(c["timestamp"].replace("Z", "+00:00"))
        age_days = (current_time - call_time).days
        
        # Check if older than 90 days
        if age_days > 90:
            archive_calls.append(c)
        else:
            active_calls.append(c)
            
    if archive_calls:
        logger.info("Archiving %d completed calls older than 90 days.", len(archive_calls))
        
        # Load existing archive
        archive_data = {"calls": []}
        if os.path.exists(ARCHIVE_LOG_PATH):
            try:
                with open(ARCHIVE_LOG_PATH, "r", encoding="utf-8") as f:
                    archive_data = json.load(f)
                    if not isinstance(archive_data, dict) or "calls" not in archive_data:
                        archive_data = {"calls": []}
            except Exception:
                pass
                
        archive_data["calls"].extend(archive_calls)
        
        # Save archive
        try:
            with open(ARCHIVE_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(archive_data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("Failed to write to calls archive: %s", exc)
            
        # Update active calls log
        calls_data["calls"] = active_calls
        save_calls(calls_data)
