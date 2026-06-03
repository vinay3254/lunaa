"""
continuous_learning.py
=======================
LUNA Autonomous Trading Agent — Continuous Learning Loop

Auto-improves model accuracy over time:
1. Checks outcomes of past predictions
2. Labels new records with actual forward returns
3. Retrains models when new data available
4. Monitors accuracy degradation
5. Auto-resets to rule-based if accuracy < 50%

Integrated into main luna.py run cycle.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

import feature_store
import ml_engine

logger = logging.getLogger("continuous_learning")
logger.setLevel(logging.INFO)

ACCURACY_TRACKING_PATH = "state/model-accuracy.json"
ACCURACY_THRESHOLD = 0.50
ACCURACY_WINDOW_DAYS = 14


def load_accuracy_tracking() -> dict:
    """Load model accuracy tracking from state/model-accuracy.json."""
    if not os.path.exists(ACCURACY_TRACKING_PATH):
        return {}
    try:
        with open(ACCURACY_TRACKING_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load accuracy tracking: %s", exc)
        return {}


def save_accuracy_tracking(data: dict) -> None:
    """Save model accuracy tracking."""
    os.makedirs(os.path.dirname(ACCURACY_TRACKING_PATH), exist_ok=True)
    try:
        with open(ACCURACY_TRACKING_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as exc:
        logger.error("Failed to save accuracy tracking: %s", exc)


def check_prediction_outcomes() -> dict:
    """
    Check outcomes of past predictions:
    - Load recent predictions from state/last-predictions.json
    - For predictions > 3d old: check 3d forward return
    - For predictions > 7d old: check 7d forward return
    - Update accuracy tracking
    """
    try:
        predictions_path = "state/last-predictions.json"
        if not os.path.exists(predictions_path):
            logger.debug("No predictions file found yet")
            return {"outcomes_checked": 0}
        
        with open(predictions_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        
        outcomes = {"outcomes_checked": 0, "by_asset_class": {}}
        now = datetime.now(tz=timezone.utc)
        
        for pred in predictions:
            timestamp = pred.get("timestamp")
            asset_class = pred.get("asset_class", "unknown")
            prediction = pred.get("bullish_probability", 0.0)
            
            if not timestamp:
                continue
            
            try:
                pred_time = datetime.fromisoformat(timestamp)
            except Exception:
                continue
            
            days_ago = (now - pred_time).days
            
            # For 3d+ old predictions, check if correct
            if days_ago >= 3:
                try:
                    import yfinance as yf
                    ticker = pred.get("asset")
                    if ticker:
                        current_price = yf.Ticker(ticker).info.get("currentPrice")
                        price_then = pred.get("price")
                        
                        if current_price and price_then:
                            ret = (current_price - price_then) / price_then * 100.0
                            was_correct = (ret > 2 and prediction > 0.5) or (ret < -2 and prediction < 0.5)
                            
                            if asset_class not in outcomes["by_asset_class"]:
                                outcomes["by_asset_class"][asset_class] = {
                                    "correct": 0,
                                    "total": 0,
                                    "accuracy": 0.0,
                                }
                            
                            outcomes["by_asset_class"][asset_class]["total"] += 1
                            if was_correct:
                                outcomes["by_asset_class"][asset_class]["correct"] += 1
                            
                            outcomes["outcomes_checked"] += 1
                            logger.debug("Checked %s 3d outcome: %s", ticker, "✓" if was_correct else "✗")
                except Exception as exc:
                    logger.debug("Failed to check outcome: %s", exc)
        
        # Calculate accuracies
        for cls, stats in outcomes["by_asset_class"].items():
            if stats["total"] > 0:
                stats["accuracy"] = stats["correct"] / stats["total"]
        
        if outcomes["outcomes_checked"] > 0:
            logger.info("Checked %d prediction outcomes", outcomes["outcomes_checked"])
        
        return outcomes
    
    except Exception as exc:
        logger.error("Failed to check prediction outcomes: %s", exc)
        return {"outcomes_checked": 0, "error": str(exc)}


def update_feature_labels() -> int:
    """Update feature store with new labels."""
    return feature_store.update_labels_with_forward_returns()


def check_model_accuracy_health(asset_class: str, min_accuracy: float = ACCURACY_THRESHOLD) -> bool:
    """
    Check if model accuracy has been low for too long.
    Returns True if model should be reset to rule-based.
    """
    tracking = load_accuracy_tracking()
    
    if asset_class not in tracking:
        return False  # No history yet
    
    history = tracking[asset_class].get("history", [])
    if not history:
        return False
    
    # Check last 14 days of accuracy
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=ACCURACY_WINDOW_DAYS)
    recent = []
    
    for entry in history:
        try:
            entry_time = datetime.fromisoformat(entry.get("timestamp", ""))
            if entry_time >= cutoff:
                recent.append(entry.get("accuracy", 1.0))
        except Exception:
            continue
    
    # If more than half of recent entries are below threshold, reset
    low_accuracy_count = sum(1 for acc in recent if acc < min_accuracy)
    if len(recent) >= 2 and low_accuracy_count >= len(recent) / 2:
        logger.warning("Model %s has low accuracy (%d/%d recent < %.0f%%) — resetting to rule-based",
                      asset_class, low_accuracy_count, len(recent), min_accuracy * 100)
        return True
    
    return False


def retrain_models_if_ready() -> dict:
    """
    Check if any asset class models need retraining.
    Returns details on retraining performed.
    """
    results = {
        "models_retrained": [],
        "models_skipped": [],
        "errors": [],
    }
    
    asset_classes = ["stock", "crypto", "forex", "commodity", "bond", "etf"]
    
    for asset_class in asset_classes:
        try:
            # Check if enough new labeled records
            labeled = feature_store.get_labeled_records_by_class(asset_class)
            
            if len(labeled) < feature_store.MIN_RECORDS_FOR_TRAINING:
                results["models_skipped"].append({
                    "asset_class": asset_class,
                    "reason": f"insufficient records ({len(labeled)} < {feature_store.MIN_RECORDS_FOR_TRAINING})",
                })
                continue
            
            # Check if model needs update
            last_trained = None
            try:
                model_path = f"state/models/{asset_class}_model.pkl"
                if os.path.exists(model_path):
                    last_trained = datetime.fromtimestamp(os.path.getmtime(model_path), tz=timezone.utc)
            except Exception:
                pass
            
            days_since_train = None
            if last_trained:
                days_since_train = (datetime.now(tz=timezone.utc) - last_trained).days
            
            # Retrain if: never trained OR 7+ days old OR new records > 50
            should_retrain = (
                days_since_train is None or
                days_since_train >= 7 or
                len(labeled) >= feature_store.MAX_RECORDS_PER_CLASS
            )
            
            if should_retrain:
                logger.info("Retraining %s model (%d labeled records)...", asset_class, len(labeled))
                
                accuracy = ml_engine.train_ensemble_models(asset_class)
                
                results["models_retrained"].append({
                    "asset_class": asset_class,
                    "records_used": len(labeled),
                    "accuracy": accuracy,
                })
                
                logger.info("✓ Model retrained: %s (accuracy: %.1f%%)", asset_class, accuracy * 100)
            else:
                results["models_skipped"].append({
                    "asset_class": asset_class,
                    "reason": f"recently trained ({days_since_train} days ago)",
                })
        
        except Exception as exc:
            logger.error("Failed to retrain %s model: %s", asset_class, exc)
            results["errors"].append({
                "asset_class": asset_class,
                "error": str(exc),
            })
    
    return results


def detect_prediction_flips(old_predictions: list[dict], new_predictions: list[dict]) -> list[dict]:
    """
    Detect if any high-confidence predictions have flipped direction.
    Used for alert generation.
    """
    flips = []
    
    # Build lookup
    old_by_asset = {p.get("asset"): p for p in old_predictions}
    
    for new_pred in new_predictions:
        asset = new_pred.get("asset")
        old_pred = old_by_asset.get(asset)
        
        if not old_pred:
            continue
        
        old_bull = old_pred.get("bullish_probability", 0.0)
        new_bull = new_pred.get("bullish_probability", 0.0)
        
        # Check if high confidence flip
        if (old_bull > 0.7 and new_bull < 0.3) or (old_bull < 0.3 and new_bull > 0.7):
            flips.append({
                "asset": asset,
                "old_bullish_prob": old_bull,
                "new_bullish_prob": new_bull,
                "flip_type": "bullish_to_bearish" if new_bull < old_bull else "bearish_to_bullish",
            })
            logger.warning("FLIP ALERT: %s changed from %.1f%% to %.1f%% bullish", 
                          asset, old_bull * 100, new_bull * 100)
    
    return flips


def run_continuous_learning_cycle() -> dict:
    """
    Execute the full continuous learning cycle.
    Should be called at the end of each LUNA run.
    """
    logger.info("=== Starting Continuous Learning Cycle ===")
    
    results = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "phases": {},
    }
    
    # 1. Check outcomes of past predictions
    logger.info("Phase 1: Checking prediction outcomes...")
    results["phases"]["check_outcomes"] = check_prediction_outcomes()
    
    # 2. Update feature store with new labels
    logger.info("Phase 2: Updating feature store labels...")
    results["phases"]["update_labels"] = {"updated": update_feature_labels()}
    
    # 3. Retrain models if ready
    logger.info("Phase 3: Checking if retraining needed...")
    results["phases"]["retrain_models"] = retrain_models_if_ready()
    
    logger.info("=== Continuous Learning Cycle Complete ===")
    
    return results
