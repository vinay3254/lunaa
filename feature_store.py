"""
feature_store.py
================
LUNA Autonomous Trading Agent — Feature Store & Data Labeling Pipeline

Manages:
1. Feature calculation for all assets (24 technical + market context features)
2. Persistent storage in state/feature-store.json
3. Auto-labeling pipeline (3d/7d forward returns)
4. Rolling dataset management (max 500 records per asset class)
5. Training/testing dataset splits
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("feature_store")
logger.setLevel(logging.INFO)

FEATURE_STORE_PATH = "state/feature-store.json"
MAX_RECORDS_PER_CLASS = 500
MIN_RECORDS_FOR_TRAINING = 50

# Feature schema
FEATURE_SCHEMA = {
    # Price momentum
    "returns_1d": float,
    "returns_3d": float,
    "returns_7d": float,
    "returns_30d": float,
    
    # Technical indicators
    "rsi_14": float,
    "rsi_7": float,
    "macd_histogram": float,
    "macd_signal_cross": int,      # 1 = bullish, -1 = bearish, 0 = none
    "ema_20_50_cross": int,        # 1 = golden, -1 = death, 0 = none
    "ema_50_200_cross": int,       # 1 = golden, -1 = death, 0 = none
    "price_vs_ema20": float,       # % above/below
    "price_vs_ema50": float,
    "price_vs_ema200": float,
    "bb_position": float,          # 0=lower, 0.5=mid, 1=upper
    "bb_width": float,
    "atr_pct": float,              # volatility
    "volume_ratio": float,         # current / 20d avg
    
    # Market context
    "vix_level": float,
    "dxy_trend": float,            # 7d return
    "regime": str,                 # RISK-ON / RISK-OFF / TRANSITIONING
    "btc_dominance": float,
    "spy_7d_return": float,
    "tnx_level": float,            # 10Y yield
    
    # Sentiment
    "sentiment_score": float,      # -1 to +1
    "news_volume": int,
    
    # Asset class
    "asset_class": str,            # stock/crypto/forex/commodity/bond/etf
    
    # Target (filled after 3d/7d)
    "forward_return_3d": float,
    "forward_return_7d": float,
    "label_3d": int,               # 1 = up >2%, -1 = down >2%, 0 = flat
    "label_7d": int,
    
    # Metadata
    "timestamp": str,              # ISO 8601
    "asset": str,                  # ticker
}


def load_feature_store() -> list[dict]:
    """Load feature store from state/feature-store.json."""
    os.makedirs(os.path.dirname(FEATURE_STORE_PATH), exist_ok=True)
    if not os.path.exists(FEATURE_STORE_PATH):
        return []
    try:
        with open(FEATURE_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load feature store: %s", exc)
        return []


def save_feature_store(records: list[dict]) -> None:
    """Save feature store, maintaining 10MB limit by archiving old records."""
    os.makedirs(os.path.dirname(FEATURE_STORE_PATH), exist_ok=True)
    
    # Keep only last 10000 records if exceeded
    if len(records) > 10000:
        logger.warning("Feature store exceeding 10MB. Archiving %d oldest records.", len(records) - 10000)
        records = records[-10000:]
    
    try:
        with open(FEATURE_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)
    except Exception as exc:
        logger.error("Failed to save feature store: %s", exc)


def record_features(
    asset: str,
    asset_class: str,
    features: dict,
) -> None:
    """Record calculated features for an asset."""
    records = load_feature_store()
    
    record = {
        "asset": asset,
        "asset_class": asset_class,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        **features,
    }
    
    records.append(record)
    
    # Keep rolling last 500 per asset class
    by_class = {}
    for r in records:
        cls = r.get("asset_class")
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(r)
    
    # Trim to 500 per class
    trimmed = []
    for cls, recs in by_class.items():
        trimmed.extend(recs[-MAX_RECORDS_PER_CLASS:])
    
    save_feature_store(trimmed)
    logger.debug("Recorded features for %s (%s)", asset, asset_class)


def update_labels_with_forward_returns() -> int:
    """
    For records 3+ days old without label_3d, fetch actual price and label.
    For records 7+ days old without label_7d, fetch actual price and label.
    Returns number of records updated.
    """
    records = load_feature_store()
    updated = 0
    
    now = datetime.now(tz=timezone.utc)
    
    for record in records:
        asset = record.get("asset")
        timestamp = record.get("timestamp")
        
        if not asset or not timestamp:
            continue
        
        try:
            rec_time = datetime.fromisoformat(timestamp)
        except Exception:
            continue
        
        days_ago = (now - rec_time).days
        
        # Update label_3d if 3+ days old and not yet labeled
        if days_ago >= 3 and record.get("label_3d") is None:
            try:
                current_price = yf.Ticker(asset).info.get("currentPrice")
                if current_price and record.get("price") is not None:
                    price_then = record["price"]
                    ret = (current_price - price_then) / price_then * 100.0
                    record["forward_return_3d"] = ret
                    record["label_3d"] = 1 if ret > 2 else (-1 if ret < -2 else 0)
                    updated += 1
                    logger.debug("Labeled %s 3d: %.2f%% → label %d", asset, ret, record["label_3d"])
            except Exception as exc:
                logger.debug("Failed to label %s 3d: %s", asset, exc)
        
        # Update label_7d if 7+ days old and not yet labeled
        if days_ago >= 7 and record.get("label_7d") is None:
            try:
                current_price = yf.Ticker(asset).info.get("currentPrice")
                if current_price and record.get("price") is not None:
                    price_then = record["price"]
                    ret = (current_price - price_then) / price_then * 100.0
                    record["forward_return_7d"] = ret
                    record["label_7d"] = 1 if ret > 2 else (-1 if ret < -2 else 0)
                    updated += 1
                    logger.debug("Labeled %s 7d: %.2f%% → label %d", asset, ret, record["label_7d"])
            except Exception as exc:
                logger.debug("Failed to label %s 7d: %s", asset, exc)
    
    save_feature_store(records)
    if updated > 0:
        logger.info("Updated %d record labels", updated)
    
    return updated


def get_labeled_records_by_class(asset_class: str) -> list[dict]:
    """Get all fully labeled records (both label_3d and label_7d) for an asset class."""
    records = load_feature_store()
    labeled = [
        r for r in records
        if r.get("asset_class") == asset_class
        and r.get("label_3d") is not None
        and r.get("label_7d") is not None
    ]
    return labeled


def new_records_since_last_train(asset_class: str, threshold: int = 20) -> bool:
    """Check if there are enough new labeled records since last training."""
    records = get_labeled_records_by_class(asset_class)
    return len(records) >= threshold


def get_training_data_for_class(asset_class: str, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get training and testing data for an asset class.
    Returns: X_train, X_test, y_train, y_test (using label_7d as target)
    
    Uses TimeSeriesSplit to avoid look-ahead bias.
    """
    records = get_labeled_records_by_class(asset_class)
    
    if len(records) < MIN_RECORDS_FOR_TRAINING:
        logger.warning("Insufficient labeled records for %s (%d < %d)", asset_class, len(records), MIN_RECORDS_FOR_TRAINING)
        return None, None, None, None
    
    # Build dataframe
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Feature columns (exclude metadata and targets)
    exclude_cols = {"asset", "asset_class", "timestamp", "forward_return_3d", "forward_return_7d", "label_3d", "label_7d"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Fill NaN in features
    X = df[feature_cols].fillna(0)
    y = df["label_7d"]
    
    # Time series split: first 80% train, last 20% test (chronological order)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info("Training data for %s: %d train records, %d test records", asset_class, len(X_train), len(X_test))
    
    return X_train, X_test, y_train, y_test


def get_feature_importance_df(feature_cols: list[str], importances: list[float]) -> pd.DataFrame:
    """Format feature importances for reporting."""
    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    })
    return df.sort_values("importance", ascending=False)
