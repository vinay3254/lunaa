"""
ml_engine.py
============
LUNA Autonomous Trading Agent — Machine Learning Scorer & Feature Store Module

Features calculated per asset are stored in state/feature-store.json.
Models are saved in state/models/ and retrained regime-aware.
StandardScaler and TimeSeriesSplit are used to prevent look-ahead bias.
Supports auto-bootstrapping with historical data for immediate use.
"""

from __future__ import annotations

import json
import logging
import os
import time
import pickle
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

import indicators

logger = logging.getLogger("ml_engine")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
FEATURE_STORE_PATH = "state/feature-store.json"
MODELS_DIR = "state/models"

# ---------------------------------------------------------------------------
# Load & Save Feature Store
# ---------------------------------------------------------------------------

def load_feature_store() -> list[dict]:
    """Load the feature store list from state/feature-store.json."""
    if not os.path.exists(FEATURE_STORE_PATH):
        # Ensure directories exist
        os.makedirs(os.path.dirname(FEATURE_STORE_PATH), exist_ok=True)
        with open(FEATURE_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []
    try:
        with open(FEATURE_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load feature store: %s", exc)
        return []


def save_feature_store(records: list[dict]) -> None:
    """Save the feature store list to state/feature-store.json, capping size to 10MB."""
    os.makedirs(os.path.dirname(FEATURE_STORE_PATH), exist_ok=True)
    
    # Cap size to 10MB: if exceeds, archive older records (rolling window)
    # 10MB of JSON is approximately 10,000 feature store records.
    if len(records) > 10000:
        logger.warning("Feature store exceeds 10MB rolling limit. Archiving %d oldest records.", len(records) - 10000)
        records = records[-10000:]
        
    try:
        with open(FEATURE_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)
    except Exception as exc:
        logger.error("Failed to save feature store: %s", exc)


def _check_crossover(fast: pd.Series, slow: pd.Series, lookback: int = 3) -> int:
    """Check golden/death cross in last *lookback* bars: 1 = golden, -1 = death, 0 = none."""
    if len(fast) < lookback + 1 or len(slow) < lookback + 1:
        return 0
    diff = fast - slow
    for i in range(len(diff) - 1, len(diff) - 1 - lookback, -1):
        if i <= 0:
            break
        prev = diff.iloc[i-1]
        curr = diff.iloc[i]
        if prev <= 0 < curr:  # crossed above (golden)
            return 1
        if prev >= 0 > curr:  # crossed below (death)
            return -1
    return 0


# ---------------------------------------------------------------------------
# Feature Engineering Scorer
# ---------------------------------------------------------------------------

def compute_asset_features(
    asset: dict, 
    macro_state: dict, 
    sentiment_score: float = 0.0, 
    news_volume: int = 0
) -> dict | None:
    """Calculate all engineering features for an asset."""
    ohlcv = asset.get("ohlcv")
    if ohlcv is None or not isinstance(ohlcv, pd.DataFrame) or len(ohlcv) < 50:
        return None

    close = ohlcv["close"].dropna()
    if len(close) < 20:
        return None

    ticker = asset.get("ticker", "UNKNOWN")
    price = asset.get("price", close.iloc[-1])
    asset_class = asset.get("asset_class", "stock")

    # 1. Price Momentum
    returns_1d = float((price - close.iloc[-2]) / close.iloc[-2] * 100.0) if len(close) >= 2 else 0.0
    returns_3d = float((price - close.iloc[-4]) / close.iloc[-4] * 100.0) if len(close) >= 4 else 0.0
    returns_7d = float((price - close.iloc[-8]) / close.iloc[-8] * 100.0) if len(close) >= 8 else 0.0
    returns_30d = float((price - close.iloc[-22]) / close.iloc[-22] * 100.0) if len(close) >= 22 else 0.0

    # 2. Technical Indicators
    rsi_14 = asset.get("rsi")
    if rsi_14 is None or np.isnan(rsi_14):
        rsi_14 = float(indicators.calculate_rsi(close, 14) or 50.0)
    
    rsi_7 = float(indicators.calculate_rsi(close, 7) or 50.0)

    macd_data = asset.get("macd")
    if macd_data is None or not isinstance(macd_data, dict) or "hist" not in macd_data:
        macd_data = indicators.calculate_macd(close) or {}
    
    macd_hist = macd_data.get("hist", [0.0])
    macd_histogram = float(macd_hist[-1] if isinstance(macd_hist, list) else macd_hist)
    
    crossover = macd_data.get("crossover")
    bars_since = macd_data.get("bars_since_crossover", 999)
    macd_signal_cross = 0
    if crossover == "bullish" and bars_since <= 3:
        macd_signal_cross = 1
    elif crossover == "bearish" and bars_since <= 3:
        macd_signal_cross = -1

    # EMAs
    ema20_val = asset.get("ema20") or indicators.calculate_ema(close, 20)
    ema50_val = asset.get("ema50") or indicators.calculate_ema(close, 50)
    ema200_val = asset.get("ema200") or indicators.calculate_ema(close, 200)

    price_vs_ema20 = float((price - ema20_val) / ema20_val * 100.0) if ema20_val else 0.0
    price_vs_ema50 = float((price - ema50_val) / ema50_val * 100.0) if ema50_val else 0.0
    price_vs_ema200 = float((price - ema200_val) / ema200_val * 100.0) if ema200_val else 0.0

    ema20_series = indicators.ema(close, 20)
    ema50_series = indicators.ema(close, 50)
    ema200_series = indicators.ema(close, 200)

    ema_20_50_cross = _check_crossover(ema20_series, ema50_series, 3)
    ema_50_200_cross = _check_crossover(ema50_series, ema200_series, 3)

    # BB
    bb_data = asset.get("bb") or asset.get("bollinger_bands") or indicators.calculate_bollinger_bands(close) or {}
    bb_lower = bb_data.get("lower", price)
    bb_upper = bb_data.get("upper", price)
    bb_middle = bb_data.get("middle", price)
    bb_position = float((price - bb_lower) / (bb_upper - bb_lower)) if (bb_upper - bb_lower) != 0 else 0.5
    bb_width = float((bb_upper - bb_lower) / bb_middle) if bb_middle else 0.0

    # ATR
    atr_val = asset.get("atr")
    if atr_val is None or np.isnan(atr_val):
        high = ohlcv["high"]
        low = ohlcv["low"]
        atr_val = indicators.calculate_atr(high, low, close)
    atr_pct = float(atr_val / price * 100.0) if atr_val else 0.0

    # Volume
    vol_data = asset.get("volume_anomaly", {})
    vol_avg = vol_data.get("avg_20d", 0.0)
    vol_curr = vol_data.get("current", 0.0)
    volume_ratio = float(vol_curr / vol_avg) if vol_avg > 0 else 1.0

    # 3. Market Context
    vix_level = float(macro_state.get("vix", 15.0))
    dxy_trend = float(macro_state.get("dxy_change_7d_pct", 0.0))
    regime = str(macro_state.get("regime", "RISK-ON"))
    btc_dominance = float(macro_state.get("btc_dominance", 50.0))
    spy_7d_return = float(macro_state.get("spy_7d_return", 0.0))
    tnx_level = float(macro_state.get("yield_10y", 4.0))

    return {
        "timestamp":          _now_iso(),
        "ticker":             ticker,
        "price":              price,
        "asset_class":        asset_class,
        "returns_1d":         returns_1d,
        "returns_3d":         returns_3d,
        "returns_7d":         returns_7d,
        "returns_30d":        returns_30d,
        "rsi_14":             rsi_14,
        "rsi_7":              rsi_7,
        "macd_histogram":     macd_histogram,
        "macd_signal_cross":  macd_signal_cross,
        "ema_20_50_cross":    ema_20_50_cross,
        "ema_50_200_cross":   ema_50_200_cross,
        "price_vs_ema20":     price_vs_ema20,
        "price_vs_ema50":     price_vs_ema50,
        "price_vs_ema200":    price_vs_ema200,
        "bb_position":        bb_position,
        "bb_width":           bb_width,
        "atr_pct":            atr_pct,
        "volume_ratio":       volume_ratio,
        "vix_level":          vix_level,
        "dxy_trend":          dxy_trend,
        "regime":             regime,
        "btc_dominance":      btc_dominance,
        "spy_7d_return":      spy_7d_return,
        "tnx_level":          tnx_level,
        "sentiment_score":    sentiment_score,
        "news_volume":        news_volume,
        "forward_return_3d":  None,
        "forward_return_7d":  None,
        "label_3d":           None,
        "label_7d":           None
    }


# ---------------------------------------------------------------------------
# Save Feature Record & Retrospective Updating
# ---------------------------------------------------------------------------

def record_asset_run(
    asset: dict, 
    macro_state: dict, 
    sentiment_score: float = 0.0, 
    news_volume: int = 0
) -> None:
    """Compute and append current run features to state/feature-store.json."""
    features = compute_asset_features(asset, macro_state, sentiment_score, news_volume)
    if features is None:
        return
        
    store = load_feature_store()
    
    # Avoid duplicate records for same asset in short time windows (e.g. 1 hour)
    now = datetime.now(timezone.utc)
    recent_dup = False
    for r in reversed(store):
        if r.get("ticker") == features["ticker"]:
            r_ts = datetime.fromisoformat(r["timestamp"])
            if (now - r_ts).total_seconds() < 3600:
                recent_dup = True
                break
                
    if not recent_dup:
        store.append(features)
        save_feature_store(store)
        logger.info("Recorded features in feature-store for %s", features["ticker"])


def update_labels() -> int:
    """Retrospectively update forward prices and label targets after 3 and 7 days."""
    store = load_feature_store()
    if not store:
        return 0

    now = datetime.now(timezone.utc)
    updated_count = 0
    tickers_to_update = []

    # Identify records requiring update
    for r in store:
        ts = datetime.fromisoformat(r["timestamp"])
        age_days = (now - ts).days
        
        # 3 day check
        if age_days >= 3 and r.get("label_3d") is None:
            tickers_to_update.append(r["ticker"])
        # 7 day check
        if age_days >= 7 and r.get("label_7d") is None:
            tickers_to_update.append(r["ticker"])

    if not tickers_to_update:
        return 0

    # Batch download daily prices for the relevant assets
    tickers_to_update = list(set(tickers_to_update))
    logger.info("Retrospectively updating forward outcomes for %d tickers...", len(tickers_to_update))
    
    try:
        # Download 15 days of data to cover any recent calendar/trading period
        df = yf.download(tickers_to_update, period="1mo", interval="1d", progress=False, auto_adjust=True)
    except Exception as exc:
        logger.error("Outcome batch download failed: %s", exc)
        return 0

    is_multi = isinstance(df.columns, pd.MultiIndex)

    for r in store:
        ticker = r["ticker"]
        ts = datetime.fromisoformat(r["timestamp"])
        age_days = (now - ts).days
        
        # Fetch matching series
        try:
            if is_multi:
                if ticker in df.columns.levels[0]:
                    closes = df[ticker]["Close"].dropna()
                else:
                    continue
            else:
                closes = df["Close"].dropna()
        except Exception:
            continue

        if closes.empty:
            continue

        signal_price = r["price"]
        
        # 3-day update
        if age_days >= 3 and r.get("label_3d") is None:
            target_date = ts + timedelta(days=3)
            # Find the closest subsequent trading day
            subsequent = closes[closes.index >= pd.Timestamp(target_date.date())]
            if not subsequent.empty:
                price_3d = float(subsequent.iloc[0])
                fwd_ret = (price_3d - signal_price) / signal_price * 100.0
                r["forward_return_3d"] = round(fwd_ret, 3)
                r["label_3d"] = 1 if fwd_ret > 2.0 else (-1 if fwd_ret < -2.0 else 0)
                updated_count += 1

        # 7-day update
        if age_days >= 7 and r.get("label_7d") is None:
            target_date = ts + timedelta(days=7)
            subsequent = closes[closes.index >= pd.Timestamp(target_date.date())]
            if not subsequent.empty:
                price_7d = float(subsequent.iloc[0])
                fwd_ret = (price_7d - signal_price) / signal_price * 100.0
                r["forward_return_7d"] = round(fwd_ret, 3)
                r["label_7d"] = 1 if fwd_ret > 2.0 else (-1 if fwd_ret < -2.0 else 0)
                updated_count += 1

    if updated_count > 0:
        save_feature_store(store)
        logger.info("Outcome updates complete: %d records modified.", updated_count)
        
    return updated_count


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "returns_1d", "returns_3d", "returns_7d", "returns_30d",
    "rsi_14", "rsi_7", "macd_histogram", "macd_signal_cross",
    "ema_20_50_cross", "ema_50_200_cross", "price_vs_ema20",
    "price_vs_ema50", "price_vs_ema200", "bb_position", "bb_width",
    "atr_pct", "volume_ratio", "vix_level", "dxy_trend",
    "btc_dominance", "spy_7d_return", "tnx_level", "sentiment_score", "news_volume"
]

def new_records_since_last_train() -> int:
    """Return count of newly labeled records since last model training."""
    # We can approximate this by comparing timestamps of models vs features
    store = load_feature_store()
    labeled = [r for r in store if r.get("label_7d") is not None]
    
    model_mtimes = []
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith(".pkl"):
                model_mtimes.append(os.path.getmtime(os.path.join(MODELS_DIR, f)))
                
    if not model_mtimes:
        return len(labeled) # Never trained
        
    last_train_time = max(model_mtimes)
    new_records = 0
    for r in labeled:
        # If timestamp is newer than last trained model file
        ts = datetime.fromisoformat(r["timestamp"])
        if ts.timestamp() > last_train_time:
            new_records += 1
            
    return new_records


def train_ensemble_models(asset_class: str = "stock") -> float:
    """Train ensemble for a specific asset class and return accuracy."""
    store = load_feature_store()
    labeled = [r for r in store if r.get("label_7d") is not None and r.get("asset_class") == asset_class]
    
    if len(labeled) < 50:
        logger.warning("Insufficient labeled records for %s (%d < 50)", asset_class, len(labeled))
        return 0.5
    
    df = pd.DataFrame(labeled)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    X = df[FEATURE_COLUMNS].fillna(0).astype(float)
    y = df["label_7d"].astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # TimeSeriesSplit to avoid look-ahead bias
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for train_idx, val_idx in tscv.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        lr = LogisticRegression(max_iter=1000)
        
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        
        # Ensemble voting accuracy
        rf_pred = rf.predict(X_val)
        gb_pred = gb.predict(X_val)
        lr_pred = lr.predict(X_val)
        
        ensemble_pred = np.sign(rf_pred + gb_pred + lr_pred)
        score = accuracy_score(y_val, ensemble_pred)
        cv_scores.append(score)
    
    avg_accuracy = np.mean(cv_scores) if cv_scores else 0.5
    
    # Train final on all data
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    rf_final = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb_final = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    lr_final = LogisticRegression(max_iter=1000)
    
    rf_final.fit(X_scaled, y)
    gb_final.fit(X_scaled, y)
    lr_final.fit(X_scaled, y)
    
    payload = {
        "scaler": scaler,
        "models": {
            "random_forest": rf_final,
            "gradient_boost": gb_final,
            "logistic": lr_final,
        },
        "accuracy": float(avg_accuracy),
        "sample_size": len(X),
    }
    
    model_path = os.path.join(MODELS_DIR, f"{asset_class}_general.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
    
    logger.info("Trained ensemble for %s: accuracy %.2f%% (%d records)", asset_class, avg_accuracy * 100, len(X))
    
    return avg_accuracy


def check_model_health(accuracy: float) -> bool:
    """Check if model accuracy is healthy. If it drops below 50% for 2 consecutive weeks,
    disable ML and return False.
    """
    health_path = os.path.join(MODELS_DIR, "model-health.json")
    now = datetime.now(timezone.utc)
    health = {
        "consecutive_days_below_50": 0,
        "last_checked": now.isoformat(),
        "disabled": False
    }
    
    if os.path.exists(health_path):
        try:
            with open(health_path, "r", encoding="utf-8") as f:
                health = json.load(f)
        except Exception:
            pass
            
    last_checked = datetime.fromisoformat(health.get("last_checked", now.isoformat()))
    days_elapsed = (now - last_checked).days
    
    if accuracy < 0.50:
        health["consecutive_days_below_50"] = health.get("consecutive_days_below_50", 0) + max(1, days_elapsed)
    else:
        health["consecutive_days_below_50"] = 0
        health["disabled"] = False
        
    if health["consecutive_days_below_50"] >= 14:
        if not health["disabled"]:
            logger.error("ML Model accuracy has been below 50%% for %d consecutive days. AUTO-RESETTING TO RULE-BASED SCORING.", 
                         health["consecutive_days_below_50"])
        health["disabled"] = True
    else:
        health["disabled"] = False
        
    health["last_checked"] = now.isoformat()
    try:
        with open(health_path, "w", encoding="utf-8") as f:
            json.dump(health, f, indent=2)
    except Exception:
        pass
        
    return not health["disabled"]


def train_models() -> dict[str, float]:
    """Train ensemble classifiers per asset class and regime.
    Uses TimeSeriesSplit and standard scaler.
    """
    store = load_feature_store()
    labeled = [r for r in store if r.get("label_7d") is not None]
    
    if len(labeled) < 50:
        logger.warning("Feature store only has %d labeled records (need 50). Skipping ML training.", len(labeled))
        return {}

    os.makedirs(MODELS_DIR, exist_ok=True)
    df_full = pd.DataFrame(labeled)
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
    df_full.sort_values("timestamp", inplace=True) # Strict chronological sorting

    accuracies = {}

    # Train per asset class: stocks, crypto
    for asset_class in ["stock", "crypto"]:
        df_ac = df_full[df_full["asset_class"] == asset_class]
        if len(df_ac) < 30:
            continue

        # Train General Asset Class Model
        X_gen = df_ac[FEATURE_COLUMNS].astype(float)
        y_gen = df_ac["label_7d"].astype(int)

        # Cross Validation using TimeSeriesSplit (No lookahead bias)
        tscv = TimeSeriesSplit(n_splits=max(2, min(5, len(df_ac) // 10)))
        scaler = StandardScaler()
        
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_gen):
            X_tr, X_val = X_gen.iloc[train_idx], X_gen.iloc[val_idx]
            y_tr, y_val = y_gen.iloc[train_idx], y_gen.iloc[val_idx]
            
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(X_tr_s, y_tr)
            preds = rf.predict(X_val_s)
            cv_scores.append(accuracy_score(y_val, preds))
            
        gen_accuracy = np.mean(cv_scores) if cv_scores else 0.5
        accuracies[f"{asset_class}_general"] = float(gen_accuracy)
        
        # Check model health and log auto-disable status
        check_model_health(gen_accuracy)

        # Fit final general models on all class data
        X_gen_s = scaler.fit_transform(X_gen)
        rf_final = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        gb_final = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        lr_final = LogisticRegression(max_iter=1000)

        rf_final.fit(X_gen_s, y_gen)
        gb_final.fit(X_gen_s, y_gen)
        lr_final.fit(X_gen_s, y_gen)

        model_payload = {
            "scaler": scaler,
            "models": {
                "random_forest": rf_final,
                "gradient_boost": gb_final,
                "logistic": lr_final,
            },
            "accuracy": gen_accuracy,
            "sample_size": len(df_ac)
        }
        
        with open(os.path.join(MODELS_DIR, f"{asset_class}_general.pkl"), "wb") as f:
            pickle.dump(model_payload, f)
        logger.info("Trained %s general model on %d samples. Accuracy: %.2f", asset_class, len(df_ac), gen_accuracy)

        # Train Regime-Aware Models: RISK-ON, RISK-OFF, TRANSITIONING
        for regime in ["RISK-ON", "RISK-OFF", "TRANSITIONING"]:
            df_reg = df_ac[df_ac["regime"] == regime]
            if len(df_reg) < 30:
                continue

            X_reg = df_reg[FEATURE_COLUMNS].astype(float)
            y_reg = df_reg["label_7d"].astype(int)

            reg_scaler = StandardScaler()
            X_reg_s = reg_scaler.fit_transform(X_reg)

            rf_reg = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            gb_reg = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            lr_reg = LogisticRegression(max_iter=1000)

            rf_reg.fit(X_reg_s, y_reg)
            gb_reg.fit(X_reg_s, y_reg)
            lr_reg.fit(X_reg_s, y_reg)

            reg_payload = {
                "scaler": reg_scaler,
                "models": {
                    "random_forest": rf_reg,
                    "gradient_boost": gb_reg,
                    "logistic": lr_reg,
                },
                "accuracy": gen_accuracy, # Use class accuracy as baseline or calculate local
                "sample_size": len(df_reg)
            }
            
            with open(os.path.join(MODELS_DIR, f"{asset_class}_{regime.replace('-', '_')}.pkl"), "wb") as f:
                pickle.dump(reg_payload, f)
            logger.info("Trained %s regime model (%s) on %d samples.", asset_class, regime, len(df_reg))
            
    return accuracies


# ---------------------------------------------------------------------------
# ML Prediction Engine
# ---------------------------------------------------------------------------

def predict_opportunity(asset: dict, macro_state: dict) -> dict:
    """Predict bullish/bearish probabilities using the Resolution Fallback Ladder.
    Regime-Specific Model -> General Class Model -> Rule-Based Fallback.
    """
    ticker = asset.get("ticker", "UNKNOWN")
    asset_class = asset.get("asset_class", "stock")
    regime = macro_state.get("regime", "RISK-ON")
    
    # Calculate features on the fly
    feats = compute_asset_features(asset, macro_state)
    
    # Baseline fallback payload
    fallback_res = {
        "asset": ticker,
        "bullish_probability": 0.0,
        "bearish_probability": 0.0,
        "neutral_probability": 1.0,
        "model_confidence": "LOW",
        "top_features": [],
        "model_accuracy_last_30d": 0.50,
        "sample_size": 0,
        "fallback": True
    }
    
    if feats is None:
        return fallback_res

    # Check if models are disabled by health tracker
    health_path = os.path.join(MODELS_DIR, "model-health.json")
    if os.path.exists(health_path):
        try:
            with open(health_path, "r", encoding="utf-8") as f:
                if json.load(f).get("disabled", False):
                    return fallback_res
        except Exception:
            pass

    # Construct input vector
    input_vector = pd.DataFrame([feats])[FEATURE_COLUMNS].astype(float)

    # Resolution Ladder
    model_payload = None
    model_name = ""
    
    # Try regime-specific model first
    regime_path = os.path.join(MODELS_DIR, f"{asset_class}_{regime.replace('-', '_')}.pkl")
    if os.path.exists(regime_path):
        try:
            with open(regime_path, "rb") as f:
                model_payload = pickle.load(f)
                model_name = f"Regime-{regime}"
        except Exception:
            pass

    # Try general model next
    if model_payload is None:
        general_path = os.path.join(MODELS_DIR, f"{asset_class}_general.pkl")
        if os.path.exists(general_path):
            try:
                with open(general_path, "rb") as f:
                    model_payload = pickle.load(f)
                    model_name = "General"
            except Exception:
                pass

    # Rule-Based fallback if no model exists
    if model_payload is None:
        return fallback_res

    try:
        scaler = model_payload["scaler"]
        models = model_payload["models"]
        accuracy = model_payload.get("accuracy", 0.50)
        sample_size = model_payload.get("sample_size", 0)

        # Scale input
        scaled_input = scaler.transform(input_vector)

        # Predict probabilities for each ensemble model
        preds = {}
        probs = []
        for name, m in models.items():
            prob = m.predict_proba(scaled_input)[0]
            probs.append(prob)
            preds[name] = int(m.predict(scaled_input)[0])

        # Average probability (bullish = index corresponding to label 1, bearish = -1, neutral = 0)
        # RF and GB might have different class orders. Check classes_ attribute
        # Standard classes: [-1, 0, 1]
        avg_prob = np.mean(probs, axis=0)
        classes = list(models["random_forest"].classes_)
        
        prob_dict = {c: p for c, p in zip(classes, avg_prob)}
        bull_prob = float(prob_dict.get(1, 0.0))
        bear_prob = float(prob_dict.get(-1, 0.0))
        neut_prob = float(prob_dict.get(0, 0.0))

        # Model confidence evaluation
        unique_votes = set(preds.values())
        if len(unique_votes) == 1:
            confidence = "HIGH"
        elif len(unique_votes) == 2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Calculate top feature drivers via Random Forest importances
        rf = models["random_forest"]
        importances = rf.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        
        top_features = []
        for i in sorted_indices[:3]:
            # Convert scaled value to float and assign importance weight
            val = float(input_vector.iloc[0, i])
            top_features.append((FEATURE_COLUMNS[i], round(importances[i], 2)))

        return {
            "asset": ticker,
            "bullish_probability": round(bull_prob, 2),
            "bearish_probability": round(bear_prob, 2),
            "neutral_probability": round(neut_prob, 2),
            "model_confidence": confidence,
            "top_features": top_features,
            "model_accuracy_last_30d": round(accuracy, 2),
            "sample_size": sample_size,
            "fallback": False,
            "model_type": model_name
        }
    except Exception as exc:
        logger.error("Prediction execution failed for %s: %s", ticker, exc)
        return fallback_res


# ---------------------------------------------------------------------------
# Auto-Bootstrapper (Resolves Cold Start immediately!)
# ---------------------------------------------------------------------------

def bootstrap_feature_store(force: bool = False) -> int:
    """Pre-populate the feature store with a rich historical dataset from yfinance.
    Calculates technical indicators historically to bootstrap the ML engine immediately.
    """
    store = load_feature_store()
    labeled = [r for r in store if r.get("label_7d") is not None]
    
    if len(labeled) >= 60 and not force:
        logger.info("Feature store already has %d labeled records. Bootstrapping skipped.", len(labeled))
        return 0

    logger.info("Initializing ML Engine auto-bootstrapper (pre-populating historical features)...")
    bootstrapped_records = []
    
    # Pre-select representative bootstrap tickers
    assets_to_bootstrap = [
        {"ticker": "NVDA", "class": "stock"},
        {"ticker": "AAPL", "class": "stock"},
        {"ticker": "MSFT", "class": "stock"},
        {"ticker": "AMZN", "class": "stock"},
        {"ticker": "BTC-USD", "class": "crypto"},
        {"ticker": "ETH-USD", "class": "crypto"}
    ]

    # Download 180 days of history
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=220)
    
    for item in assets_to_bootstrap:
        t = item["ticker"]
        ac = item["class"]
        
        try:
            logger.info("Bootstrapping %s (%s)...", t, ac)
            df = yf.download(t, start=start_date, end=end_date, interval="1d", progress=False, auto_adjust=True)
            if df is None or df.empty or len(df) < 100:
                continue
                
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"adj close": "close"})
            
            # Pad missing columns with nan
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    df[col] = float("nan")
                    
            df = df[["open", "high", "low", "close", "volume"]].dropna().copy()
            
            # Loop historically through dates
            # We start at index 60 (to ensure enough history for EMA 50) and stop 10 days before today
            for idx in range(60, len(df) - 10):
                # Slice historical data up to this point in time
                hist_slice = df.iloc[:idx+1].copy()
                hist_date = hist_slice.index[-1]
                
                # Mock a macro context for this day
                vix_level = 15.0 + np.random.normal(0, 3.0)
                vix_level = max(9.0, min(45.0, vix_level))
                regime = "RISK-ON" if vix_level < 18 else ("RISK-OFF" if vix_level > 24 else "TRANSITIONING")
                
                macro_state = {
                    "vix": vix_level,
                    "dxy_change_7d_pct": np.random.normal(0.1, 0.5),
                    "regime": regime,
                    "btc_dominance": 52.0 + np.random.normal(0, 1.0),
                    "spy_7d_return": np.random.normal(0.2, 1.0),
                    "yield_10y": 4.2 + np.random.normal(0, 0.2)
                }

                # Compute standard indicators for this day
                try:
                    asset_indicator_data = indicators.calculate_all_indicators({
                        "symbol": t,
                        "ohlcv": hist_slice,
                        "price": hist_slice["close"].iloc[-1]
                    })
                except Exception:
                    continue

                asset_indicator_data["asset_class"] = ac
                asset_indicator_data["ticker"] = t
                asset_indicator_data["price"] = hist_slice["close"].iloc[-1]
                
                # Compute features
                feats = compute_asset_features(asset_indicator_data, macro_state)
                if feats is None:
                    continue
                
                # Compute retrospective actual forward outcomes
                price_signal = hist_slice["close"].iloc[-1]
                
                # Close price 3 days later
                price_3d = float(df["close"].iloc[idx+3])
                fwd_ret_3d = (price_3d - price_signal) / price_signal * 100.0
                feats["forward_return_3d"] = round(fwd_ret_3d, 3)
                feats["label_3d"] = 1 if fwd_ret_3d > 2.0 else (-1 if fwd_ret_3d < -2.0 else 0)

                # Close price 7 days later
                price_7d = float(df["close"].iloc[idx+7])
                fwd_ret_7d = (price_7d - price_signal) / price_signal * 100.0
                feats["forward_return_7d"] = round(fwd_ret_7d, 3)
                feats["label_7d"] = 1 if fwd_ret_7d > 2.0 else (-1 if fwd_ret_7d < -2.0 else 0)
                
                # Overwrite timestamp
                feats["timestamp"] = hist_date.isoformat()
                
                bootstrapped_records.append(feats)
        except Exception as exc:
            logger.error("Auto-bootstrapping failed for ticker %s: %s", t, exc)

    if bootstrapped_records:
        # Merge with existing store
        existing = [r for r in store if r["ticker"] not in [x["ticker"] for x in assets_to_bootstrap]]
        combined = existing + bootstrapped_records
        save_feature_store(combined)
        logger.info("ML Engine successfully auto-bootstrapped! Populated %d records.", len(bootstrapped_records))
        
        # Train immediately
        train_models()
        
    return len(bootstrapped_records)


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()
