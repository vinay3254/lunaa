"""
indicators.py
=============
Technical indicator calculations for the autonomous trading research agent.

All functions accept pandas Series / DataFrames of OHLCV data and return
plain Python scalars or dicts so they are JSON-serialisable.

Dependencies:
    pip install pandas numpy ta
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pure pandas/numpy implementations — no pandas_ta required (Python 3.11 compat)
# ---------------------------------------------------------------------------

class _TA:
    """Minimal TA library shim using pure pandas/numpy."""

    @staticmethod
    def rsi(close: pd.Series, length: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
        avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ema(close: pd.Series, length: int) -> pd.Series:
        return close.ewm(span=length, adjust=False).mean()

    @staticmethod
    def macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame | None:
        fast_ema   = close.ewm(span=fast,   adjust=False).mean()
        slow_ema   = close.ewm(span=slow,   adjust=False).mean()
        macd_line  = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist        = macd_line - signal_line
        df = pd.DataFrame({
            f"MACD_{fast}_{slow}_{signal}":  macd_line,
            f"MACDh_{fast}_{slow}_{signal}": hist,
            f"MACDs_{fast}_{slow}_{signal}": signal_line,
        }, index=close.index)
        return df

    @staticmethod
    def bbands(
        close: pd.Series,
        length: int = 20,
        std: float = 2.0,
    ) -> pd.DataFrame | None:
        middle = close.rolling(window=length).mean()
        stddev = close.rolling(window=length).std(ddof=0)
        upper  = middle + std * stddev
        lower  = middle - std * stddev
        bandwidth = (upper - lower) / middle.replace(0, np.nan)
        pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
        df = pd.DataFrame({
            f"BBL_{length}_{std}": lower,
            f"BBM_{length}_{std}": middle,
            f"BBU_{length}_{std}": upper,
            f"BBB_{length}_{std}": bandwidth,
            f"BBP_{length}_{std}": pct_b,
        }, index=close.index)
        return df

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(com=length - 1, min_periods=length).mean()

    @staticmethod
    def stochrsi(
        close: pd.Series,
        length: int = 14,
        rsi_length: int = 14,
        k: int = 3,
        d: int = 3,
    ) -> pd.DataFrame | None:
        """Stochastic RSI."""
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=rsi_length - 1, min_periods=rsi_length).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=rsi_length - 1, min_periods=rsi_length).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))

        rsi_min = rsi.rolling(window=length).min()
        rsi_max = rsi.rolling(window=length).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)

        k_series = stoch_rsi.rolling(window=k).mean() * 100
        d_series = k_series.rolling(window=d).mean()

        df = pd.DataFrame({
            f"STOCHRSIk_{length}_{rsi_length}_{k}_{d}": k_series,
            f"STOCHRSId_{length}_{rsi_length}_{k}_{d}": d_series,
        }, index=close.index)
        return df

    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14,
    ) -> pd.DataFrame | None:
        """Average Directional Index."""
        prev_close = close.shift(1)
        prev_high  = high.shift(1)
        prev_low   = low.shift(1)

        plus_dm  = (high - prev_high).clip(lower=0)
        minus_dm = (prev_low - low).clip(lower=0)
        # Where both are positive, keep the larger one
        both_pos = (plus_dm > 0) & (minus_dm > 0)
        plus_dm[both_pos & (plus_dm < minus_dm)]  = 0
        minus_dm[both_pos & (minus_dm <= plus_dm)] = 0

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr14    = tr.ewm(com=length - 1, min_periods=length).mean()
        plus_di  = 100 * plus_dm.ewm(com=length - 1, min_periods=length).mean() / atr14.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(com=length - 1, min_periods=length).mean() / atr14.replace(0, np.nan)

        dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(com=length - 1, min_periods=length).mean()

        df = pd.DataFrame({
            f"ADX_{length}": adx,
            f"DMP_{length}": plus_di,
            f"DMN_{length}": minus_di,
        }, index=close.index)
        return df


ta = _TA()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    """Return a Python float or None for NaN / Inf / non-numeric values."""
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _require_length(series: pd.Series, minimum: int, name: str = "series") -> bool:
    """Log a warning and return False when a series is too short."""
    if len(series) < minimum:
        logger.warning(
            "%s has only %d rows; need at least %d.  Skipping calculation.",
            name, len(series), minimum,
        )
        return False
    return True


def _dropna_series(*series: pd.Series) -> tuple[pd.Series, ...]:
    """Align and drop rows where ANY of the supplied series has NaN."""
    df = pd.concat(list(series), axis=1).dropna()
    return tuple(df.iloc[:, i] for i in range(df.shape[1]))


# ---------------------------------------------------------------------------
# Individual indicator functions
# ---------------------------------------------------------------------------

def calculate_rsi(close: pd.Series, period: int = 14) -> float | None:
    """
    Relative Strength Index (RSI).

    Parameters
    ----------
    close  : closing-price series
    period : look-back window (default 14)

    Returns
    -------
    Most-recent RSI value as a float, or None if it cannot be computed.
    """
    if not _require_length(close, period + 1, "close"):
        return None
    try:
        rsi_series = ta.rsi(close, length=period)
        if rsi_series is None or rsi_series.dropna().empty:
            logger.debug("RSI returned an empty series.")
            return None
        return _safe_float(rsi_series.iloc[-1])
    except Exception:
        logger.exception("Error computing RSI.")
        return None


def calculate_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict:
    """
    MACD line, signal line, histogram and crossover state.

    Parameters
    ----------
    close  : closing-price series
    fast   : fast EMA period   (default 12)
    slow   : slow EMA period   (default 26)
    signal : signal EMA period (default  9)

    Returns
    -------
    dict with keys:
        macd      – float | None
        signal    – float | None
        hist      – float | None
        crossover – 'bullish' | 'bearish' | 'none'
    """
    default: dict = {"macd": None, "signal": None, "hist": None, "crossover": "none"}
    if not _require_length(close, slow + signal, "close"):
        return default
    try:
        macd_df = ta.macd(close, fast=fast, slow=slow, signal=signal)
        if macd_df is None or macd_df.dropna().empty:
            logger.debug("MACD returned an empty DataFrame.")
            return default

        # pandas_ta column names: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        macd_col   = [c for c in macd_df.columns if c.startswith("MACD_")]
        hist_col   = [c for c in macd_df.columns if c.startswith("MACDh_")]
        signal_col = [c for c in macd_df.columns if c.startswith("MACDs_")]

        if not (macd_col and hist_col and signal_col):
            logger.warning("Unexpected MACD column names: %s", list(macd_df.columns))
            return default

        macd_val   = _safe_float(macd_df[macd_col[0]].iloc[-1])
        signal_val = _safe_float(macd_df[signal_col[0]].iloc[-1])
        hist_val   = _safe_float(macd_df[hist_col[0]].iloc[-1])

        # Crossover: compare last two MACD-minus-signal values
        crossover = "none"
        if len(macd_df.dropna()) >= 2:
            prev_diff = _safe_float(
                macd_df[macd_col[0]].iloc[-2] - macd_df[signal_col[0]].iloc[-2]
            )
            curr_diff = _safe_float(
                macd_df[macd_col[0]].iloc[-1] - macd_df[signal_col[0]].iloc[-1]
            )
            if prev_diff is not None and curr_diff is not None:
                if prev_diff < 0 and curr_diff >= 0:
                    crossover = "bullish"
                elif prev_diff > 0 and curr_diff <= 0:
                    crossover = "bearish"

        return {
            "macd": macd_val,
            "signal": signal_val,
            "hist": hist_val,
            "crossover": crossover,
        }
    except Exception:
        logger.exception("Error computing MACD.")
        return default


def calculate_ema(close: pd.Series, period: int) -> float | None:
    """
    Exponential Moving Average.

    Parameters
    ----------
    close  : closing-price series
    period : EMA window

    Returns
    -------
    Most-recent EMA value as a float, or None.
    """
    if not _require_length(close, period, "close"):
        return None
    try:
        ema_series = ta.ema(close, length=period)
        if ema_series is None or ema_series.dropna().empty:
            return None
        return _safe_float(ema_series.iloc[-1])
    except Exception:
        logger.exception("Error computing EMA(period=%d).", period)
        return None


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> dict:
    """
    Bollinger Bands with squeeze detection.

    Squeeze is defined as: current BB width < 20-period rolling-average
    BB width * 0.8

    Parameters
    ----------
    close   : closing-price series
    period  : look-back window (default 20)
    std_dev : standard-deviation multiplier (default 2)

    Returns
    -------
    dict with keys:
        upper      – float | None
        middle     – float | None
        lower      – float | None
        width      – float | None  (normalised: (upper-lower)/middle)
        pct_b      – float | None  (%B indicator)
        is_squeeze – bool
    """
    default: dict = {
        "upper": None, "middle": None, "lower": None,
        "width": None, "pct_b": None, "is_squeeze": False,
    }
    if not _require_length(close, period, "close"):
        return default
    try:
        bb_df = ta.bbands(close, length=period, std=std_dev)
        if bb_df is None or bb_df.dropna().empty:
            logger.debug("Bollinger Bands returned an empty DataFrame.")
            return default

        # pandas_ta column names: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        lower_col  = [c for c in bb_df.columns if c.startswith("BBL_")]
        middle_col = [c for c in bb_df.columns if c.startswith("BBM_")]
        upper_col  = [c for c in bb_df.columns if c.startswith("BBU_")]
        bband_col  = [c for c in bb_df.columns if c.startswith("BBB_")]  # bandwidth
        pct_b_col  = [c for c in bb_df.columns if c.startswith("BBP_")]  # %B

        if not (lower_col and middle_col and upper_col):
            logger.warning("Unexpected BB column names: %s", list(bb_df.columns))
            return default

        upper_val  = _safe_float(bb_df[upper_col[0]].iloc[-1])
        middle_val = _safe_float(bb_df[middle_col[0]].iloc[-1])
        lower_val  = _safe_float(bb_df[lower_col[0]].iloc[-1])

        # Normalised width = (upper - lower) / middle
        width_val: float | None = None
        if upper_val is not None and lower_val is not None and middle_val:
            width_val = (upper_val - lower_val) / middle_val

        # %B from pandas_ta (already normalised 0-1)
        pct_b_val: float | None = None
        if pct_b_col:
            pct_b_val = _safe_float(bb_df[pct_b_col[0]].iloc[-1])
        elif upper_val is not None and lower_val is not None:
            band_range = upper_val - lower_val
            current_price = _safe_float(close.iloc[-1])
            if band_range and current_price is not None:
                pct_b_val = (current_price - lower_val) / band_range

        # Squeeze: current width < rolling-average width * 0.8
        is_squeeze = False
        if bband_col:
            bw_series = bb_df[bband_col[0]].dropna()
        else:
            # Reconstruct bandwidth series: (upper - lower) / middle
            bw_series = (
                (bb_df[upper_col[0]] - bb_df[lower_col[0]])
                / bb_df[middle_col[0]]
            ).dropna()

        if len(bw_series) >= period and width_val is not None:
            rolling_avg_width = float(bw_series.rolling(period).mean().iloc[-1])
            squeeze_threshold = rolling_avg_width * 0.8
            current_bw = float(bw_series.iloc[-1])
            is_squeeze = bool(current_bw < squeeze_threshold)

        return {
            "upper": upper_val,
            "middle": middle_val,
            "lower": lower_val,
            "width": width_val,
            "pct_b": pct_b_val,
            "is_squeeze": is_squeeze,
        }
    except Exception:
        logger.exception("Error computing Bollinger Bands.")
        return default


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> float | None:
    """
    Average True Range.

    Parameters
    ----------
    high, low, close : OHLC component series (must be aligned)
    period           : ATR window (default 14)

    Returns
    -------
    Most-recent ATR value as a float, or None.
    """
    if not _require_length(close, period + 1, "close"):
        return None
    try:
        atr_series = ta.atr(high, low, close, length=period)
        if atr_series is None or atr_series.dropna().empty:
            return None
        return _safe_float(atr_series.iloc[-1])
    except Exception:
        logger.exception("Error computing ATR.")
        return None


def calculate_support_resistance(
    ohlcv: pd.DataFrame,
    lookback: int = 90,
) -> dict:
    """
    Pivot-based support and resistance levels.

    Algorithm
    ---------
    1. Restrict to the last ``lookback`` rows.
    2. Identify pivot highs  (local maxima of the high column).
    3. Identify pivot lows   (local minima of the low column).
    4. A bar is a pivot high when its high > the highs of the two bars on
       either side.  Pivot lows are symmetric.
    5. Return the 3 resistance levels closest to (and above) the current
       price and the 3 support levels closest to (and below) the current
       price.  If there are not enough pivots above/below, include the
       closest pivots regardless of direction.

    Parameters
    ----------
    ohlcv    : DataFrame with at minimum columns ['high', 'low', 'close']
    lookback : number of most-recent bars to analyse (default 90)

    Returns
    -------
    dict with keys:
        support    – list[float]  (ascending, closest first)
        resistance – list[float]  (ascending, closest first)
    """
    default: dict = {"support": [], "resistance": []}
    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(set(ohlcv.columns)):
        logger.warning(
            "ohlcv must have columns %s; found %s", required_cols, list(ohlcv.columns)
        )
        return default
    if len(ohlcv) < 5:
        return default

    try:
        window = ohlcv.tail(lookback).copy().reset_index(drop=True)
        current_price = _safe_float(window["close"].iloc[-1])
        if current_price is None:
            return default

        highs  = window["high"].values
        lows   = window["low"].values

        pivot_highs: list[float] = []
        pivot_lows:  list[float] = []

        # Wing of 2 bars on each side for robust pivot detection
        wing = 2
        for i in range(wing, len(window) - wing):
            left_highs  = highs[i - wing : i]
            right_highs = highs[i + 1 : i + wing + 1]
            left_lows   = lows[i - wing : i]
            right_lows  = lows[i + 1 : i + wing + 1]

            if highs[i] > max(left_highs) and highs[i] > max(right_highs):
                pivot_highs.append(float(highs[i]))

            if lows[i] < min(left_lows) and lows[i] < min(right_lows):
                pivot_lows.append(float(lows[i]))

        # Deduplicate pivots that are within 0.5 % of each other
        def _deduplicate(levels: list[float], tol_pct: float = 0.005) -> list[float]:
            if not levels:
                return []
            levels_sorted = sorted(levels)
            merged: list[float] = [levels_sorted[0]]
            for lvl in levels_sorted[1:]:
                if abs(lvl - merged[-1]) / (merged[-1] + 1e-10) > tol_pct:
                    merged.append(lvl)
                else:
                    merged[-1] = (merged[-1] + lvl) / 2   # average nearby levels
            return merged

        unique_highs = _deduplicate(pivot_highs)
        unique_lows  = _deduplicate(pivot_lows)

        # Resistance = pivot highs at or above current price
        resistance_levels = sorted(
            [h for h in unique_highs if h >= current_price]
        )
        # Support = pivot lows at or below current price
        support_levels = sorted(
            [l for l in unique_lows if l <= current_price],
            reverse=True,                                   # closest (highest) first
        )

        # Fallback: if we don't have 3, add from the other list sorted by distance
        if len(resistance_levels) < 3:
            extras = sorted(unique_highs, key=lambda x: abs(x - current_price))
            for h in extras:
                if h not in resistance_levels:
                    resistance_levels.append(h)
                if len(resistance_levels) >= 3:
                    break
            resistance_levels = sorted(resistance_levels)

        if len(support_levels) < 3:
            extras = sorted(unique_lows, key=lambda x: abs(x - current_price))
            for l in extras:
                if l not in support_levels:
                    support_levels.append(l)
                if len(support_levels) >= 3:
                    break
            support_levels = sorted(support_levels, reverse=True)

        return {
            "support":    [round(v, 6) for v in support_levels[:3]],
            "resistance": [round(v, 6) for v in resistance_levels[:3]],
        }
    except Exception:
        logger.exception("Error computing support/resistance.")
        return default


def detect_volume_anomaly(volume: pd.Series) -> dict:
    """
    Detect unusual volume spikes relative to the 20-day average.

    Parameters
    ----------
    volume : volume series

    Returns
    -------
    dict with keys:
        current    – float | None  (most-recent volume)
        avg_20d    – float | None  (20-bar rolling mean)
        ratio      – float | None  (current / avg_20d)
        is_anomaly – bool          (True when ratio > 2.0)
    """
    default: dict = {
        "current": None, "avg_20d": None, "ratio": None, "is_anomaly": False
    }
    if not _require_length(volume, 21, "volume"):
        return default
    try:
        current_vol = _safe_float(volume.iloc[-1])
        avg_20 = _safe_float(volume.iloc[-21:-1].mean())     # exclude current bar

        if current_vol is None or avg_20 is None or avg_20 == 0:
            return {**default, "current": current_vol, "avg_20d": avg_20}

        ratio = current_vol / avg_20
        return {
            "current":    current_vol,
            "avg_20d":    avg_20,
            "ratio":      round(ratio, 4),
            "is_anomaly": bool(ratio > 2.0),
        }
    except Exception:
        logger.exception("Error detecting volume anomaly.")
        return default


def calculate_correlation(
    close1: pd.Series,
    close2: pd.Series,
) -> float | None:
    """
    Pearson correlation between two closing-price series.

    The two series are inner-joined on their index before computing the
    correlation so mismatched timestamps are handled gracefully.

    Parameters
    ----------
    close1, close2 : closing-price series for the two assets

    Returns
    -------
    Pearson correlation as a float in [-1, 1], or None.
    """
    try:
        aligned = pd.concat(
            [close1.rename("a"), close2.rename("b")], axis=1, join="inner"
        ).dropna()
        if len(aligned) < 2:
            logger.warning("Too few aligned rows to compute correlation.")
            return None
        corr = aligned["a"].corr(aligned["b"])
        return _safe_float(corr)
    except Exception:
        logger.exception("Error computing correlation.")
        return None


def detect_rsi_divergence(
    price: pd.Series,
    rsi: pd.Series,
    lookback: int = 14,
) -> str:
    """
    Detect bullish or bearish RSI divergence over the last ``lookback`` bars.

    Logic
    -----
    • Bullish  divergence: price makes a lower low  BUT RSI makes a higher low.
    • Bearish  divergence: price makes a higher high BUT RSI makes a lower high.
    • No divergence otherwise.

    Both series are trimmed to the last ``lookback`` bars and NaNs are dropped
    before the comparison.

    Parameters
    ----------
    price    : closing-price series
    rsi      : pre-computed RSI series (aligned with price)
    lookback : number of bars to examine (default 14)

    Returns
    -------
    'bullish' | 'bearish' | 'none'
    """
    try:
        aligned = pd.concat(
            [price.rename("price"), rsi.rename("rsi")], axis=1, join="inner"
        ).dropna().tail(lookback)

        if len(aligned) < 4:
            return "none"

        price_vals = aligned["price"].values
        rsi_vals   = aligned["rsi"].values

        # Split into first half (older) and second half (newer) for comparison
        mid = len(aligned) // 2

        # Highs and lows in each half
        price_low_old  = float(np.min(price_vals[:mid]))
        price_low_new  = float(np.min(price_vals[mid:]))
        price_high_old = float(np.max(price_vals[:mid]))
        price_high_new = float(np.max(price_vals[mid:]))

        rsi_low_old    = float(np.min(rsi_vals[:mid]))
        rsi_low_new    = float(np.min(rsi_vals[mid:]))
        rsi_high_old   = float(np.max(rsi_vals[:mid]))
        rsi_high_new   = float(np.max(rsi_vals[mid:]))

        # Bullish: price lower low + RSI higher low
        if price_low_new < price_low_old and rsi_low_new > rsi_low_old:
            return "bullish"

        # Bearish: price higher high + RSI lower high
        if price_high_new > price_high_old and rsi_high_new < rsi_high_old:
            return "bearish"

        return "none"
    except Exception:
        logger.exception("Error detecting RSI divergence.")
        return "none"


def detect_ema_crossover(
    close: pd.Series,
    fast_period: int = 50,
    slow_period: int = 200,
    lookback: int = 5,
) -> dict:
    """
    Detect 50/200 EMA crossovers within the last ``lookback`` bars.

    Parameters
    ----------
    close       : closing-price series
    fast_period : fast EMA period (default 50)
    slow_period : slow EMA period (default 200)
    lookback    : number of bars to look back for a fresh cross (default 5)

    Returns
    -------
    dict with keys:
        golden_cross – bool   (50 EMA crossed above 200 EMA recently)
        death_cross  – bool   (50 EMA crossed below 200 EMA recently)
        trend        – str    'bullish' | 'bearish' | 'neutral'
    """
    default: dict = {"golden_cross": False, "death_cross": False, "trend": "neutral"}
    if not _require_length(close, slow_period + lookback, "close"):
        return default
    try:
        fast_ema = ta.ema(close, length=fast_period)
        slow_ema = ta.ema(close, length=slow_period)

        if fast_ema is None or slow_ema is None:
            return default

        diff = (fast_ema - slow_ema).dropna()
        if len(diff) < lookback + 1:
            return default

        # Look at the last ``lookback`` bars for a sign change
        recent_diff  = diff.iloc[-lookback:]
        prev_diff    = diff.iloc[-(lookback + 1) : -1]

        golden_cross = bool(
            any(
                (p < 0 and c >= 0)
                for p, c in zip(prev_diff.values, recent_diff.values)
            )
        )
        death_cross = bool(
            any(
                (p > 0 and c <= 0)
                for p, c in zip(prev_diff.values, recent_diff.values)
            )
        )

        # Current trend based on sign of latest diff value
        latest_diff = _safe_float(diff.iloc[-1])
        if latest_diff is None:
            trend = "neutral"
        elif latest_diff > 0:
            trend = "bullish"
        elif latest_diff < 0:
            trend = "bearish"
        else:
            trend = "neutral"

        return {
            "golden_cross": golden_cross,
            "death_cross":  death_cross,
            "trend":        trend,
        }
    except Exception:
        logger.exception("Error detecting EMA crossover.")
        return default


def check_52w_proximity(
    price: float,
    high_52w: float,
    low_52w: float,
    proximity_threshold_pct: float = 3.0,
) -> dict:
    """
    Measure how close the current price is to its 52-week high/low.

    Parameters
    ----------
    price                    : current (last) price
    high_52w                 : 52-week high
    low_52w                  : 52-week low
    proximity_threshold_pct  : consider "near" when within this % (default 3 %)

    Returns
    -------
    dict with keys:
        near_high     – bool
        near_low      – bool
        pct_from_high – float  (negative: price below the high)
        pct_from_low  – float  (positive: price above the low)
    """
    default: dict = {
        "near_high": False, "near_low": False,
        "pct_from_high": None, "pct_from_low": None,
    }
    try:
        price    = float(price)
        high_52w = float(high_52w)
        low_52w  = float(low_52w)

        if high_52w <= 0 or low_52w <= 0 or price <= 0:
            return default

        pct_from_high = ((price - high_52w) / high_52w) * 100   # ≤ 0
        pct_from_low  = ((price - low_52w)  / low_52w)  * 100   # ≥ 0

        near_high = abs(pct_from_high) <= proximity_threshold_pct
        near_low  = abs(pct_from_low)  <= proximity_threshold_pct

        return {
            "near_high":     bool(near_high),
            "near_low":      bool(near_low),
            "pct_from_high": round(pct_from_high, 4),
            "pct_from_low":  round(pct_from_low,  4),
        }
    except (TypeError, ValueError, ZeroDivisionError):
        logger.exception("Error computing 52-week proximity.")
        return default


# ---------------------------------------------------------------------------
# Composite calculator
# ---------------------------------------------------------------------------

def calculate_all_indicators(asset_data: dict) -> dict:
    """
    Compute the full suite of indicators for a single asset.

    Expected keys in ``asset_data``
    --------------------------------
    ohlcv        – pd.DataFrame with columns: open, high, low, close, volume
                   (column names are matched case-insensitively)
    high_52w     – float  (optional; derived from ohlcv when absent)
    low_52w      – float  (optional; derived from ohlcv when absent)
    symbol       – str    (optional; used only for log messages)

    Returns
    -------
    dict with all computed indicator values.  Any indicator that could not
    be calculated is set to None (scalars) or an empty structure.
    """
    symbol = asset_data.get("symbol", "UNKNOWN")
    result: dict = {"symbol": symbol, "error": None}

    # ── Validate / normalise the DataFrame ──────────────────────────────────
    ohlcv: pd.DataFrame | None = asset_data.get("ohlcv")
    if ohlcv is None or not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
        msg = f"[{symbol}] 'ohlcv' key is missing or empty."
        logger.error(msg)
        result["error"] = msg
        return result

    # Normalise column names to lowercase
    ohlcv = ohlcv.copy()
    ohlcv.columns = [str(c).lower() for c in ohlcv.columns]

    required = {"open", "high", "low", "close", "volume"}
    missing  = required - set(ohlcv.columns)
    if missing:
        msg = f"[{symbol}] ohlcv is missing columns: {missing}"
        logger.error(msg)
        result["error"] = msg
        return result

    # Cast to numeric; drop rows that are fully NaN in core columns
    for col in required:
        ohlcv[col] = pd.to_numeric(ohlcv[col], errors="coerce")
    ohlcv.dropna(subset=list(required), inplace=True)

    if ohlcv.empty:
        msg = f"[{symbol}] ohlcv is empty after cleaning."
        logger.error(msg)
        result["error"] = msg
        return result

    close  = ohlcv["close"]
    high   = ohlcv["high"]
    low    = ohlcv["low"]
    volume = ohlcv["volume"]

    current_price = _safe_float(close.iloc[-1])
    result["current_price"] = current_price

    # ── 52-week high / low ──────────────────────────────────────────────────
    tail_252 = ohlcv.tail(252)   # ~1 trading year
    high_52w = asset_data.get("high_52w") or _safe_float(tail_252["high"].max())
    low_52w  = asset_data.get("low_52w")  or _safe_float(tail_252["low"].min())
    result["high_52w"] = high_52w
    result["low_52w"]  = low_52w

    # ── RSI ─────────────────────────────────────────────────────────────────
    result["rsi"] = calculate_rsi(close)

    # ── MACD ────────────────────────────────────────────────────────────────
    result["macd"] = calculate_macd(close)

    # ── EMAs ────────────────────────────────────────────────────────────────
    result["ema_9"]   = calculate_ema(close, 9)
    result["ema_20"]  = calculate_ema(close, 20)
    result["ema_50"]  = calculate_ema(close, 50)
    result["ema_200"] = calculate_ema(close, 200)

    # ── Bollinger Bands ─────────────────────────────────────────────────────
    result["bollinger_bands"] = calculate_bollinger_bands(close)

    # ── ATR ─────────────────────────────────────────────────────────────────
    result["atr"] = calculate_atr(high, low, close)

    # ── Support / Resistance ────────────────────────────────────────────────
    result["support_resistance"] = calculate_support_resistance(ohlcv)

    # ── Volume anomaly ──────────────────────────────────────────────────────
    result["volume_anomaly"] = detect_volume_anomaly(volume)

    # ── EMA crossover ───────────────────────────────────────────────────────
    result["ema_crossover"] = detect_ema_crossover(close)

    # ── RSI divergence ──────────────────────────────────────────────────────
    rsi_series: pd.Series | None = None
    try:
        rsi_series = ta.rsi(close, length=14)
    except Exception:
        logger.warning("[%s] Could not compute RSI series for divergence check.", symbol)

    if rsi_series is not None and not rsi_series.dropna().empty:
        result["rsi_divergence"] = detect_rsi_divergence(close, rsi_series)
    else:
        result["rsi_divergence"] = "none"

    # ── 52-week proximity ───────────────────────────────────────────────────
    if current_price is not None and high_52w is not None and low_52w is not None:
        result["proximity_52w"] = check_52w_proximity(current_price, high_52w, low_52w)
    else:
        result["proximity_52w"] = {
            "near_high": False, "near_low": False,
            "pct_from_high": None, "pct_from_low": None,
        }

    # ── Stochastic RSI (bonus) ───────────────────────────────────────────────
    try:
        stoch_rsi = ta.stochrsi(close)
        if stoch_rsi is not None and not stoch_rsi.dropna().empty:
            k_col = [c for c in stoch_rsi.columns if "STOCHRSIk" in c]
            d_col = [c for c in stoch_rsi.columns if "STOCHRSId" in c]
            result["stoch_rsi"] = {
                "k": _safe_float(stoch_rsi[k_col[0]].iloc[-1]) if k_col else None,
                "d": _safe_float(stoch_rsi[d_col[0]].iloc[-1]) if d_col else None,
            }
        else:
            result["stoch_rsi"] = {"k": None, "d": None}
    except Exception:
        result["stoch_rsi"] = {"k": None, "d": None}

    # ── ADX (trend strength) ────────────────────────────────────────────────
    try:
        adx_df = ta.adx(high, low, close)
        if adx_df is not None and not adx_df.dropna().empty:
            adx_col = [c for c in adx_df.columns if c.startswith("ADX_")]
            result["adx"] = _safe_float(adx_df[adx_col[0]].iloc[-1]) if adx_col else None
        else:
            result["adx"] = None
    except Exception:
        result["adx"] = None

    logger.debug("[%s] All indicators calculated successfully.", symbol)
    return result


# ---------------------------------------------------------------------------
# Batch enrichment
# ---------------------------------------------------------------------------

def enrich_all_assets(market_data: dict) -> dict:
    """
    Add technical indicators to every asset in a market-data mapping.

    Parameters
    ----------
    market_data : dict keyed by asset symbol.  Each value must be a dict
                  compatible with ``calculate_all_indicators`` (i.e. must
                  contain at minimum an 'ohlcv' key with a valid DataFrame).

    Returns
    -------
    A new dict keyed by symbol; values are the original asset dicts **merged**
    with the computed indicator dict (indicators take precedence on conflicts).

    Example
    -------
    >>> market_data = {
    ...     "AAPL": {"ohlcv": aapl_df, "symbol": "AAPL"},
    ...     "MSFT": {"ohlcv": msft_df, "symbol": "MSFT"},
    ... }
    >>> enriched = enrich_all_assets(market_data)
    >>> enriched["AAPL"]["rsi"]
    63.2
    """
    if not market_data:
        logger.warning("enrich_all_assets received an empty market_data dict.")
        return {}

    enriched: dict = {}
    total   = len(market_data)
    success = 0
    failed  = 0

    for symbol, asset_data in market_data.items():
        # Ensure the symbol field is set so calculate_all_indicators can use it
        if isinstance(asset_data, dict) and "symbol" not in asset_data:
            asset_data = {**asset_data, "symbol": symbol}

        try:
            indicators = calculate_all_indicators(asset_data)
            # Merge: start with original data, overlay with indicators
            merged = {**(asset_data if isinstance(asset_data, dict) else {}), **indicators}
            enriched[symbol] = merged
            if indicators.get("error") is None:
                success += 1
            else:
                failed += 1
                logger.warning(
                    "Indicator calculation failed for %s: %s",
                    symbol, indicators["error"],
                )
        except Exception:
            failed += 1
            logger.exception("Unexpected error enriching asset '%s'.", symbol)
            enriched[symbol] = {
                **(asset_data if isinstance(asset_data, dict) else {}),
                "symbol": symbol,
                "error":  "Unexpected exception during enrichment.",
            }

    logger.info(
        "enrich_all_assets complete — %d/%d succeeded, %d failed.",
        success, total, failed,
    )
    return enriched


# ---------------------------------------------------------------------------
# Self-test / smoke-test (run with:  python indicators.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import random
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    # --- Build a synthetic 300-bar OHLCV DataFrame ---
    rng = np.random.default_rng(42)
    n   = 300
    close_prices = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    high_prices  = close_prices + rng.uniform(0.2, 1.5, n)
    low_prices   = close_prices - rng.uniform(0.2, 1.5, n)
    open_prices  = close_prices + rng.standard_normal(n) * 0.3
    volumes      = rng.integers(500_000, 5_000_000, n).astype(float)
    # Inject a volume spike on the last bar
    volumes[-1]  = volumes[-21:-1].mean() * 3.5

    fake_ohlcv = pd.DataFrame({
        "open":   open_prices,
        "high":   high_prices,
        "low":    low_prices,
        "close":  close_prices,
        "volume": volumes,
    })

    asset = {"symbol": "SYNTHETIC", "ohlcv": fake_ohlcv}
    result = calculate_all_indicators(asset)

    print("\n=== Indicator Results ===")
    for key, val in result.items():
        print(f"  {key:25s}: {val}")

    # --- Batch test ---
    batch = {
        "SYN_A": {"ohlcv": fake_ohlcv},
        "SYN_B": {"ohlcv": fake_ohlcv.copy()},
        "BAD":   {"ohlcv": pd.DataFrame()},       # should produce an error, not crash
    }
    enriched = enrich_all_assets(batch)
    print("\n=== Enriched Symbols ===")
    for sym, data in enriched.items():
        print(f"  {sym}: error={data.get('error')}, rsi={data.get('rsi')}")
