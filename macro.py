"""
macro.py — Macro Regime Detection & Intermarket Analysis
=========================================================
Part of the fully autonomous trading research agent.

Determines market regime (RISK-ON / RISK-OFF / TRANSITIONING) and provides
intermarket analysis across equities, bonds, currencies, commodities, and crypto.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] macro.py — %(message)s")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGIME_ASSETS: list[str] = [
    "SPY",       # S&P 500 ETF
    "^VIX",      # Volatility index
    "HYG",       # High-yield bond ETF
    "TLT",       # 20+ Year Treasury ETF
    "IWM",       # Russell 2000 (small caps)
    "GLD",       # Gold ETF
    "GC=F",      # Gold futures
    "CL=F",      # Crude oil futures
    "DX-Y.NYB",  # US Dollar index
]

TREASURY_TICKERS: list[str] = [
    "^TNX",   # 10-Year Treasury yield
    "^IRX",   # 13-Week Treasury Bill (proxy for 2Y direction)
    "^FVX",   # 5-Year Treasury yield
    "^TYX",   # 30-Year Treasury yield
]

COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
COINGECKO_TIMEOUT = 10  # seconds
YFINANCE_LOOKBACK_DAYS = 90  # extra buffer for EMA calculation

# Regime thresholds
VIX_RISK_ON_THRESHOLD = 18.0
VIX_RISK_OFF_THRESHOLD = 25.0
VIX_SPIKE_THRESHOLD_PCT = 20.0        # intra-session spike flag
DXY_SHARP_GAIN_THRESHOLD_PCT = 0.5   # > +0.5% 5d = "sharply strengthening"
RISK_ON_SIGNAL_REQUIRED = 4
RISK_OFF_SIGNAL_REQUIRED = 4
SIGNAL_CONFLICT_THRESHOLD = 3         # < 3 opposite signals for clear regime

# EMA period for SPY regime filter
SPY_EMA_PERIOD = 50


# ===========================================================================
# Helper utilities
# ===========================================================================

def _safe_last(series: pd.Series) -> Optional[float]:
    """Return the last valid float from a pandas Series, or None."""
    if series is None or series.empty:
        return None
    val = series.dropna()
    if val.empty:
        return None
    return float(val.iloc[-1])


def _pct_change_nd(series: pd.Series, n: int) -> Optional[float]:
    """
    Return the n-day percentage change (not annualised) of the last value
    vs the value n trading days ago.  Returns None if insufficient data.
    """
    clean = series.dropna()
    if len(clean) < n + 1:
        return None
    old = float(clean.iloc[-(n + 1)])
    new = float(clean.iloc[-1])
    if old == 0:
        return None
    return (new - old) / abs(old) * 100.0


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _fetch_yfinance(
    tickers: list[str],
    period: str = "3mo",
    interval: str = "1d",
    retries: int = 3,
    delay: float = 1.5,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for a list of tickers via yfinance.
    Returns a dict mapping ticker → DataFrame (with at least a 'Close' column).
    Missing tickers get an empty DataFrame.
    """
    result: dict[str, pd.DataFrame] = {}

    # Download all at once — faster and avoids rate limits
    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            break
        except Exception as exc:
            logger.warning("yfinance download attempt %d failed: %s", attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                logger.error("All yfinance download attempts failed for %s", tickers)
                return {t: pd.DataFrame() for t in tickers}

    # yfinance returns a MultiIndex DataFrame when multiple tickers are requested
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                df = raw.xs(ticker, level=1, axis=1).copy()
                df.dropna(how="all", inplace=True)
                result[ticker] = df
            except KeyError:
                logger.warning("Ticker %s not found in downloaded data.", ticker)
                result[ticker] = pd.DataFrame()
    else:
        # Single ticker — yfinance returns flat columns
        ticker = tickers[0]
        raw.dropna(how="all", inplace=True)
        result[ticker] = raw

    return result


# ===========================================================================
# Core data-fetch functions
# ===========================================================================

def fetch_regime_assets() -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for all regime-relevant assets via yfinance.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are ticker symbols; values are DataFrames with a 'Close' column
        (and OHLCV columns where available).  Empty DataFrames signal fetch
        failure for that ticker.
    """
    logger.info("Fetching regime assets: %s", REGIME_ASSETS)
    data = _fetch_yfinance(REGIME_ASSETS, period="3mo", interval="1d")

    # Validate each ticker
    for ticker in REGIME_ASSETS:
        df = data.get(ticker, pd.DataFrame())
        if df.empty or "Close" not in df.columns:
            logger.warning("No usable Close data for %s", ticker)
            data[ticker] = pd.DataFrame()
        else:
            logger.debug(
                "%s: %d rows, latest close=%.4f",
                ticker,
                len(df),
                _safe_last(df["Close"]) or float("nan"),
            )

    return data


def fetch_treasury_data() -> dict[str, pd.DataFrame]:
    """
    Fetch US Treasury yield data via yfinance.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ^TNX (10Y), ^IRX (13W / 2Y proxy), ^FVX (5Y), ^TYX (30Y).
        Values: DataFrames with 'Close' column representing yield in percent.
    """
    logger.info("Fetching treasury yield data: %s", TREASURY_TICKERS)
    return _fetch_yfinance(TREASURY_TICKERS, period="3mo", interval="1d")


# ===========================================================================
# VIX status
# ===========================================================================

def get_vix_status(vix_data: pd.DataFrame) -> dict:
    """
    Analyse VIX levels and detect intra-session spikes.

    Parameters
    ----------
    vix_data : pd.DataFrame
        DataFrame for ^VIX with at least a 'Close' column.

    Returns
    -------
    dict with keys:
        current     (float)  — latest VIX close
        change_pct  (float)  — 1-day % change
        spiked      (bool)   — True if today's move > VIX_SPIKE_THRESHOLD_PCT
        status      (str)    — human-readable summary
    """
    default = {
        "current": float("nan"),
        "change_pct": float("nan"),
        "spiked": False,
        "status": "VIX data unavailable",
    }

    if vix_data is None or vix_data.empty or "Close" not in vix_data.columns:
        logger.warning("VIX data missing or malformed.")
        return default

    close = vix_data["Close"].dropna()
    if len(close) < 2:
        logger.warning("Insufficient VIX data points.")
        return default

    current = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    change_pct = (current - prev) / prev * 100.0 if prev != 0 else 0.0
    spiked = abs(change_pct) >= VIX_SPIKE_THRESHOLD_PCT

    if current < VIX_RISK_ON_THRESHOLD:
        status = f"VIX at {current:.1f} — low fear; complacency / risk-on environment"
    elif current < VIX_RISK_OFF_THRESHOLD:
        status = f"VIX at {current:.1f} — moderate uncertainty; market on edge"
    else:
        status = f"VIX at {current:.1f} — elevated fear; risk-off conditions prevailing"

    if spiked:
        status += f" (SESSION SPIKE: {change_pct:+.1f}%)"

    logger.debug("VIX status: %s", status)
    return {
        "current": current,
        "change_pct": round(change_pct, 2),
        "spiked": spiked,
        "status": status,
    }


# ===========================================================================
# Dollar cycle
# ===========================================================================

def analyze_dollar_cycle(dxy_data: pd.DataFrame) -> dict:
    """
    Analyse the US Dollar Index trend and its market impact.

    Parameters
    ----------
    dxy_data : pd.DataFrame
        DataFrame for DX-Y.NYB with at least a 'Close' column.

    Returns
    -------
    dict with keys:
        level          (float) — latest DXY close
        trend          (str)   — 'rising' | 'falling' | 'flat'
        change_5d_pct  (float) — 5-day percentage change
        impact         (str)   — 1-sentence market implication
    """
    default = {
        "level": float("nan"),
        "trend": "unknown",
        "change_5d_pct": float("nan"),
        "impact": "USD data unavailable; dollar cycle analysis skipped.",
    }

    if dxy_data is None or dxy_data.empty or "Close" not in dxy_data.columns:
        logger.warning("DXY data missing.")
        return default

    close = dxy_data["Close"].dropna()
    level = _safe_last(close)
    if level is None:
        return default

    change_5d = _pct_change_nd(close, 5)
    if change_5d is None:
        change_5d = 0.0

    if change_5d > 0.3:
        trend = "rising"
    elif change_5d < -0.3:
        trend = "falling"
    else:
        trend = "flat"

    if trend == "rising" and change_5d > DXY_SHARP_GAIN_THRESHOLD_PCT:
        impact = (
            "Strong USD pressures gold, crypto, and emerging markets "
            "while supporting US bonds."
        )
    elif trend == "rising":
        impact = (
            "Modestly rising USD creates mild headwinds for commodities "
            "and risk assets priced in dollars."
        )
    elif trend == "falling":
        impact = (
            "Weakening USD is broadly supportive of commodities, gold, "
            "and international equities; may lift inflation expectations."
        )
    else:
        impact = (
            "USD trading sideways; limited directional macro pressure "
            "from the dollar cycle at this time."
        )

    logger.debug("DXY level=%.2f, 5d chg=%.2f%%, trend=%s", level, change_5d, trend)
    return {
        "level": round(level, 3),
        "trend": trend,
        "change_5d_pct": round(change_5d, 3),
        "impact": impact,
    }


# ===========================================================================
# Rate environment
# ===========================================================================

def analyze_rate_environment(
    treasury_data: dict[str, pd.DataFrame],
    regime_assets: dict[str, pd.DataFrame],
) -> dict:
    """
    Analyse the interest rate environment including yield curve shape,
    spread dynamics, and the directional trend.

    Parameters
    ----------
    treasury_data : dict[str, pd.DataFrame]
        Output of fetch_treasury_data().
    regime_assets : dict[str, pd.DataFrame]
        Output of fetch_regime_assets() — used for TLT trend as a proxy.

    Returns
    -------
    dict with keys:
        fed_funds_rate         (float)  — placeholder / ^IRX proxy
        yield_10y              (float)
        yield_2y               (float)  — from ^IRX (13-week bill, directional proxy)
        yield_30y              (float)
        yield_curve            (str)    — 'normal' | 'flat' | 'inverted'
        yield_curve_spread_bps (float)  — (10Y − 2Y) × 100 basis points
        yield_curve_inverted   (bool)
        rate_trend             (str)    — 'rising' | 'falling' | 'flat'
        impact                 (str)    — market implication
    """
    default = {
        "fed_funds_rate": float("nan"),
        "yield_10y": float("nan"),
        "yield_2y": float("nan"),
        "yield_30y": float("nan"),
        "yield_curve": "unknown",
        "yield_curve_spread_bps": float("nan"),
        "yield_curve_inverted": False,
        "rate_trend": "unknown",
        "impact": "Treasury data unavailable; rate environment analysis skipped.",
    }

    def _get_yield(ticker: str) -> tuple[Optional[float], Optional[pd.Series]]:
        df = treasury_data.get(ticker, pd.DataFrame())
        if df.empty or "Close" not in df.columns:
            return None, None
        s = df["Close"].dropna()
        return _safe_last(s), s

    yield_10y, series_10y = _get_yield("^TNX")
    yield_2y_proxy, series_2y = _get_yield("^IRX")   # 13W is directionally similar to 2Y
    yield_30y, _ = _get_yield("^TYX")
    yield_5y, _ = _get_yield("^FVX")

    # Fallbacks
    if yield_10y is None:
        logger.warning("10Y yield (^TNX) unavailable.")
        return default

    # ^IRX is annualised discount yield × 100; treat as our "short rate"
    y2 = yield_2y_proxy if yield_2y_proxy is not None else float("nan")
    y10 = yield_10y
    y30 = yield_30y if yield_30y is not None else float("nan")

    spread_pct = y10 - y2  # in percent (e.g. 0.50 = 50bps)
    spread_bps = spread_pct * 100.0 if not (np.isnan(y2)) else float("nan")

    if np.isnan(spread_bps):
        yield_curve = "unknown"
        inverted = False
    elif spread_bps < -5:
        yield_curve = "inverted"
        inverted = True
    elif spread_bps < 25:
        yield_curve = "flat"
        inverted = False
    else:
        yield_curve = "normal"
        inverted = False

    # Rate trend: compare 10Y today vs 20 sessions ago
    rate_trend = "flat"
    if series_10y is not None and len(series_10y) >= 21:
        old_10y = float(series_10y.iloc[-21])
        delta = y10 - old_10y
        if delta > 0.10:
            rate_trend = "rising"
        elif delta < -0.10:
            rate_trend = "falling"

    # Fall back to TLT direction as a proxy (TLT moves inverse to rates)
    if rate_trend == "flat":
        tlt_df = regime_assets.get("TLT", pd.DataFrame())
        if not tlt_df.empty and "Close" in tlt_df.columns:
            tlt_chg = _pct_change_nd(tlt_df["Close"], 20)
            if tlt_chg is not None:
                if tlt_chg < -1.5:
                    rate_trend = "rising"
                elif tlt_chg > 1.5:
                    rate_trend = "falling"

    if rate_trend == "rising":
        impact = (
            "Rising rates are a headwind for growth stocks and bonds; "
            "supportive of banks and the USD."
        )
    elif rate_trend == "falling":
        impact = (
            "Falling rates benefit long-duration assets, growth equities, "
            "and gold; USD may weaken."
        )
    else:
        impact = (
            "Rate environment is stable; bond and equity valuations are "
            "not facing incremental rate pressure."
        )

    if inverted:
        impact += (
            " Inverted yield curve signals potential recession risk "
            "and pressures bank net-interest margins."
        )

    # ^IRX is in annualised percent (e.g. 5.25 means 5.25%)
    fed_funds_proxy = y2  # closest available approximation

    logger.debug(
        "Rate env — 10Y=%.2f, 2Y_proxy=%.2f, spread=%.1fbps, curve=%s, trend=%s",
        y10,
        y2 if not np.isnan(y2) else -999,
        spread_bps if not np.isnan(spread_bps) else -999,
        yield_curve,
        rate_trend,
    )

    return {
        "fed_funds_rate": round(fed_funds_proxy, 3) if not np.isnan(fed_funds_proxy) else float("nan"),
        "yield_10y": round(y10, 3),
        "yield_2y": round(y2, 3) if not np.isnan(y2) else float("nan"),
        "yield_30y": round(y30, 3) if not np.isnan(y30) else float("nan"),
        "yield_curve": yield_curve,
        "yield_curve_spread_bps": round(spread_bps, 2) if not np.isnan(spread_bps) else float("nan"),
        "yield_curve_inverted": inverted,
        "rate_trend": rate_trend,
        "impact": impact,
    }


# ===========================================================================
# Inflation analysis
# ===========================================================================

def analyze_inflation(fred_data: dict) -> dict:
    """
    Analyse inflation conditions from FRED data (CPI, PCE).

    Parameters
    ----------
    fred_data : dict
        Expected keys (optional): 'cpi' and 'pce', each a float or a
        pandas Series.  Values represent the latest year-over-year reading
        in percent (e.g. 3.4 for 3.4% CPI YoY).

    Returns
    -------
    dict with keys:
        cpi    (float) — latest CPI YoY %
        pce    (float) — latest PCE YoY %
        trend  (str)   — 'accelerating' | 'decelerating' | 'stable'
        impact (str)   — market implication
    """
    default = {
        "cpi": float("nan"),
        "pce": float("nan"),
        "trend": "unknown",
        "impact": "Inflation data unavailable; analysis skipped.",
    }

    if not fred_data:
        return default

    def _extract_scalar(key: str) -> Optional[float]:
        val = fred_data.get(key)
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, pd.Series):
            return _safe_last(val)
        if isinstance(val, pd.DataFrame) and not val.empty:
            return float(val.iloc[-1, 0])
        return None

    cpi = _extract_scalar("cpi")
    pce = _extract_scalar("pce")

    # Trend from series history
    trend = "stable"
    cpi_series = fred_data.get("cpi")
    if isinstance(cpi_series, pd.Series) and len(cpi_series) >= 3:
        recent = cpi_series.dropna()
        if len(recent) >= 3:
            slope = float(recent.iloc[-1]) - float(recent.iloc[-3])
            if slope > 0.2:
                trend = "accelerating"
            elif slope < -0.2:
                trend = "decelerating"

    # If no series trend available, use absolute level
    if trend == "stable" and cpi is not None:
        if cpi > 4.0:
            trend = "elevated"
        elif cpi < 2.0:
            trend = "below-target"

    cpi_val = cpi if cpi is not None else float("nan")
    pce_val = pce if pce is not None else float("nan")

    if trend == "accelerating":
        impact = (
            "Accelerating inflation pressures the Fed toward tighter policy, "
            "weighing on bond prices and rate-sensitive equities."
        )
    elif trend == "decelerating":
        impact = (
            "Decelerating inflation raises hopes for Fed rate cuts, "
            "supporting bonds, growth stocks, and risk assets."
        )
    elif trend == "elevated":
        impact = (
            "Persistently elevated inflation constrains monetary easing, "
            "keeping real rates high and pressuring valuations."
        )
    elif trend == "below-target":
        impact = (
            "Below-target inflation gives the Fed flexibility to ease, "
            "a tailwind for duration and growth assets."
        )
    else:
        impact = (
            "Inflation near target; monetary policy backdrop is relatively "
            "neutral for equities and bonds."
        )

    logger.debug("Inflation — CPI=%.2f, PCE=%.2f, trend=%s", cpi_val, pce_val, trend)
    return {
        "cpi": round(cpi_val, 2) if not np.isnan(cpi_val) else float("nan"),
        "pce": round(pce_val, 2) if not np.isnan(pce_val) else float("nan"),
        "trend": trend,
        "impact": impact,
    }


# ===========================================================================
# BTC dominance
# ===========================================================================

def assess_btc_dominance() -> dict:
    """
    Fetch Bitcoin dominance from the CoinGecko /global endpoint.

    Returns
    -------
    dict with keys:
        btc_dominance (float) — BTC % of total crypto market cap
        trend         (str)   — 'rising' | 'falling' | 'flat' | 'unknown'
        impact        (str)   — market implication
    """
    default = {
        "btc_dominance": float("nan"),
        "trend": "unknown",
        "impact": "BTC dominance data unavailable; CoinGecko may be rate-limiting.",
    }

    try:
        headers = {"Accept": "application/json", "User-Agent": "trading-bot/1.0"}
        response = requests.get(
            COINGECKO_GLOBAL_URL, headers=headers, timeout=COINGECKO_TIMEOUT
        )
        response.raise_for_status()
        payload = response.json()

        btc_dom = payload.get("data", {}).get("market_cap_percentage", {}).get("btc")
        if btc_dom is None:
            logger.warning("BTC dominance field missing from CoinGecko response.")
            return default

        btc_dom = float(btc_dom)

        # Trend heuristic based on absolute level
        # > 50%: BTC dominant (risk-off within crypto, alt season fading)
        # < 40%: Alt season / broader crypto risk-on
        if btc_dom > 55:
            trend = "rising"
            impact = (
                "High BTC dominance signals risk-off within crypto; capital "
                "rotating from altcoins into Bitcoin as a relative safe haven."
            )
        elif btc_dom > 48:
            trend = "flat"
            impact = (
                "BTC dominance near 50% — neutral crypto regime; no strong "
                "alt-season or BTC dominance trend in effect."
            )
        else:
            trend = "falling"
            impact = (
                "Low BTC dominance signals alt-season conditions; broader "
                "crypto risk-on with capital flowing into higher-beta altcoins."
            )

        logger.debug("BTC dominance=%.2f%%, trend=%s", btc_dom, trend)
        return {
            "btc_dominance": round(btc_dom, 2),
            "trend": trend,
            "impact": impact,
        }

    except requests.exceptions.Timeout:
        logger.error("CoinGecko request timed out after %ds.", COINGECKO_TIMEOUT)
        return default
    except requests.exceptions.HTTPError as exc:
        logger.error("CoinGecko HTTP error: %s", exc)
        return default
    except requests.exceptions.RequestException as exc:
        logger.error("CoinGecko request failed: %s", exc)
        return default
    except (KeyError, ValueError, TypeError) as exc:
        logger.error("CoinGecko response parsing error: %s", exc)
        return default


# ===========================================================================
# Regime score calculation
# ===========================================================================

def calculate_regime_score(
    regime_assets: dict[str, pd.DataFrame],
    treasury_data: dict[str, pd.DataFrame],
    last_state: Optional[dict] = None,
) -> dict:
    """
    Calculate the macro market regime based on Risk-On / Risk-Off signals.

    Risk-On signals (+1 each):
        1. SPY above its 50 EMA
        2. VIX below 18
        3. HYG rising (positive 5d return)
        4. Small caps (IWM) outperforming large caps (SPY) over last 10 days
        5. Commodities ex-gold rising (CL=F oil positive 5d)
        6. DXY weakening (negative 5d return)

    Risk-Off signals (+1 each):
        1. SPY below its 50 EMA
        2. VIX above 25
        3. TLT rising (flight to safety)
        4. Gold rising while SPY falling
        5. DXY strengthening sharply (>0.5% 5d gain)
        6. Yield curve inverting further (10Y-2Y spread widening negatively)

    Regime:
        RISK-ON       if risk_on_score >= 4 AND risk_off_score < 3
        RISK-OFF      if risk_off_score >= 4 AND risk_on_score < 3
        TRANSITIONING otherwise

    Parameters
    ----------
    regime_assets : dict[str, pd.DataFrame]
        Output of fetch_regime_assets().
    treasury_data : dict[str, pd.DataFrame]
        Output of fetch_treasury_data().
    last_state : dict, optional
        Previous macro state; used to detect regime changes.

    Returns
    -------
    dict with keys:
        regime          (str)       — 'RISK-ON' | 'RISK-OFF' | 'TRANSITIONING'
        risk_on_score   (int)
        risk_off_score  (int)
        risk_on_signals (list[str]) — active signal descriptions
        risk_off_signals(list[str]) — active signal descriptions
        regime_changed  (bool)      — True if different from last_state
    """
    risk_on_signals: list[str] = []
    risk_off_signals: list[str] = []

    # ------------------------------------------------------------------ helpers
    def close(ticker: str) -> Optional[pd.Series]:
        df = regime_assets.get(ticker, pd.DataFrame())
        if df.empty or "Close" not in df.columns:
            return None
        return df["Close"].dropna()

    def treasury_close(ticker: str) -> Optional[pd.Series]:
        df = treasury_data.get(ticker, pd.DataFrame())
        if df.empty or "Close" not in df.columns:
            return None
        return df["Close"].dropna()

    # ------------------------------------------------------------------ SPY vs 50 EMA
    spy_close = close("SPY")
    if spy_close is not None and len(spy_close) >= SPY_EMA_PERIOD:
        ema50 = _ema(spy_close, SPY_EMA_PERIOD)
        last_spy = float(spy_close.iloc[-1])
        last_ema = float(ema50.iloc[-1])
        if last_spy > last_ema:
            risk_on_signals.append(
                f"SPY ({last_spy:.2f}) above 50 EMA ({last_ema:.2f}) — bullish trend"
            )
        else:
            risk_off_signals.append(
                f"SPY ({last_spy:.2f}) below 50 EMA ({last_ema:.2f}) — bearish trend"
            )
    else:
        logger.debug("SPY EMA signal skipped — insufficient data.")

    # ------------------------------------------------------------------ VIX level
    vix_close = close("^VIX")
    if vix_close is not None:
        vix_current = float(vix_close.iloc[-1])
        if vix_current < VIX_RISK_ON_THRESHOLD:
            risk_on_signals.append(
                f"VIX at {vix_current:.1f} — below {VIX_RISK_ON_THRESHOLD} (low fear)"
            )
        if vix_current > VIX_RISK_OFF_THRESHOLD:
            risk_off_signals.append(
                f"VIX at {vix_current:.1f} — above {VIX_RISK_OFF_THRESHOLD} (high fear)"
            )
    else:
        logger.debug("VIX signal skipped — data unavailable.")

    # ------------------------------------------------------------------ HYG 5d return
    hyg_close = close("HYG")
    if hyg_close is not None:
        hyg_5d = _pct_change_nd(hyg_close, 5)
        if hyg_5d is not None:
            if hyg_5d > 0:
                risk_on_signals.append(
                    f"HYG rising +{hyg_5d:.2f}% over 5 days — credit risk appetite healthy"
                )
            else:
                logger.debug("HYG 5d chg=%.2f%% — not a risk-on signal.", hyg_5d)
    else:
        logger.debug("HYG signal skipped — data unavailable.")

    # ------------------------------------------------------------------ TLT rising (risk-off)
    tlt_close = close("TLT")
    if tlt_close is not None:
        tlt_5d = _pct_change_nd(tlt_close, 5)
        if tlt_5d is not None and tlt_5d > 0:
            risk_off_signals.append(
                f"TLT rising +{tlt_5d:.2f}% over 5 days — flight to safety in bonds"
            )
    else:
        logger.debug("TLT signal skipped — data unavailable.")

    # ------------------------------------------------------------------ Small caps vs large caps (IWM vs SPY, 10d)
    iwm_close = close("IWM")
    if iwm_close is not None and spy_close is not None:
        iwm_10d = _pct_change_nd(iwm_close, 10)
        spy_10d = _pct_change_nd(spy_close, 10)
        if iwm_10d is not None and spy_10d is not None:
            if iwm_10d > spy_10d:
                risk_on_signals.append(
                    f"IWM outperforming SPY over 10d "
                    f"({iwm_10d:+.2f}% vs {spy_10d:+.2f}%) — small cap leadership (risk-on)"
                )
            else:
                logger.debug(
                    "IWM lagging SPY 10d (IWM %+.2f vs SPY %+.2f) — no risk-on signal.",
                    iwm_10d,
                    spy_10d,
                )
    else:
        logger.debug("IWM/SPY relative signal skipped — data unavailable.")

    # ------------------------------------------------------------------ Gold vs SPY (risk-off)
    gld_close = close("GLD")
    gc_close = close("GC=F")
    gold_series = gld_close if gld_close is not None else gc_close

    if gold_series is not None and spy_close is not None:
        gold_5d = _pct_change_nd(gold_series, 5)
        spy_5d_for_gold = _pct_change_nd(spy_close, 5)
        if gold_5d is not None and spy_5d_for_gold is not None:
            if gold_5d > 0 and spy_5d_for_gold < 0:
                risk_off_signals.append(
                    f"Gold rising +{gold_5d:.2f}% while SPY falling {spy_5d_for_gold:.2f}% "
                    f"— classic risk-off rotation"
                )
    else:
        logger.debug("Gold vs SPY signal skipped — data unavailable.")

    # ------------------------------------------------------------------ Oil / commodities (CL=F, 5d)
    oil_close = close("CL=F")
    if oil_close is not None:
        oil_5d = _pct_change_nd(oil_close, 5)
        if oil_5d is not None:
            if oil_5d > 0:
                risk_on_signals.append(
                    f"Crude oil (CL=F) +{oil_5d:.2f}% over 5 days — commodity demand rising (risk-on)"
                )
            else:
                logger.debug("Oil 5d chg=%.2f%% — not a risk-on signal.", oil_5d)
    else:
        logger.debug("Oil signal skipped — data unavailable.")

    # ------------------------------------------------------------------ DXY 5d return
    dxy_close = close("DX-Y.NYB")
    if dxy_close is not None:
        dxy_5d = _pct_change_nd(dxy_close, 5)
        if dxy_5d is not None:
            if dxy_5d < 0:
                risk_on_signals.append(
                    f"DXY weakening {dxy_5d:.2f}% over 5 days — USD weakness supports risk assets"
                )
            if dxy_5d > DXY_SHARP_GAIN_THRESHOLD_PCT:
                risk_off_signals.append(
                    f"DXY sharply rising +{dxy_5d:.2f}% over 5 days (>{DXY_SHARP_GAIN_THRESHOLD_PCT}%) "
                    f"— USD strength pressures risk assets"
                )
    else:
        logger.debug("DXY signal skipped — data unavailable.")

    # ------------------------------------------------------------------ Yield curve (10Y - 2Y)
    t10y = treasury_close("^TNX")
    t2y = treasury_close("^IRX")
    if t10y is not None and t2y is not None:
        # Check if the spread is worsening (becoming more negative)
        current_spread = float(t10y.iloc[-1]) - float(t2y.iloc[-1])
        if len(t10y) >= 6 and len(t2y) >= 6:
            prev_spread = float(t10y.iloc[-6]) - float(t2y.iloc[-6])
            spread_change = current_spread - prev_spread  # positive = steepening, negative = inverting further
            if current_spread < 0:
                if spread_change < 0:
                    risk_off_signals.append(
                        f"Yield curve inverting further — 10Y-2Y spread "
                        f"{current_spread * 100:.1f}bps (down {abs(spread_change) * 100:.1f}bps in 5d); "
                        f"recession risk signal"
                    )
                else:
                    risk_off_signals.append(
                        f"Yield curve inverted — 10Y-2Y spread {current_spread * 100:.1f}bps; "
                        f"recession risk flagged"
                    )
    else:
        logger.debug("Yield curve signal skipped — treasury data unavailable.")

    # ------------------------------------------------------------------ Regime determination
    risk_on_score = len(risk_on_signals)
    risk_off_score = len(risk_off_signals)

    if risk_on_score >= RISK_ON_SIGNAL_REQUIRED and risk_off_score < SIGNAL_CONFLICT_THRESHOLD:
        regime = "RISK-ON"
    elif risk_off_score >= RISK_OFF_SIGNAL_REQUIRED and risk_on_score < SIGNAL_CONFLICT_THRESHOLD:
        regime = "RISK-OFF"
    else:
        regime = "TRANSITIONING"

    # ------------------------------------------------------------------ Regime changed?
    prev_regime_val = (last_state or {}).get("regime")
    if isinstance(prev_regime_val, dict):
        prev_regime = prev_regime_val.get("regime", "UNKNOWN")
    elif isinstance(prev_regime_val, str):
        prev_regime = prev_regime_val
    else:
        prev_regime = "UNKNOWN"
    regime_changed = prev_regime != regime and prev_regime != "UNKNOWN"

    if regime_changed:
        logger.info(
            "REGIME CHANGE detected: %s → %s (risk_on=%d, risk_off=%d)",
            prev_regime,
            regime,
            risk_on_score,
            risk_off_score,
        )
    else:
        logger.info(
            "Regime: %s (risk_on=%d, risk_off=%d)",
            regime,
            risk_on_score,
            risk_off_score,
        )

    return {
        "regime": regime,
        "risk_on_score": risk_on_score,
        "risk_off_score": risk_off_score,
        "risk_on_signals": risk_on_signals,
        "risk_off_signals": risk_off_signals,
        "regime_changed": regime_changed,
    }


# ===========================================================================
# Intermarket analysis
# ===========================================================================

def intermarket_analysis(
    regime_assets: dict[str, pd.DataFrame],
    treasury_data: dict[str, pd.DataFrame],
) -> dict:
    """
    Perform cross-asset intermarket analysis to identify rotation signals,
    credit-spread dynamics, and asset class rankings for the current regime.

    Parameters
    ----------
    regime_assets : dict[str, pd.DataFrame]
        Output of fetch_regime_assets().
    treasury_data : dict[str, pd.DataFrame]
        Output of fetch_treasury_data().

    Returns
    -------
    dict with keys:
        bonds_vs_stocks        (str)  — relationship / rotation narrative
        gold_vs_usd            (str)
        oil_vs_inflation       (str)
        credit_spreads         (str)  — HYG vs TLT relative performance
        regime_asset_rankings  (list[str]) — ordered list of favoured asset classes
    """
    def _close_series(ticker: str) -> Optional[pd.Series]:
        df = regime_assets.get(ticker, pd.DataFrame())
        if df.empty or "Close" not in df.columns:
            return None
        return df["Close"].dropna()

    # ------------------------------------------------------------------ Bonds vs Stocks
    tlt_s = _close_series("TLT")
    spy_s = _close_series("SPY")
    bonds_vs_stocks = "Insufficient data for bonds vs stocks analysis."
    if tlt_s is not None and spy_s is not None:
        tlt_5d = _pct_change_nd(tlt_s, 5)
        spy_5d = _pct_change_nd(spy_s, 5)
        if tlt_5d is not None and spy_5d is not None:
            if tlt_5d > 0.5 and spy_5d < -0.5:
                bonds_vs_stocks = (
                    f"TLT rising {tlt_5d:+.2f}% while SPY falling {spy_5d:.2f}% — "
                    f"classic risk-off rotation; capital fleeing equities for bonds."
                )
            elif tlt_5d < -0.5 and spy_5d > 0.5:
                bonds_vs_stocks = (
                    f"SPY rising {spy_5d:+.2f}% while TLT falling {tlt_5d:.2f}% — "
                    f"risk-on rotation; equities leading, bonds selling off."
                )
            elif tlt_5d > 0 and spy_5d > 0:
                bonds_vs_stocks = (
                    f"Both SPY ({spy_5d:+.2f}%) and TLT ({tlt_5d:+.2f}%) rising — "
                    f"'buy everything' regime or conflicting signals; monitor closely."
                )
            elif tlt_5d < 0 and spy_5d < 0:
                bonds_vs_stocks = (
                    f"Both SPY ({spy_5d:+.2f}%) and TLT ({tlt_5d:+.2f}%) falling — "
                    f"liquidation / stagflation signals; cash and commodities may outperform."
                )
            else:
                bonds_vs_stocks = (
                    f"SPY 5d: {spy_5d:+.2f}%, TLT 5d: {tlt_5d:+.2f}% — "
                    f"no strong directional bond/equity divergence detected."
                )

    # ------------------------------------------------------------------ Gold vs USD
    dxy_s = _close_series("DX-Y.NYB")
    gld_s = _close_series("GLD")
    gc_s = _close_series("GC=F")
    gold_s = gld_s if gld_s is not None else gc_s
    gold_vs_usd = "Insufficient data for gold vs USD analysis."
    if gold_s is not None and dxy_s is not None:
        gold_5d = _pct_change_nd(gold_s, 5)
        dxy_5d = _pct_change_nd(dxy_s, 5)
        if gold_5d is not None and dxy_5d is not None:
            if gold_5d > 0 and dxy_5d < 0:
                gold_vs_usd = (
                    f"Gold ({gold_5d:+.2f}%) rising as DXY ({dxy_5d:+.2f}%) weakens — "
                    f"textbook inverse correlation; bullish for gold and crypto."
                )
            elif gold_5d > 0 and dxy_5d > 0:
                gold_vs_usd = (
                    f"Gold ({gold_5d:+.2f}%) rising despite strong DXY ({dxy_5d:+.2f}%) — "
                    f"flight-to-safety demand overwhelming USD relationship; very bullish gold."
                )
            elif gold_5d < 0 and dxy_5d > 0:
                gold_vs_usd = (
                    f"Gold ({gold_5d:+.2f}%) falling as DXY ({dxy_5d:+.2f}%) strengthens — "
                    f"USD strength pressuring precious metals; bearish for gold and EM assets."
                )
            else:
                gold_vs_usd = (
                    f"Gold {gold_5d:+.2f}%, DXY {dxy_5d:+.2f}% — "
                    f"weak divergence; no actionable gold/USD regime signal."
                )
    elif gold_s is not None:
        gold_5d = _pct_change_nd(gold_s, 5)
        if gold_5d is not None:
            gold_vs_usd = (
                f"Gold {'rising' if gold_5d > 0 else 'falling'} {gold_5d:+.2f}% over 5 days. "
                f"DXY data unavailable for correlation analysis."
            )

    # ------------------------------------------------------------------ Oil vs Inflation
    oil_s = _close_series("CL=F")
    oil_vs_inflation = "Insufficient data for oil vs inflation analysis."
    if oil_s is not None:
        oil_5d = _pct_change_nd(oil_s, 5)
        oil_20d = _pct_change_nd(oil_s, 20)
        if oil_5d is not None and oil_20d is not None:
            if oil_5d > 3 and oil_20d > 5:
                oil_vs_inflation = (
                    f"Crude oil up {oil_5d:+.2f}% (5d) and {oil_20d:+.2f}% (20d) — "
                    f"rising energy costs likely to push CPI higher; inflation risk increasing."
                )
            elif oil_5d < -3 and oil_20d < -5:
                oil_vs_inflation = (
                    f"Crude oil down {oil_5d:+.2f}% (5d) and {oil_20d:+.2f}% (20d) — "
                    f"energy deflation may soften headline CPI; disinflationary tailwind."
                )
            elif oil_5d > 0:
                oil_vs_inflation = (
                    f"Crude oil modestly rising {oil_5d:+.2f}% over 5 days — "
                    f"mild upward pressure on energy components of inflation."
                )
            else:
                oil_vs_inflation = (
                    f"Crude oil {oil_5d:+.2f}% over 5 days — "
                    f"limited near-term inflationary impulse from energy."
                )
        elif oil_5d is not None:
            oil_vs_inflation = (
                f"Crude oil {oil_5d:+.2f}% over 5 days — "
                f"{'upside' if oil_5d > 0 else 'downside'} inflation risk from energy."
            )

    # ------------------------------------------------------------------ Credit spreads (HYG vs TLT)
    hyg_s = _close_series("HYG")
    credit_spreads = "Insufficient data for credit spread analysis."
    if hyg_s is not None and tlt_s is not None:
        hyg_5d = _pct_change_nd(hyg_s, 5)
        tlt_5d_cs = _pct_change_nd(tlt_s, 5)
        if hyg_5d is not None and tlt_5d_cs is not None:
            spread_diff = hyg_5d - tlt_5d_cs  # positive = HYG outperforming = tightening spreads
            if spread_diff > 0.5:
                credit_spreads = (
                    f"HYG outperforming TLT by {spread_diff:.2f}pp over 5d — "
                    f"credit spreads tightening; risk appetite healthy; high yield demand strong."
                )
            elif spread_diff < -0.5:
                credit_spreads = (
                    f"HYG underperforming TLT by {abs(spread_diff):.2f}pp over 5d — "
                    f"credit spreads widening; risk aversion rising; watch for contagion."
                )
            else:
                credit_spreads = (
                    f"HYG and TLT moving roughly in tandem (spread diff: {spread_diff:+.2f}pp) — "
                    f"credit spreads stable; no strong stress signal in high yield."
                )

    # ------------------------------------------------------------------ Regime asset rankings
    # Score assets based on their 5d and 20d returns; rank by composite
    asset_map = {
        "US Equities (SPY)": spy_s,
        "High Yield Bonds (HYG)": hyg_s,
        "Long Treasuries (TLT)": tlt_s,
        "Small Caps (IWM)": _close_series("IWM"),
        "Gold (GLD/GC=F)": gold_s,
        "Crude Oil (CL=F)": oil_s,
        "US Dollar (DX-Y.NYB)": dxy_s,
    }

    scores: list[tuple[str, float]] = []
    for name, series in asset_map.items():
        if series is None:
            continue
        r5 = _pct_change_nd(series, 5) or 0.0
        r20 = _pct_change_nd(series, 20) or 0.0
        composite = 0.6 * r5 + 0.4 * r20  # weight recent more heavily
        scores.append((name, composite))

    scores.sort(key=lambda x: x[1], reverse=True)
    regime_asset_rankings = [
        f"{name} ({score:+.2f}% composite momentum)"
        for name, score in scores
    ]

    if not regime_asset_rankings:
        regime_asset_rankings = ["Asset ranking unavailable — insufficient price data."]

    logger.debug("Intermarket analysis complete.")
    return {
        "bonds_vs_stocks": bonds_vs_stocks,
        "gold_vs_usd": gold_vs_usd,
        "oil_vs_inflation": oil_vs_inflation,
        "credit_spreads": credit_spreads,
        "regime_asset_rankings": regime_asset_rankings,
    }


# ---------------------------------------------------------------------------
# Sector Rotation Tracker (11 Industry ETFs vs SPY over 1W and 1M)
# ---------------------------------------------------------------------------

def calculate_sector_rotation() -> dict:
    """
    Download 1W and 1M price history for 11 sector ETFs vs SPY.
    Computes performance of each ETF relative to SPY.
    Returns ranked list of leaders/laggards and rotation signals.
    """
    tickers = ["SPY", "XLK", "XLF", "XLY", "XLP", "XLV", "XLE", "XLI", "XLB", "XLRE", "XLU", "XLC"]
    etf_names = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLV": "Health Care",
        "XLE": "Energy",
        "XLI": "Industrials",
        "XLB": "Materials",
        "XLRE": "Real Estate",
        "XLU": "Utilities",
        "XLC": "Communication Services"
    }
    
    try:
        logger.info("Executing Sector Rotation analysis...")
        # Download 3 months of daily data
        df = yf.download(tickers, period="3mo", interval="1d", auto_adjust=True, progress=False)
        
        if df is None or df.empty:
            logger.warning("Failed to download sector ETF data: empty DataFrame.")
            return {}
            
        closes = None
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.levels[0]:
                closes = df["Close"]
        else:
            # Single ticker fallback or flat columns if only Close was fetched
            closes = df
            
        if closes is None or closes.empty:
            logger.warning("Could not extract Close prices for sector rotation.")
            return {}
            
        # Ensure SPY and sector columns exist in closes
        closes = closes.dropna(how="all")
        if len(closes) < 22:
            logger.warning("Insufficient trading days (%d < 22) for 1M relative performance.", len(closes))
            return {}
            
        perf_1w = {}
        perf_1m = {}
        
        # Calculate SPY raw performance
        spy_start_1w = float(closes["SPY"].iloc[-6]) if "SPY" in closes.columns else float("nan")
        spy_end = float(closes["SPY"].iloc[-1]) if "SPY" in closes.columns else float("nan")
        spy_start_1m = float(closes["SPY"].iloc[-22]) if "SPY" in closes.columns else float("nan")
        
        if np.isnan(spy_end) or spy_start_1w <= 0 or spy_start_1m <= 0:
            logger.warning("Invalid SPY price data for relative returns.")
            return {}
            
        spy_1w = (spy_end - spy_start_1w) / spy_start_1w * 100
        spy_1m = (spy_end - spy_start_1m) / spy_start_1m * 100
        
        for t in etf_names.keys():
            if t in closes.columns:
                etf_start_1w = float(closes[t].iloc[-6])
                etf_end = float(closes[t].iloc[-1])
                etf_start_1m = float(closes[t].iloc[-22])
                
                if etf_start_1w > 0 and etf_start_1m > 0 and not np.isnan(etf_end):
                    etf_1w_pct = (etf_end - etf_start_1w) / etf_start_1w * 100
                    etf_1m_pct = (etf_end - etf_start_1m) / etf_start_1m * 100
                    
                    # Relative return vs SPY
                    perf_1w[t] = etf_1w_pct - spy_1w
                    perf_1m[t] = etf_1m_pct - spy_1m
                    
        if not perf_1w or not perf_1m:
            logger.warning("No sectors successfully calculated.")
            return {}
            
        ranked_1w = sorted(perf_1w.items(), key=lambda x: x[1], reverse=True)
        ranked_1m = sorted(perf_1m.items(), key=lambda x: x[1], reverse=True)
        
        leaders_1w = ranked_1w[:3]
        laggards_1w = ranked_1w[-3:]
        
        leaders_1m = ranked_1m[:3]
        laggards_1m = ranked_1m[-3:]
        
        rotation_signals = []
        bottom_3_1m = [x[0] for x in ranked_1m[-3:]]
        top_3_1w = [x[0] for x in ranked_1w[:3]]
        
        for etf in top_3_1w:
            if etf in bottom_3_1m:
                rotation_signals.append({
                    "ticker": etf,
                    "name": etf_names[etf],
                    "type": "BULLISH_ROTATION",
                    "details": f"Sector **{etf_names[etf]} ({etf})** rotated from 1M Laggard to 1W Leader!"
                })
                
        top_3_1m = [x[0] for x in ranked_1m[:3]]
        bottom_3_1w = [x[0] for x in ranked_1w[-3:]]
        for etf in bottom_3_1w:
            if etf in top_3_1m:
                rotation_signals.append({
                    "ticker": etf,
                    "name": etf_names[etf],
                    "type": "BEARISH_ROTATION",
                    "details": f"Sector **{etf_names[etf]} ({etf})** rotated from 1M Leader to 1W Laggard!"
                })
                
        return {
            "spy_1w": spy_1w,
            "spy_1m": spy_1m,
            "ranked_1w": [{"ticker": k, "rel_perf": v, "name": etf_names[k]} for k, v in ranked_1w],
            "ranked_1m": [{"ticker": k, "rel_perf": v, "name": etf_names[k]} for k, v in ranked_1m],
            "leaders_1w": [{"ticker": k, "rel_perf": v, "name": etf_names[k]} for k, v in leaders_1w],
            "laggards_1w": [{"ticker": k, "rel_perf": v, "name": etf_names[k]} for k, v in laggards_1w],
            "leaders_1m": [{"ticker": k, "rel_perf": v, "name": etf_names[k]} for k, v in leaders_1m],
            "laggards_1m": [{"ticker": k, "rel_perf": v, "name": etf_names[k]} for k, v in laggards_1m],
            "rotation_signals": rotation_signals
        }
    except Exception as e:
        logger.error("Error calculating sector rotation: %s", e)
        return {}


# ===========================================================================
# Master orchestration function
# ===========================================================================

def analyze_macro(
    market_data: dict,
    treasury_data: dict[str, pd.DataFrame],
    fred_data: dict,
    last_state: dict,
) -> dict:
    """
    Master macro analysis function. Orchestrates all sub-analyses and
    returns a comprehensive macro state dictionary.

    Parameters
    ----------
    market_data : dict
        May contain pre-fetched regime assets under key 'regime_assets'.
        If absent, assets will be fetched fresh.
    treasury_data : dict[str, pd.DataFrame]
        Treasury yield DataFrames (output of fetch_treasury_data()).
    fred_data : dict
        FRED economic data; expected keys 'cpi', 'pce' (float or Series).
    last_state : dict
        Previous macro state; used to detect regime changes.
        Expected structure: {'regime': {'regime': 'RISK-ON'|...}}.

    Returns
    -------
    dict — complete macro state with keys:
        timestamp         (str)  — ISO 8601 UTC
        regime            (dict) — from calculate_regime_score()
        vix               (dict) — from get_vix_status()
        dollar            (dict) — from analyze_dollar_cycle()
        rates             (dict) — from analyze_rate_environment()
        inflation         (dict) — from analyze_inflation()
        intermarket       (dict) — from intermarket_analysis()
        btc_dominance     (dict) — from assess_btc_dominance()
        macro_summary     (str)  — human-readable one-paragraph summary
    """
    logger.info("=" * 60)
    logger.info("Starting macro analysis run — %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)

    # ------------------------------------------------------------------ Regime assets
    regime_assets: dict[str, pd.DataFrame] = {}
    if "regime_assets" in market_data and isinstance(market_data["regime_assets"], dict) and market_data["regime_assets"]:
        regime_assets = market_data["regime_assets"]
        logger.info("Using pre-fetched regime assets from market_data.")
    else:
        logger.info("Fetching regime assets fresh.")
        regime_assets = fetch_regime_assets()

    # ------------------------------------------------------------------ VIX status
    vix_df = regime_assets.get("^VIX", pd.DataFrame())
    vix_status = get_vix_status(vix_df)

    # ------------------------------------------------------------------ Dollar cycle
    dxy_df = regime_assets.get("DX-Y.NYB", pd.DataFrame())
    dollar = analyze_dollar_cycle(dxy_df)

    # ------------------------------------------------------------------ Rate environment
    rates = analyze_rate_environment(treasury_data, regime_assets)

    # ------------------------------------------------------------------ Inflation
    inflation = analyze_inflation(fred_data)

    # ------------------------------------------------------------------ Regime score
    regime = calculate_regime_score(regime_assets, treasury_data, last_state)

    # ------------------------------------------------------------------ Intermarket
    intermarket = intermarket_analysis(regime_assets, treasury_data)

    # ------------------------------------------------------------------ BTC dominance
    btc_dom = assess_btc_dominance()

    # ------------------------------------------------------------------ Cross-asset correlations
    correlations = calculate_correlations()

    # ------------------------------------------------------------------ Macro summary
    macro_summary = _build_macro_summary(regime, vix_status, dollar, rates, inflation, btc_dom)

    # ------------------------------------------------------------------ Sector Rotation
    sector_rotation = calculate_sector_rotation()

    timestamp = datetime.now(timezone.utc).isoformat()

    result = {
        "timestamp": timestamp,
        "regime": regime,
        "vix": vix_status,
        "dollar": dollar,
        "rates": rates,
        "inflation": inflation,
        "intermarket": intermarket,
        "btc_dominance": btc_dom,
        "correlations": correlations,
        "macro_summary": macro_summary,
        "sector_rotation": sector_rotation,
    }

    logger.info(
        "Macro analysis complete — Regime: %s | VIX: %.1f | DXY trend: %s | Rates: %s",
        regime["regime"],
        vix_status.get("current", float("nan")),
        dollar.get("trend", "unknown"),
        rates.get("rate_trend", "unknown"),
    )

    return result


def _build_macro_summary(
    regime: dict,
    vix: dict,
    dollar: dict,
    rates: dict,
    inflation: dict,
    btc_dom: dict,
) -> str:
    """
    Construct a concise human-readable macro overview paragraph.
    """
    parts: list[str] = []

    # Regime sentence
    r = regime.get("regime", "UNKNOWN")
    ron = regime.get("risk_on_score", 0)
    roff = regime.get("risk_off_score", 0)
    changed = regime.get("regime_changed", False)
    regime_clause = (
        f"The market regime is currently {r} "
        f"(risk-on score: {ron}/6, risk-off score: {roff}/6)"
    )
    if changed:
        regime_clause += " — REGIME CHANGE DETECTED"
    parts.append(regime_clause + ".")

    # VIX
    vix_val = vix.get("current", float("nan"))
    if not np.isnan(vix_val):
        spike_note = " (session spike flagged)" if vix.get("spiked") else ""
        parts.append(f"Volatility: VIX at {vix_val:.1f}{spike_note}.")

    # Dollar
    dxy_trend = dollar.get("trend", "unknown")
    dxy_chg = dollar.get("change_5d_pct", float("nan"))
    if not np.isnan(dxy_chg):
        parts.append(
            f"USD: DXY {dxy_trend} ({dxy_chg:+.2f}% over 5 days). {dollar.get('impact', '')}"
        )

    # Rates
    y10 = rates.get("yield_10y", float("nan"))
    curve = rates.get("yield_curve", "unknown")
    r_trend = rates.get("rate_trend", "unknown")
    if not np.isnan(y10):
        parts.append(
            f"Rates: 10Y yield at {y10:.2f}% — curve is {curve}, trend {r_trend}. "
            f"{rates.get('impact', '')}"
        )

    # Inflation
    cpi_val = inflation.get("cpi", float("nan"))
    inf_trend = inflation.get("trend", "unknown")
    if not np.isnan(cpi_val):
        parts.append(f"Inflation: CPI {cpi_val:.1f}% ({inf_trend}). {inflation.get('impact', '')}")

    # BTC dominance
    btc_pct = btc_dom.get("btc_dominance", float("nan"))
    if not np.isnan(btc_pct):
        parts.append(
            f"Crypto: BTC dominance at {btc_pct:.1f}% ({btc_dom.get('trend', 'unknown')}). "
            f"{btc_dom.get('impact', '')}"
        )

    return " ".join(parts)


EXPECTED_CORRELATIONS = {
    ("GC=F", "DX-Y.NYB"): "negative",      # Gold vs DXY (normally inverse)
    ("^TNX", "^GSPC"): "negative",           # 10Y yield vs S&P (normally inverse)
    ("CL=F", "^GSPC"): "positive",           # Oil vs S&P (normally positive in risk-on)
    ("BTC", "^NDX"): "positive",             # BTC vs NASDAQ (positive since 2020)
    ("GC=F", "^VIX"): "positive",            # Gold vs VIX (both rise in fear)
    ("HYG", "^GSPC"): "positive",            # HY bonds vs stocks (risk appetite proxy)
    ("EURUSD=X", "GC=F"): "positive",        # EUR/USD vs Gold (both anti-dollar)
    ("^TNX", "DX-Y.NYB"): "positive",        # Yields vs DXY (higher rates = stronger dollar)
    ("CL=F", "USDINR=X"): "negative",        # Oil vs INR (India imports oil, high oil = weak INR)
}

CORRELATIONS_PATH = os.path.join(os.path.dirname(__file__), "state", "correlations.json")

def calculate_correlations() -> dict:
    """
    Calculate 30-day and 7-day rolling correlations for expected pairs,
    detect anomalies, and write interpretations.
    """
    logger.info("Running cross-asset correlation engine...")
    
    # 1. Collect unique tickers in expected pairs
    unique_tickers = set()
    for pair in EXPECTED_CORRELATIONS.keys():
        unique_tickers.add(pair[0])
        unique_tickers.add(pair[1])
        
    # Translate crypto tickers for yfinance download
    download_tickers = []
    ticker_map = {} # maps standard ticker name to yfinance ticker
    for t in unique_tickers:
        y_t = t
        if t in ["BTC", "ETH", "SOL"]:
            y_t = f"{t}-USD"
        download_tickers.append(y_t)
        ticker_map[t] = y_t
        
    # 2. Download historical close prices (60 days gives at least 30 trading days)
    try:
        raw_df = yf.download(
            tickers=download_tickers,
            period="60d",
            interval="1d",
            auto_adjust=True,
            progress=False
        )
        if "Close" in raw_df.columns:
            prices_df = raw_df["Close"]
        else:
            prices_df = raw_df
    except Exception as exc:
        logger.error("Failed to download correlation data: %s", exc)
        return {"last_updated": datetime.now(timezone.utc).isoformat(), "pairs": {}}

    # Load existing correlations to preserve 'anomaly_since' dates
    existing_corrs = {}
    if os.path.exists(CORRELATIONS_PATH):
        try:
            with open(CORRELATIONS_PATH, "r", encoding="utf-8") as f:
                existing_corrs = json.load(f).get("pairs", {})
        except Exception:
            pass

    corrs_out = {}
    
    # Interpretation templates mapping
    templates = {
        ("GC=F", "DX-Y.NYB"): {
            "rising_both": "Gold and DXY both rising simultaneously (normally inverse) — unusual dollar AND gold strength suggests flight to safety from something specific. Watch for geopolitical catalyst.",
            "falling_both": "Gold and DXY both falling simultaneously (normally inverse) — market moving away from safe havens into other currencies or equities.",
            "default": "Gold and DXY decoupled from their standard inverse relationship."
        },
        ("BTC", "^NDX"): {
            "diverging": "BTC diverging from NASDAQ (normally positive) — crypto moving on its own catalyst, likely ETF flow or on-chain event rather than macro.",
            "default": "BTC has decoupled from NASDAQ, indicating independent crypto-specific drivers."
        },
        ("^TNX", "^GSPC"): {
            "diverging": "10Y yield and S&P 500 moving together (normally inverse) — yields and stocks decoupling. High yields not currently hurting stock sentiment.",
            "default": "Bond yields and stocks are decoupled, reducing macro valuation pressure on equities."
        },
        ("CL=F", "^GSPC"): {
            "diverging": "Oil and S&P 500 diverging (normally positive) — rising energy costs might be worrying equity markets, or demand concerns are decoupling them.",
            "default": "Oil and stock prices decoupled, decoupling inflation expectations from stock performance."
        },
        ("GC=F", "^VIX"): {
            "diverging": "Gold and VIX decoupling (normally positive in fear) — gold rising without a VIX spike suggests steady institutional accumulation rather than panic.",
            "default": "Gold and VIX decoupled, showing protective asset accumulation without short-term panic."
        },
        ("HYG", "^GSPC"): {
            "diverging": "High-yield bonds and stocks decoupling (normally positive) — credit markets showing stress that hasn't fully registered in equity markets yet.",
            "default": "Credit spreads and stocks decoupled, watch for delayed equity reaction to credit stress."
        },
        ("EURUSD=X", "GC=F"): {
            "diverging": "EUR/USD and Gold decoupling (normally positive) — gold moves decoupling from general currency dynamics, showing independent strength.",
            "default": "Euro and Gold decoupled, showing independent gold-specific demand."
        },
        ("^TNX", "DX-Y.NYB"): {
            "diverging": "Yields and DXY decoupling (normally positive) — dollar weakening despite rising yields suggests capital flows favoring other currencies.",
            "default": "Treasury yields and DXY decoupled, showing divergence in US monetary and exchange rate expectations."
        },
        ("CL=F", "USDINR=X"): {
            "diverging": "Oil and INR decoupling (normally inverse) — rupee showing resilience despite rising oil import costs, indicating strong macro inflows.",
            "default": "Oil price and Rupee decoupled, showing local economic strength independent of energy imports."
        }
    }

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for pair, expected in EXPECTED_CORRELATIONS.items():
        t1, t2 = pair
        yt1, yt2 = ticker_map[t1], ticker_map[t2]
        
        # Check if we have columns in prices_df
        col1 = yt1 if yt1 in prices_df.columns else (t1 if t1 in prices_df.columns else None)
        col2 = yt2 if yt2 in prices_df.columns else (t2 if t2 in prices_df.columns else None)
        
        if not col1 or not col2:
            continue
            
        try:
            # Handle multiple tickers DataFrame shape safely
            if isinstance(prices_df, pd.DataFrame):
                s1 = prices_df[col1].dropna()
                s2 = prices_df[col2].dropna()
            else:
                s1 = prices_df.dropna()
                s2 = prices_df.dropna() # fallback
        except Exception:
            continue
        
        # Align series by dates
        aligned = pd.concat([s1, s2], axis=1, join="inner").dropna()
        if len(aligned) < 30:
            continue
            
        # Pearson correlation
        c30d = float(aligned.iloc[-30:, 0].corr(aligned.iloc[-30:, 1]))
        c7d = float(aligned.iloc[-7:, 0].corr(aligned.iloc[-7:, 1]))
        
        if np.isnan(c30d) or np.isnan(c7d):
            continue
            
        # Check anomaly criteria
        anomaly = False
        if expected == "negative" and c30d > 0.3:
            anomaly = True
        elif expected == "positive" and c30d < -0.3:
            anomaly = True
        elif abs(c7d - c30d) > 0.4:
            anomaly = True
            
        # Anomaly tracking since
        pair_key = f"{t1}_{t2}"
        anomaly_since = None
        if anomaly:
            prev_data = existing_corrs.get(pair_key, {})
            if prev_data.get("anomaly"):
                anomaly_since = prev_data.get("anomaly_since") or today_str
            else:
                anomaly_since = today_str
                
        # Generate 2-line interpretation
        interpretation = ""
        if anomaly:
            pair_t = templates.get(pair, templates.get((t2, t1), {}))
            if pair == ("GC=F", "DX-Y.NYB"):
                # Check recent trend to distinguish rising both
                try:
                    s1_recent = aligned.iloc[-5:, 0]
                    s2_recent = aligned.iloc[-5:, 1]
                    if s1_recent.iloc[-1] > s1_recent.iloc[0] and s2_recent.iloc[-1] > s2_recent.iloc[0]:
                        interpretation = pair_t.get("rising_both")
                    elif s1_recent.iloc[-1] < s1_recent.iloc[0] and s2_recent.iloc[-1] < s2_recent.iloc[0]:
                        interpretation = pair_t.get("falling_both")
                except Exception:
                    pass
            if not interpretation:
                interpretation = pair_t.get("diverging", pair_t.get("default", f"Divergence between {t1} and {t2} (normally {expected})"))
        else:
            interpretation = f"Normal aligned relationship ({expected} correlation)."
            
        corrs_out[pair_key] = {
            "expected": expected,
            "correlation_30d": round(c30d, 4),
            "correlation_7d": round(c7d, 4),
            "anomaly": anomaly,
            "anomaly_since": anomaly_since,
            "interpretation": interpretation
        }
        
    payload = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "pairs": corrs_out
    }
    
    # Save to file
    try:
        os.makedirs(os.path.dirname(CORRELATIONS_PATH), exist_ok=True)
        with open(CORRELATIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Saved correlations to %s.", CORRELATIONS_PATH)
    except Exception as exc:
        logger.error("Failed to save correlations: %s", exc)
        
    return payload


# ===========================================================================
# CLI entry point (for standalone testing)
# ===========================================================================

if __name__ == "__main__":
    import pprint

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("\n" + "=" * 70)
    print("  MACRO REGIME DETECTION — STANDALONE TEST RUN")
    print("=" * 70 + "\n")

    # Fetch fresh data
    print("Fetching regime assets...")
    _regime_assets = fetch_regime_assets()

    print("Fetching treasury data...")
    _treasury_data = fetch_treasury_data()

    # Simulate empty FRED data (would be injected by the parent agent in production)
    _fred_data: dict = {}

    # No prior state
    _last_state: dict = {}

    _market_data = {"regime_assets": _regime_assets}

    print("\nRunning full macro analysis...\n")
    _result = analyze_macro(_market_data, _treasury_data, _fred_data, _last_state)

    print("\n" + "=" * 70)
    print("  MACRO ANALYSIS RESULT")
    print("=" * 70)
    pprint.pprint(_result, width=100, sort_dicts=False)

    print("\n" + "-" * 70)
    print("MACRO SUMMARY:")
    print("-" * 70)
    print(_result.get("macro_summary", "No summary generated."))
    print()
