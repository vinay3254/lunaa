# 🌙 LUNA — Autonomous Intermarket Intelligence Agent

LUNA is a highly sophisticated, production-grade autonomous trading research and analysis system. It utilizes intermarket regime detection, cross-asset Pearson correlation matrices, concurrent batched data-gathering pipelines, source-credibility weighted news sentiment analysis, rolling asset memories, active weight auto-tuning, and a natural language ask interface.

---

## 📊 Core Architecture & Features

1. **Mass Monitoring Engine:** Tracks 100+ watchlist assets concurrently using `concurrent.futures.ThreadPoolExecutor` and batches downloads to `yfinance` and `CoinGecko` APIs in groups of 50 to optimize network traffic and avoid rate limits.
2. **Investment Portfolio Tracker:** Calculates position-level metrics (unrealized/realized P&L, stop loss/take profit proximity indicators, days held, max drawdowns, and reward-to-risk ratios) and portfolio-wide capital-at-risk heat maps.
3. **Macro Regime & Sector Rotation:** Analyzes economic indicators, yield spreads, VIX, DXY, and performs a 1W and 1M rolling outperformance analysis of the 11 sector ETFs vs. SPY.
4. **Enhanced Prioritized Alert System:** Employs four distinct alert priority levels (P0 CRITICAL to P3 LOW) with Telegram alerts and a 4-hour duplicate suppression filter tracked in `state/alerts-sent.json`.
5. **Catalyst & Economic Calendar:** Compiles upcoming corporate earnings, token unlock events, and scheduled FRED releases into a unified 14-day chronological schedule.
6. **Dynamic Weights Adaptation:** Automatically tunes scoring weights (global and regime-specific) every 20 completed trades based on historical indicator accuracy (RSI, MACD, Volume, Sentiment, Catalyst) with an underperformance safety reset.
7. **Natural Language Query:** Ephemeral query interface allowing you to ask questions about your portfolio, watchlist, and market reports in plain English.

---

## 🚀 CLI Run Modes

Run LUNA using `python luna.py` with one of the following modes:

```bash
python luna.py --run              # Run full research cycle (macro + scan + portfolio + reports)
python luna.py --schedule         # Continuous scheduler: runs full cycle every 4h, checks alerts every 30m
python luna.py --quick            # Fast cycle (prices + portfolio + alerts only) in under 3 minutes
python luna.py --macro            # Macro + sector rotation + cross-asset correlations dashboard only
python luna.py --scan             # Technical opportunities scanner only using cached data
python luna.py --portfolio        # Investment portfolio status check and report update only
python luna.py --alert-check      # Watchlist price alert check vs last state only
python luna.py --check-outcomes   # Check pending call outcomes (3d/7d) and adapt scoring weights
python luna.py --performance      # Compile historical agent accuracy into reports/agent-performance.md
python luna.py --ask "question"   # Ask a natural language question against all gathered state
python luna.py --help             # Display this help menu and project usage guidelines
```

---
*Disclaimer: Automated research output. Not financial advice. Always do your own research.*
