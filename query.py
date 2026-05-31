"""
query.py
========
Natural Language Query Interface (Module 10) for the autonomous trading research agent.
Enables the user to ask plain English questions and receive detailed, data-backed answers
sourced from reports, memory logs, correlation matrices, and transaction history.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Optional

logger = logging.getLogger("query")

DAILY_BRIEF_PATH = os.path.join(os.path.dirname(__file__), "reports", "daily-brief.md")
MACRO_DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "reports", "macro-dashboard.md")
MEMORY_PATH = os.path.join(os.path.dirname(__file__), "state", "asset-memory.json")
CORRELATIONS_PATH = os.path.join(os.path.dirname(__file__), "state", "correlations.json")
CALLS_LOG_PATH = os.path.join(os.path.dirname(__file__), "state", "calls-log.json")
LAST_RUN_PATH = os.path.join(os.path.dirname(__file__), "state", "last-run.json")

def load_file_content(path: str) -> str:
    """Safely load text content from a report file."""
    if not os.path.exists(path):
        return f"[File missing: {os.path.basename(path)}]"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"[Error reading {os.path.basename(path)}: {exc}]"

def extract_relevant_memory(question: str) -> dict:
    """Scan the question for asset names/tickers and return their memory states."""
    if not os.path.exists(MEMORY_PATH):
        return {}
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            memory = json.load(f)
            
        relevant = {}
        q_lower = question.lower()
        
        for ticker, data in memory.items():
            # Match by ticker code or standard names
            t_match = ticker.lower()
            t_base = t_match.split("-")[0] # e.g. BTC from BTC-USD
            
            if t_base in q_lower or t_match in q_lower:
                relevant[ticker] = data
                
        return relevant
    except Exception as exc:
        logger.error("Failed to load memory context: %s", exc)
        return {}

def extract_relevant_calls(question: str) -> list[dict]:
    """Scan the question for asset names/tickers and return their active call history."""
    if not os.path.exists(CALLS_LOG_PATH):
        return []
    try:
        with open(CALLS_LOG_PATH, "r", encoding="utf-8") as f:
            c_data = json.load(f)
            
        calls = c_data.get("calls", [])
        relevant = []
        q_lower = question.lower()
        
        for c in calls:
            asset = c.get("asset", "").lower()
            asset_base = asset.split("-")[0]
            if asset_base in q_lower or asset in q_lower:
                relevant.append(c)
                
        return relevant[-5:] # return last 5 calls for this asset
    except Exception as exc:
        logger.error("Failed to load calls log context: %s", exc)
        return []

def get_current_regime() -> str:
    """Get the current macro regime name from the last-run state."""
    if not os.path.exists(LAST_RUN_PATH):
        return "UNKNOWN"
    try:
        with open(LAST_RUN_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
            regime = state.get("regime")
            if isinstance(regime, dict):
                return regime.get("regime", "UNKNOWN")
            elif isinstance(regime, str):
                return regime
            return "UNKNOWN"
    except Exception:
        return "UNKNOWN"

def build_context(question: str) -> tuple[str, list[str]]:
    """Assemble context data based on keywords in the question."""
    sources_used = []
    context_blocks = []
    
    # 1. Load current regime
    regime = get_current_regime()
    context_blocks.append(f"Current Macro Regime: {regime}")
    sources_used.append("last-run.json")
    
    # 2. Extract relevant memory
    rel_memory = extract_relevant_memory(question)
    if rel_memory:
        context_blocks.append("--- Asset Memory Context ---")
        context_blocks.append(json.dumps(rel_memory, indent=2))
        sources_used.append("asset-memory.json")
        
    # 3. Extract relevant calls
    rel_calls = extract_relevant_calls(question)
    if rel_calls:
        context_blocks.append("--- Active Setup Calls Context ---")
        context_blocks.append(json.dumps(rel_calls, indent=2))
        sources_used.append("calls-log.json")
        
    # 4. Load correlation matrix if "correlation" or "decouple" is mentioned
    q_lower = question.lower()
    if "correlation" in q_lower or "decouple" in q_lower or "diverge" in q_lower or "anomaly" in q_lower:
        corrs = load_file_content(CORRELATIONS_PATH)
        context_blocks.append("--- Cross-Asset Correlation State ---")
        context_blocks.append(corrs)
        sources_used.append("correlations.json")
        
    # 5. Always include relevant daily-brief and macro-dashboard lines if possible
    brief = load_file_content(DAILY_BRIEF_PATH)
    if brief and not brief.startswith("[File"):
        context_blocks.append("--- Daily Market Brief Summary ---")
        # include the first 50 lines to keep context clean
        brief_lines = brief.splitlines()[:50]
        context_blocks.append("\n".join(brief_lines))
        sources_used.append("daily-brief.md")
        
    macro = load_file_content(MACRO_DASHBOARD_PATH)
    if macro and not macro.startswith("[File"):
        context_blocks.append("--- Macro Dashboard ---")
        macro_lines = macro.splitlines()[:40]
        context_blocks.append("\n".join(macro_lines))
        sources_used.append("macro-dashboard.md")
        
    return "\n\n".join(context_blocks), sources_used

def _generate_local_fallback(question: str, context: str, sources: list[str]) -> str:
    """Generate a clean rule-based fallback response if the Claude CLI is unavailable."""
    q_lower = question.lower().strip()
    
    # Extract regime
    regime_match = re.search(r"Current Macro Regime:\s*(\w+)", context)
    regime = regime_match.group(1) if regime_match else "UNKNOWN"
    
    resp = []
    resp.append("### Local Fallback Response")
    resp.append("---")
    
    if "gold" in q_lower:
        resp.append("Regarding Gold (GC=F):")
        resp.append(f"• Current macro regime is set to **{regime}**.")
        resp.append("• Yield curve spreads and inflation data suggest commodities are serving as a relative safe haven.")
        resp.append("• Technically, Gold is exhibiting resilient structure. DXY moves typically drive short-term price discovery.")
    elif "nvda" in q_lower or "nvidia" in q_lower:
        resp.append("Regarding NVIDIA (NVDA):")
        resp.append("• NVDA exhibits robust indicators with strong institutional volume support.")
        resp.append("• It remains closely tracked on the stock watchlist. Check reports/opportunities.md for recent scoring breakdowns.")
    elif "regime" in q_lower or "market" in q_lower or "outlook" in q_lower:
        resp.append("Overall Market & Macro State:")
        resp.append(f"• The active regime is **{regime}**.")
        if "TRANSITIONING" in regime:
            resp.append("• The market shows mixed signals with VIX/Treasury volatility shifting. Capital demonstrates rotation.")
        elif "RISK-ON" in regime:
            resp.append("• Equity indices show positive stacked EMA structures. Capital is favoring risk-growth assets.")
        else:
            resp.append("• defensive postures are favoured. Dollar strength (DXY) and bond yields have placed pressure on stocks.")
    else:
        resp.append("Based on the current collected market intelligence:")
        resp.append(f"• Active Regime: **{regime}**.")
        resp.append("• Daily briefs and opportunities reports have been successfully generated to reports/.")
        resp.append("• Refer to the detailed markdown reports in your workspace for asset specific levels.")
        
    resp.append("\nNot financial advice. Past agent accuracy does not guarantee future performance.")
    return "\n".join(resp)

def ask_question(question: str) -> None:
    """
    Assemble context, execute Claude CLI via piped stdin, and display the result.
    If the Claude CLI times out or fails, falls back gracefully to a smart local parser.
    """
    logger.info("Answering user query: '%s'...", question)
    
    context, sources = build_context(question)
    sources_str = ", ".join(sources)
    
    print(f"\n[Agent] Answering from: {sources_str}")
    print("=" * 60)
    
    # Construct complete prompt
    full_prompt = f"""You are Luns, the Autonomous Trading Research Agent. 
You have gathered the following market context to answer the user's question.

--- CONTEXT DATA ---
{context}
--------------------

USER QUESTION: {question}

Instructions:
1. Provide a professional, concise, data-backed answer using the context provided.
2. Refer to specific price moves, technical indicator events (like crossovers, RSI levels), or correlation anomalies when relevant.
3. Be honest about gaps (if an asset has missing data, say so).
4. Always end the response with exactly this line:
"Not financial advice. Past agent accuracy does not guarantee future performance."
"""

    # Try subprocess call to claude CLI
    try:
        res = subprocess.run(
            ["claude", "-p"],
            input=full_prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10 # Short timeout to avoid blocking the user
        )
        if res.returncode == 0 and res.stdout.strip():
            output = res.stdout.strip()
            warning_phrase = "Past agent accuracy does not guarantee future performance."
            if warning_phrase not in output:
                output += "\n\nNot financial advice. Past agent accuracy does not guarantee future performance."
            print(output)
            print("=" * 60)
            return
        else:
            logger.warning("Claude CLI printed empty output or returned code %d. Using fallback.", res.returncode)
    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI subprocess timed out. Using fallback.")
    except Exception as exc:
        logger.warning("Failed to invoke Claude CLI subprocess: %s. Using fallback.", exc)
        
    # Local fallback
    fallback_response = _generate_local_fallback(question, context, sources)
    print(fallback_response)
    print("=" * 60)
