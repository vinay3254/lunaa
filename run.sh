#!/usr/bin/env bash
# =============================================================================
# Trading Research Agent — Runner Script
# =============================================================================
set -euo pipefail

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$AGENT_DIR"

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Ensure dependencies installed
pip install -q -r requirements.txt

# Copy .env if it doesn't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "📋 Created .env from .env.example — edit it to add API keys"
fi

# Ensure directories exist
mkdir -p state reports

# Parse first argument as mode (default: --run)
MODE="${1:---run}"

echo "🚀 Starting Trading Research Agent in mode: $MODE"
python luna.py "$MODE"
