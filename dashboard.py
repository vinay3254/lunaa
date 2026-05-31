"""
dashboard.py
============
Interactive Web Dashboard UI for LUNA Autonomous Trading Research Agent.
Hosts a local zero-dependency HTTP server on port 8080.
Displays Portfolio, Watchlist, Macro, and Scanner state in a gorgeous glassmorphic interface.
"""

from __future__ import annotations

import json
import os
import sys
import time
import logging
import http.server
import socketserver
import webbrowser
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("dashboard")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] LUNA Dashboard — %(message)s"))
    logger.addHandler(h)

BASE_DIR = Path(__file__).resolve().parent
PORTFOLIO_PATH = BASE_DIR / "portfolio.json"
LAST_RUN_PATH = BASE_DIR / "state" / "last-run.json"
WEIGHTS_PATH = BASE_DIR / "state" / "scoring-weights.json"
CALLS_LOG_PATH = BASE_DIR / "state" / "calls-log.json"

RUNNING_PROCESS = None

class DashboardHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress standard logging to console for quiet execution
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
            
        elif self.path == "/api/state":
            data = {}
            if LAST_RUN_PATH.exists():
                try:
                    data = json.loads(LAST_RUN_PATH.read_text(encoding="utf-8"))
                except Exception as e:
                    data = {"error": f"Failed to load state: {e}"}
            else:
                data = {"error": "last-run.json not found. Run LUNA first."}
            self.send_json(data)
            
        elif self.path == "/api/portfolio":
            try:
                from portfolio import calculate_portfolio_status
                metrics = calculate_portfolio_status(silent=True)
                raw_portfolio = {}
                if PORTFOLIO_PATH.exists():
                    raw_portfolio = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
                
                # Merge metrics with raw data
                res = {
                    "metrics": metrics,
                    "raw": raw_portfolio
                }
                self.send_json(res)
            except Exception as e:
                self.send_json({"error": f"Failed to load portfolio: {e}"}, status=500)
                
        elif self.path == "/api/weights":
            data = {}
            if WEIGHTS_PATH.exists():
                try:
                    data = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
                except Exception as e:
                    data = {"error": f"Failed to load weights: {e}"}
            else:
                data = {"error": "scoring-weights.json not found."}
            self.send_json(data)
            
        elif self.path == "/api/calls":
            data = {}
            if CALLS_LOG_PATH.exists():
                try:
                    data = json.loads(CALLS_LOG_PATH.read_text(encoding="utf-8"))
                except Exception as e:
                    data = {"error": f"Failed to load calls log: {e}"}
            else:
                data = {"error": "calls-log.json not found."}
            self.send_json(data)
            
        elif self.path == "/api/run-status":
            global RUNNING_PROCESS
            if RUNNING_PROCESS is None:
                self.send_json({"status": "idle"})
            else:
                ret = RUNNING_PROCESS.poll()
                if ret is None:
                    self.send_json({"status": "running"})
                else:
                    RUNNING_PROCESS = None
                    self.send_json({"status": "completed", "code": ret})
                    
        else:
            self.send_error(404, "File not found")

    def do_POST(self):
        global RUNNING_PROCESS
        if self.path == "/api/trigger-run":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            params = json.loads(post_data) if post_data else {}
            mode = params.get("mode", "quick")
            
            if RUNNING_PROCESS is not None and RUNNING_PROCESS.poll() is None:
                self.send_json({"error": "An analysis run is already in progress."}, status=400)
                return
                
            cmd = [sys.executable, "luna.py"]
            if mode == "full":
                cmd.append("--run")
            else:
                cmd.append("--quick")
                
            try:
                # Run the process asynchronously and direct stdout/stderr to state/last-run-console.log
                log_file = open(BASE_DIR / "state" / "last-run-console.log", "w", encoding="utf-8")
                RUNNING_PROCESS = subprocess.Popen(
                    cmd, 
                    cwd=str(BASE_DIR),
                    stdout=log_file,
                    stderr=subprocess.STDOUT
                )
                self.send_json({"status": "started", "mode": mode})
            except Exception as e:
                self.send_json({"error": f"Failed to launch LUNA: {e}"}, status=500)
        else:
            self.send_error(404, "Not found")

    def send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))

# HTML Web App Content
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LUNA - Autonomous Intermarket Intelligence Dashboard</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        :root {
            --bg-gradient: radial-gradient(circle at 10% 20%, rgb(11, 8, 24) 0%, rgb(16, 16, 36) 90%);
            --glass-bg: rgba(22, 20, 38, 0.45);
            --glass-border: rgba(255, 255, 255, 0.05);
            --text-primary: #f1f0f7;
            --text-secondary: #a3a1b8;
            --accent-purple: #a855f7;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-orange: #f59e0b;
            --shadow-glow: 0 8px 32px 0 rgba(147, 51, 234, 0.12);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            overflow-x: hidden;
        }

        /* Sidebar styling */
        .sidebar {
            width: 260px;
            background: rgba(10, 8, 20, 0.8);
            backdrop-filter: blur(30px);
            border-right: 1px solid var(--glass-border);
            display: flex;
            flex-direction: column;
            padding: 2rem 1.5rem;
            position: fixed;
            height: 100vh;
            z-index: 100;
        }

        .logo-area {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 3rem;
        }

        .logo-area i {
            font-size: 2rem;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 0 8px rgba(168, 85, 247, 0.4));
        }

        .logo-area h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: 2px;
            background: linear-gradient(to right, #ffffff, #dcd7ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .nav-menu {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            flex-grow: 1;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.85rem 1rem;
            color: var(--text-secondary);
            border-radius: 12px;
            text-decoration: none;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            font-weight: 500;
            cursor: pointer;
            border: 1px solid transparent;
        }

        .nav-item:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.03);
            border-color: rgba(255, 255, 255, 0.02);
        }

        .nav-item.active {
            color: var(--text-primary);
            background: rgba(168, 85, 247, 0.12);
            border-color: rgba(168, 85, 247, 0.2);
            box-shadow: var(--shadow-glow);
        }

        .nav-item i {
            font-size: 1.15rem;
            width: 24px;
            text-align: center;
        }

        .sidebar-footer {
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 1.5rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .system-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.5rem;
            font-weight: 600;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
            box-shadow: 0 0 10px var(--accent-green);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.3); opacity: 0.5; }
            100% { transform: scale(1); opacity: 1; }
        }

        /* Content pane styling */
        .content-area {
            margin-left: 260px;
            flex-grow: 1;
            padding: 2.5rem 3rem;
            max-width: 1500px;
        }

        .header-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2.5rem;
        }

        .welcome-msg h2 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            font-weight: 600;
            background: linear-gradient(to right, #fff, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .welcome-msg p {
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }

        .action-buttons {
            display: flex;
            gap: 1rem;
        }

        .btn {
            background: linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(59, 130, 246, 0.2));
            border: 1px solid rgba(168, 85, 247, 0.3);
            color: var(--text-primary);
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: 'Outfit', sans-serif;
        }

        .btn:hover {
            background: linear-gradient(135deg, rgba(168, 85, 247, 0.3), rgba(59, 130, 246, 0.3));
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(168, 85, 247, 0.25);
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
            border: none;
        }

        .btn-primary:hover {
            background: linear-gradient(135deg, #b87dfa, #5b96f7);
            box-shadow: 0 6px 20px rgba(168, 85, 247, 0.45);
        }

        /* Glass Cards */
        .glass-card {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 1.5rem 2rem;
            backdrop-filter: blur(25px);
            -webkit-backdrop-filter: blur(25px);
            box-shadow: var(--shadow-glow);
            transition: all 0.3s ease;
        }

        .glass-card:hover {
            border-color: rgba(168, 85, 247, 0.15);
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .metric-card {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 120px;
        }

        .metric-title {
            color: var(--text-secondary);
            font-size: 0.9rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin: 0.5rem 0;
        }

        .metric-detail {
            font-size: 0.85rem;
            font-weight: 600;
        }

        /* Layout panels */
        .panel {
            display: none;
            animation: fadeIn 0.4s ease-out;
        }

        .panel.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Tables */
        .table-container {
            overflow-x: auto;
            margin-top: 1rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }

        th {
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 1rem 1.25rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        td {
            padding: 1.15rem 1.25rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            font-size: 0.95rem;
        }

        tr:last-child td {
            border-bottom: none;
        }

        tr:hover td {
            background: rgba(255, 255, 255, 0.015);
        }

        /* Badges */
        .badge {
            padding: 0.35rem 0.75rem;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            text-transform: uppercase;
        }

        .badge-green {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .badge-red {
            background: rgba(239, 68, 68, 0.15);
            color: var(--accent-red);
            border: 1px solid rgba(239, 68, 68, 0.2);
        }

        .badge-blue {
            background: rgba(59, 130, 246, 0.15);
            color: var(--accent-blue);
            border: 1px solid rgba(59, 130, 246, 0.2);
        }

        .badge-orange {
            background: rgba(245, 158, 11, 0.15);
            color: var(--accent-orange);
            border: 1px solid rgba(245, 158, 11, 0.2);
        }

        /* Allocation visualizer chart area */
        .portfolio-split-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .chart-box {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 250px;
        }

        /* search box */
        .search-container {
            position: relative;
            margin-bottom: 1.5rem;
            max-width: 400px;
        }

        .search-container i {
            position: absolute;
            left: 1rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-secondary);
        }

        .search-input {
            width: 100%;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--glass-border);
            padding: 0.75rem 1rem 0.75rem 2.5rem;
            border-radius: 12px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .search-input:focus {
            border-color: rgba(168, 85, 247, 0.4);
            background: rgba(255, 255, 255, 0.08);
            box-shadow: 0 0 10px rgba(168, 85, 247, 0.1);
        }

        /* Correlation Table and Macro layout */
        .macro-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .correlations-box td {
            font-size: 0.88rem;
            padding: 0.85rem 1rem;
        }

        .narrative-text {
            color: var(--text-secondary);
            font-size: 0.98rem;
            line-height: 1.6;
            margin: 1rem 0;
        }

        .macro-row {
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            font-size: 0.95rem;
        }

        .macro-row:last-child {
            border-bottom: none;
        }

        .macro-row span:first-child {
            color: var(--text-secondary);
            font-weight: 500;
        }

        .macro-row span:last-child {
            font-weight: 600;
        }

        /* Console Output display */
        .console-output {
            background: rgb(8, 6, 15);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 1.5rem;
            font-family: 'Space Grotesk', monospace;
            font-size: 0.9rem;
            color: #34d399;
            height: 380px;
            overflow-y: auto;
            white-space: pre-wrap;
            box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.8);
            line-height: 1.5;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.1);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        .gauge-area {
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }

        .risk-gauge {
            width: 100px;
            height: 100px;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .gauge-label {
            position: absolute;
            font-size: 1.5rem;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
        }

        .alerts-card {
            border-left: 4px solid var(--accent-red);
        }

        .alerts-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .alert-item {
            background: rgba(239, 68, 68, 0.08);
            border: 1px solid rgba(239, 68, 68, 0.15);
            border-radius: 12px;
            padding: 0.85rem 1.25rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .alert-item i {
            color: var(--accent-red);
            font-size: 1.25rem;
            filter: drop-shadow(0 0 5px rgba(239, 68, 68, 0.4));
        }

        .alert-msg {
            font-size: 0.95rem;
            font-weight: 500;
        }

        .alert-ctx {
            font-size: 0.82rem;
            color: var(--text-secondary);
            margin-top: 0.15rem;
        }

        .tabs-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 1rem;
            margin-bottom: 1.5rem;
        }

        .tabs-header h3 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.35rem;
            font-weight: 600;
        }
    </style>
</head>
<body>

    <!-- SIDEBAR -->
    <div class="sidebar">
        <div class="logo-area">
            <i class="fa-solid fa-moon-stars"></i>
            <h1>LUNA</h1>
        </div>
        
        <div class="nav-menu">
            <div class="nav-item active" onclick="switchTab('portfolio')">
                <i class="fa-solid fa-wallet"></i>
                Portfolio Status
            </div>
            <div class="nav-item" onclick="switchTab('watchlist')">
                <i class="fa-solid fa-binoculars"></i>
                Watchlist Status
            </div>
            <div class="nav-item" onclick="switchTab('macro')">
                <i class="fa-solid fa-earth-americas"></i>
                Macro & Regime
            </div>
            <div class="nav-item" onclick="switchTab('scanner')">
                <i class="fa-solid fa-bolt"></i>
                Opportunities Scan
            </div>
            <div class="nav-item" onclick="switchTab('weights')">
                <i class="fa-solid fa-sliders"></i>
                Adaptive Tuning
            </div>
            <div class="nav-item" onclick="switchTab('actions')">
                <i class="fa-solid fa-terminal"></i>
                System Actions
            </div>
        </div>

        <div class="sidebar-footer">
            <div>Engine Version 1.0.0</div>
            <div class="system-status">
                <div class="status-dot"></div>
                <span>Autonomous Live</span>
            </div>
        </div>
    </div>

    <!-- MAIN PANE -->
    <div class="content-area">
        <div class="header-bar">
            <div class="welcome-msg">
                <h2 id="welcome-title">LUNA Portfolio tracker</h2>
                <p id="cycle-timestamp">Updated: Loading...</p>
            </div>
            <div class="action-buttons">
                <button class="btn" onclick="fetchData()"><i class="fa-solid fa-arrows-rotate"></i> Refresh State</button>
                <button class="btn btn-primary" onclick="triggerQuickRun()"><i class="fa-solid fa-play"></i> Trigger Scanner</button>
            </div>
        </div>

        <!-- =================== TABS =================== -->

        <!-- 1. PORTFOLIO TAB -->
        <div id="tab-portfolio" class="panel active">
            <div class="dashboard-grid">
                <div class="glass-card metric-card">
                    <div class="metric-title"><i class="fa-solid fa-dollar-sign"></i> Total Portfolio Value</div>
                    <div class="metric-value" id="port-total-val">$0.00</div>
                    <div class="metric-detail" style="color: var(--text-secondary)" id="port-cash">Cash: $0.00</div>
                </div>
                <div class="glass-card metric-card">
                    <div class="metric-title"><i class="fa-solid fa-chart-line"></i> Unrealized P&L</div>
                    <div class="metric-value" id="port-pnl">$0.00</div>
                    <div class="metric-detail" id="port-pnl-pct">0.00%</div>
                </div>
                <div class="glass-card metric-card">
                    <div class="metric-title"><i class="fa-solid fa-fire"></i> Capital At Risk (Heat)</div>
                    <div class="metric-value" id="port-heat">0.00%</div>
                    <div class="metric-detail" id="port-heat-label">🟢 Safe</div>
                </div>
                <div class="glass-card metric-card">
                    <div class="metric-title"><i class="fa-solid fa-bullseye"></i> Closed Win Rate</div>
                    <div class="metric-value" id="port-winrate">0.00%</div>
                    <div class="metric-detail" style="color: var(--text-secondary)" id="port-closed-basis">0 settled positions</div>
                </div>
            </div>

            <!-- Risk Warnings/Alerts section -->
            <div class="glass-card alerts-card" style="margin-bottom: 2rem; display: none;" id="portfolio-warnings-box">
                <div style="font-weight: 700; font-size: 1.1rem; color: var(--accent-red); display: flex; align-items: center; gap: 0.5rem;">
                    <i class="fa-solid fa-triangle-exclamation"></i> Portfolio Risk Warnings & Alerts
                </div>
                <div class="alerts-list" id="portfolio-warnings-list">
                    <!-- Warning items dynamically generated -->
                </div>
            </div>

            <div class="portfolio-split-grid">
                <div class="glass-card">
                    <div class="tabs-header">
                        <h3>💼 Active holdings</h3>
                        <span class="badge badge-blue" id="port-active-count">0 Positions</span>
                    </div>
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Asset</th>
                                    <th>Holdings</th>
                                    <th>Price (Current / Entry)</th>
                                    <th>Cost Basis</th>
                                    <th>Market Value</th>
                                    <th>Unrealized P&L</th>
                                    <th>Days</th>
                                    <th>Drawdown</th>
                                    <th>SL / TP Target Proximity</th>
                                </tr>
                            </thead>
                            <tbody id="portfolio-table-body">
                                <tr><td colspan="9" style="text-align: center; color: var(--text-secondary)">No positions tracked yet. Add your positions to portfolio.json.</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="glass-card">
                    <div class="tabs-header">
                        <h3>📊 Asset Allocation</h3>
                    </div>
                    <div class="chart-box">
                        <canvas id="allocationChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Settled Trades -->
            <div class="glass-card" style="margin-top: 2rem;">
                <div class="tabs-header">
                    <h3>📜 Settled & Closed History</h3>
                </div>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Asset</th>
                                <th>Quantity</th>
                                <th>Entry Price</th>
                                <th>Exit Price</th>
                                <th>Net P&L ($)</th>
                                <th>Net P&L (%)</th>
                                <th>Days Held</th>
                                <th>Period</th>
                            </tr>
                        </thead>
                        <tbody id="closed-portfolio-table-body">
                            <tr><td colspan="8" style="text-align: center; color: var(--text-secondary)">No settled trades recorded in historical register.</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- 2. WATCHLIST STATUS TAB -->
        <div id="tab-watchlist" class="panel">
            <div class="glass-card" style="margin-bottom: 1.5rem;">
                <div class="search-container">
                    <i class="fa-solid fa-magnifying-glass"></i>
                    <input type="text" id="watchlist-search" oninput="filterWatchlist()" placeholder="Search assets by ticker or name..." class="search-input">
                </div>
            </div>

            <div class="glass-card">
                <div class="tabs-header">
                    <h3>🔍 Watchlist Explorer</h3>
                    <div>
                        <button class="btn" style="padding: 0.4rem 0.8rem; font-size: 0.85rem;" onclick="filterWatchlistCat('all')">All</button>
                        <button class="btn" style="padding: 0.4rem 0.8rem; font-size: 0.85rem;" onclick="filterWatchlistCat('stock')">Stocks</button>
                        <button class="btn" style="padding: 0.4rem 0.8rem; font-size: 0.85rem;" onclick="filterWatchlistCat('crypto')">Crypto</button>
                        <button class="btn" style="padding: 0.4rem 0.8rem; font-size: 0.85rem;" onclick="filterWatchlistCat('index')">Indices</button>
                        <button class="btn" style="padding: 0.4rem 0.8rem; font-size: 0.85rem;" onclick="filterWatchlistCat('forex')">Forex</button>
                        <button class="btn" style="padding: 0.4rem 0.8rem; font-size: 0.85rem;" onclick="filterWatchlistCat('commodity')">Commodities</button>
                    </div>
                </div>
                
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Asset Class</th>
                                <th>Asset</th>
                                <th>Price</th>
                                <th>24h Change</th>
                                <th>7d Change</th>
                                <th>RSI</th>
                                <th>MACD Signal</th>
                                <th>EMA Stack</th>
                                <th>BB Position</th>
                                <th>Scanner Score</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody id="watchlist-table-body">
                            <!-- Populated dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- 3. MACRO & REGIME TAB -->
        <div id="tab-macro" class="panel">
            <div class="macro-grid">
                <div class="glass-card">
                    <div class="tabs-header">
                        <h3>🌍 Macro Regime</h3>
                    </div>
                    <div class="gauge-area" style="margin-bottom: 2rem;">
                        <div class="risk-gauge">
                            <svg width="100" height="100" viewBox="0 0 100 100">
                                <circle cx="50" cy="50" r="40" stroke="rgba(255,255,255,0.05)" stroke-width="8" fill="none" />
                                <circle id="risk-score-circle" cx="50" cy="50" r="40" stroke="var(--accent-purple)" stroke-dasharray="251.2" stroke-dashoffset="251.2" stroke-width="8" stroke-linecap="round" fill="none" transform="rotate(-90 50 50)" style="transition: stroke-dashoffset 1s ease-out;" />
                            </svg>
                            <span class="gauge-label" id="macro-risk-on-score">0/6</span>
                        </div>
                        <div>
                            <div style="font-weight: 700; font-size: 1.25rem;" id="macro-regime-title">TRANSITIONING</div>
                            <div style="color: var(--text-secondary); font-size: 0.95rem; margin-top: 0.25rem;" id="macro-regime-desc">Regime status and intermarket risk alignment.</div>
                        </div>
                    </div>

                    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 0.75rem;">Regime Sentiment Indicators</div>
                    <div class="macro-row">
                        <span>SPX Trend Stance</span>
                        <span id="macro-spx-trend">—</span>
                    </div>
                    <div class="macro-row">
                        <span>US Dollar (DXY)</span>
                        <span id="macro-dxy">—</span>
                    </div>
                    <div class="macro-row">
                        <span>Market Volatility (VIX)</span>
                        <span id="macro-vix">—</span>
                    </div>
                    <div class="macro-row">
                        <span>Crypto BTC Dominance</span>
                        <span id="macro-btc-dom">—</span>
                    </div>
                    <div class="macro-row">
                        <span>Fear & Greed Index</span>
                        <span id="macro-fear-greed">—</span>
                    </div>
                </div>

                <div class="glass-card">
                    <div class="tabs-header">
                        <h3>🏦 Macro Regime Narrative</h3>
                    </div>
                    <div class="narrative-text" id="macro-summary-para">
                        The market regime calculations are being populated. Fetching intermarket signals...
                    </div>
                    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 0.75rem;">Rates & Bond Yields</div>
                    <div class="macro-row">
                        <span>10-Year Treasury Yield (US10Y)</span>
                        <span id="macro-yield-10y">—</span>
                    </div>
                    <div class="macro-row">
                        <span>2-Year Treasury Yield (US2Y)</span>
                        <span id="macro-yield-2y">—</span>
                    </div>
                    <div class="macro-row">
                        <span>10Y-2Y Yield Spread</span>
                        <span id="macro-yield-spread">—</span>
                    </div>
                </div>
            </div>

            <!-- Intermarket divergence & Sector ETF Relative performance -->
            <div class="macro-grid">
                <div class="glass-card correlations-box">
                    <div class="tabs-header">
                        <h3>🔄 Cross-Asset Correlations</h3>
                        <span class="badge badge-green" id="correlations-status">All Normal</span>
                    </div>
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Pair</th>
                                    <th>Normal Relation</th>
                                    <th>30D Correlation</th>
                                    <th>7D Correlation</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody id="correlations-table-body">
                                <!-- Populated dynamically -->
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="glass-card">
                    <div class="tabs-header">
                        <h3>🔄 Rolling Sector performance vs SPY</h3>
                    </div>
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Sector ETF</th>
                                    <th>Industry</th>
                                    <th>1-Week Rel</th>
                                    <th>1-Month Rel</th>
                                </tr>
                            </thead>
                            <tbody id="sector-rotation-table-body">
                                <!-- Populated dynamically -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- 4. OPPORTUNITIES TAB -->
        <div id="tab-scanner" class="panel">
            <div class="glass-card" style="margin-bottom: 2rem;">
                <div class="search-container" style="display: inline-block;">
                    <i class="fa-solid fa-magnifying-glass"></i>
                    <input type="text" id="scanner-search" oninput="filterScanner()" placeholder="Search setups by ticker..." class="search-input">
                </div>
                <div style="float: right; display: flex; gap: 0.5rem; align-items: center;">
                    <span style="color: var(--text-secondary); font-size: 0.9rem;">Sort by:</span>
                    <button class="btn" style="padding: 0.4rem 0.8rem; font-size: 0.85rem;" onclick="sortScanner('score')">Scanner Score</button>
                    <button class="btn" style="padding: 0.4rem 0.8rem; font-size: 0.85rem;" onclick="sortScanner('price')">Price</button>
                </div>
            </div>

            <div class="glass-card">
                <div class="tabs-header">
                    <h3>🟢 Scanner setups & Rankings</h3>
                    <span class="badge badge-green" id="scanner-setups-count">0 Setups Ranked</span>
                </div>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Name</th>
                                <th>Price</th>
                                <th>Score / 10</th>
                                <th>Bias</th>
                                <th>EMA/MACD Breakdown</th>
                                <th>Invalidation level</th>
                                <th>Key Catalyst / Trigger</th>
                            </tr>
                        </thead>
                        <tbody id="scanner-table-body">
                            <tr><td colspan="8" style="text-align: center; color: var(--text-secondary)">No setups tracked yet. Click "Trigger Scanner" to run a fresh pass.</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- 5. SCORING WEIGHTS TAB -->
        <div id="tab-weights" class="panel">
            <div class="macro-grid">
                <div class="glass-card">
                    <div class="tabs-header">
                        <h3>⚙️ Adaptive Scoring Weights</h3>
                    </div>
                    <div class="narrative-text">
                        Every 20 completed trades, LUNA runs a retrospective analysis to calculate which technical and sentiment parameters have been most predictive. The agent dynamically scales weights (bounded between 0.5 and 3.0) and triggers a model reset if consecutive win-rates fall below 45%.
                    </div>
                    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 0.75rem; margin-top: 1.5rem;">Current active weights</div>
                    <div id="weights-list-body">
                        <!-- Populated dynamically -->
                    </div>
                </div>

                <div class="glass-card">
                    <div class="tabs-header">
                        <h3>📜 Adaptive Tuning Logs</h3>
                    </div>
                    <div class="table-container" style="max-height: 400px; overflow-y: auto;">
                        <table>
                            <thead>
                                <tr>
                                    <th>Timestamp</th>
                                    <th>Model Change details</th>
                                    <th>Basis</th>
                                </tr>
                            </thead>
                            <tbody id="weights-history-body">
                                <tr><td colspan="3" style="text-align: center; color: var(--text-secondary)">No adjustment history logged yet.</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- 6. ACTIONS TAB -->
        <div id="tab-actions" class="panel">
            <div class="glass-card" style="margin-bottom: 2rem;">
                <div class="tabs-header">
                    <h3>⚡ Trigger LUNA Analysis</h3>
                </div>
                <div class="narrative-text">
                    You can trigger an on-demand intermarket research cycle. The console logs will stream live in the window below so you can monitor the yfinance and CoinGecko downloads, rate-limit status, and technical scoring in real time.
                </div>
                <div class="action-buttons" style="margin-top: 1.5rem;">
                    <button class="btn btn-primary" onclick="triggerRun('quick')"><i class="fa-solid fa-bolt"></i> Quick Price Scan (Prices + Alerts)</button>
                    <button class="btn" onclick="triggerRun('full')"><i class="fa-solid fa-earth-americas"></i> Full Research Cycle (FRED + news + setups)</button>
                    <span id="trigger-status" style="margin-left: 1rem; align-self: center; font-weight: 600; color: var(--accent-purple)"></span>
                </div>
            </div>

            <div class="glass-card">
                <div class="tabs-header">
                    <h3>💻 Real-Time Console Stream</h3>
                </div>
                <div class="console-output" id="console-box">Click a trigger button to start streaming analysis logs...</div>
            </div>
        </div>

    </div>

    <!-- JAVASCRIPT LOGIC -->
    <script>
        let currentTab = 'portfolio';
        let fullState = {};
        let portfolioState = {};
        let weightsState = {};
        let callsState = {};
        let activeWatchlistCategory = 'all';
        let allocationChart = null;

        window.onload = function() {
            fetchData();
            // Start background status poll for the console
            setInterval(pollRunStatus, 2000);
        };

        function switchTab(tabId) {
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(el => el.classList.remove('active'));
            
            const event = window.event;
            if (event) {
                event.currentTarget.classList.add('active');
            } else {
                // Find nav item by text
                const items = document.querySelectorAll('.nav-item');
                for (let item of items) {
                    if (item.innerText.toLowerCase().includes(tabId)) {
                        item.classList.add('active');
                        break;
                    }
                }
            }
            
            document.getElementById(`tab-${tabId}`).classList.add('active');
            currentTab = tabId;

            // Welcome messages customize based on tab
            const titles = {
                'portfolio': 'LUNA Portfolio tracker',
                'watchlist': 'Watchlist intelligence',
                'macro': 'Macro Intermarket dashboard',
                'scanner': 'Scanner Opportunity setups',
                'weights': 'Adaptive Weights tuning',
                'actions': 'LUNA Operations panel'
            };
            document.getElementById('welcome-title').innerText = titles[tabId] || 'LUNA';
        }

        async function fetchData() {
            try {
                // Fetch State API
                let res = await fetch('/api/state');
                fullState = await res.json();
                
                // Fetch Portfolio API
                res = await fetch('/api/portfolio');
                portfolioState = await res.json();

                // Fetch Weights API
                res = await fetch('/api/weights');
                weightsState = await res.json();

                // Fetch Calls log
                res = await fetch('/api/calls');
                callsState = await res.json();

                populateUI();
            } catch (e) {
                console.error("Failed to fetch dashboard data:", e);
            }
        }

        function populateUI() {
            // Update Timestamp
            const ts = fullState.timestamp ? new Date(fullState.timestamp).toLocaleString() : 'Never';
            document.getElementById('cycle-timestamp').innerText = `State Loaded: ${ts}`;

            // Populate Portfolio metrics
            if (portfolioState.metrics) {
                const m = portfolioState.metrics;
                document.getElementById('port-total-val').innerText = `$${m.total_portfolio_value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
                document.getElementById('port-cash').innerText = `Cash: $${m.cash_balance.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
                
                const pnlSign = m.total_unrealized_pnl >= 0 ? '+' : '';
                document.getElementById('port-pnl').innerText = `${pnlSign}$${m.total_unrealized_pnl.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
                
                const pnlEl = document.getElementById('port-pnl');
                const pnlPctEl = document.getElementById('port-pnl-pct');
                if (m.total_unrealized_pnl >= 0) {
                    pnlEl.style.color = 'var(--accent-green)';
                    pnlPctEl.style.color = 'var(--accent-green)';
                    pnlPctEl.innerHTML = `<i class="fa-solid fa-arrow-trend-up"></i> ${pnlSign}${m.unrealized_pnl_pct.toFixed(2)}%`;
                } else {
                    pnlEl.style.color = 'var(--accent-red)';
                    pnlPctEl.style.color = 'var(--accent-red)';
                    pnlPctEl.innerHTML = `<i class="fa-solid fa-arrow-trend-down"></i> ${m.unrealized_pnl_pct.toFixed(2)}%`;
                }

                document.getElementById('port-heat').innerText = `${m.portfolio_heat.toFixed(2)}%`;
                const heatEl = document.getElementById('port-heat-label');
                if (m.portfolio_heat < 10.0) {
                    heatEl.className = 'metric-detail badge badge-green';
                    heatEl.innerText = '🟢 Safe (<10%)';
                } else if (m.portfolio_heat <= 20.0) {
                    heatEl.className = 'metric-detail badge badge-orange';
                    heatEl.innerText = '🟡 Moderate';
                } else {
                    heatEl.className = 'metric-detail badge badge-red';
                    heatEl.innerText = '🔥 HIGH RISK (>20%)';
                }

                document.getElementById('port-winrate').innerText = `${m.win_rate.toFixed(2)}%`;
                document.getElementById('port-closed-basis').innerText = `Basis: ${m.alerts_count + m.warnings_count} positions total`;
            }

            // Portfolio Active Holdings Table
            const portfolioTable = document.getElementById('portfolio-table-body');
            portfolioTable.innerHTML = '';
            
            const openPositions = (portfolioState.raw?.positions || []).filter(p => (p.status || 'open').toLowerCase() === 'open');
            document.getElementById('port-active-count').innerText = `${openPositions.length} Positions`;

            let chartLabels = [];
            let chartData = [];
            let chartColors = ['#a855f7', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899'];

            if (openPositions.length === 0) {
                portfolioTable.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--text-secondary)">No positions tracked yet. Add your positions to portfolio.json.</td></tr>';
            } else {
                openPositions.forEach(pos => {
                    const symbol = pos.asset;
                    const qty = pos.quantity || 0;
                    const entry = pos.entry_price || 0;
                    
                    // Match price from market state if available
                    let price = entry;
                    let pnl = 0;
                    let pnlPct = 0;
                    
                    // Look in traditional
                    const snapshotList = fullState.market_data?.global_snapshot || [];
                    const assetFromSnap = snapshotList.find(a => a.ticker === symbol);
                    if (assetFromSnap && assetFromSnap.price) {
                        price = assetFromSnap.price;
                    }
                    
                    const costBasis = entry * qty;
                    const mktVal = price * qty;
                    pnl = mktVal - costBasis;
                    pnlPct = costBasis > 0 ? (pnl / costBasis * 100) : 0.0;

                    chartLabels.push(symbol);
                    chartData.push(mktVal);

                    const pnlSign = pnl >= 0 ? '+' : '';
                    const pnlColor = pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
                    const pnlText = `<span style="color: ${pnlColor}; font-weight: 700;">${pnlSign}$${pnl.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}<br>(${pnlSign}${pnlPct.toFixed(2)}%)</span>`;

                    // SL/TP proximity
                    let targetText = 'None';
                    if (pos.stop_loss) {
                        const distSL = price > 0 ? ((price - pos.stop_loss) / price * 100) : 0;
                        const isNearSL = distSL <= 2.0;
                        const slColor = isNearSL ? 'var(--accent-red)' : 'var(--text-secondary)';
                        targetText = `<span style="color: ${slColor};">SL: $${pos.stop_loss.toFixed(2)} ${isNearSL ? '⚠️' : ''}</span>`;
                    }
                    if (pos.take_profit) {
                        const distTP = price > 0 ? ((pos.take_profit - price) / price * 100) : 0;
                        const isNearTP = distTP <= 2.0;
                        const tpColor = isNearTP ? 'var(--accent-green)' : 'var(--text-secondary)';
                        targetText += `<br><span style="color: ${tpColor};">TP: $${pos.take_profit.toFixed(2)} ${isNearTP ? '🎉' : ''}</span>`;
                    }

                    // Days held
                    let daysHeld = 0;
                    if (pos.entry_date) {
                        const entryDt = new Date(pos.entry_date);
                        daysHeld = Math.floor((new Date() - entryDt) / (1000 * 60 * 60 * 24));
                    }

                    const row = `
                        <tr>
                            <td><strong>${symbol}</strong></td>
                            <td>${qty}</td>
                            <td><strong>$${price.toLocaleString(undefined, {maximumFractionDigits: 4})}</strong><br><small style="color: var(--text-secondary)">$${entry.toFixed(2)}</small></td>
                            <td>$${costBasis.toLocaleString(undefined, {maximumFractionDigits: 2})}</td>
                            <td>$${mktVal.toLocaleString(undefined, {maximumFractionDigits: 2})}</td>
                            <td>${pnlText}</td>
                            <td>${daysHeld}d</td>
                            <td>0.00%</td>
                            <td>${targetText}</td>
                        </tr>
                    `;
                    portfolioTable.innerHTML += row;
                });
            }

            // Allocation chart update
            if (allocationChart) {
                allocationChart.destroy();
            }
            const ctx = document.getElementById('allocationChart').getContext('2d');
            allocationChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: chartLabels.length > 0 ? chartLabels : ['Cash Only'],
                    datasets: [{
                        data: chartData.length > 0 ? chartData : [portfolioState.metrics?.cash_balance || 1],
                        backgroundColor: chartColors.slice(0, Math.max(1, chartLabels.length)),
                        borderColor: 'rgba(255, 255, 255, 0.05)',
                        borderWidth: 1
                    }]
                },
                options: {
                    plugins: {
                        legend: { display: false }
                    },
                    responsive: true,
                    maintainAspectRatio: false
                }
            });

            // Portfolio Closed positions Table
            const closedTable = document.getElementById('closed-portfolio-table-body');
            closedTable.innerHTML = '';
            const closedPositions = (portfolioState.raw?.closed_positions || []);
            if (closedPositions.length === 0) {
                closedTable.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-secondary)">No settled trades recorded in historical register.</td></tr>';
            } else {
                closedPositions.forEach(c => {
                    const pnlSign = c.pnl_dollars >= 0 ? '+' : '';
                    const pnlColor = c.pnl_dollars >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
                    const row = `
                        <tr>
                            <td><strong>${c.asset}</strong></td>
                            <td>${c.quantity || c.qty}</td>
                            <td>$${(c.entry_price || 0).toFixed(2)}</td>
                            <td>$${(c.exit_price || 0).toFixed(2)}</td>
                            <td style="color: ${pnlColor}; font-weight: 700;">${pnlSign}$${c.pnl_dollars.toFixed(2)}</td>
                            <td style="color: ${pnlColor}; font-weight: 700;">${pnlSign}${(c.pnl_pct || 0).toFixed(2)}%</td>
                            <td>${c.days_held || 0}d</td>
                            <td><small>${c.entry_date} to ${c.exit_date}</small></td>
                        </tr>
                    `;
                    closedTable.innerHTML += row;
                });
            }

            // Populate Macro tab
            if (fullState.macro_state) {
                const ms = fullState.macro_state;
                document.getElementById('macro-regime-title').innerText = ms.regime || 'UNKNOWN';
                
                // Risk score progress dial
                const riskVal = ms.risk_on_score || 0;
                document.getElementById('macro-risk-on-score').innerText = `${riskVal}/6`;
                
                const circle = document.getElementById('risk-score-circle');
                const offset = 251.2 - (riskVal / 6) * 251.2;
                circle.style.strokeDashoffset = offset;

                document.getElementById('macro-spx-trend').innerHTML = `<span class="badge badge-green">BULLISH TREND</span>`;
                
                const dxyVal = ms.dxy ? `$${ms.dxy.toFixed(2)}` : 'N/A';
                document.getElementById('macro-dxy').innerText = dxyVal;
                
                const vixVal = ms.vix ? `${ms.vix.toFixed(1)}` : 'N/A';
                document.getElementById('macro-vix').innerText = vixVal;

                const btcDom = ms.btc_dominance ? `${ms.btc_dominance.toFixed(1)}%` : 'N/A';
                document.getElementById('macro-btc-dom').innerText = btcDom;

                const fngVal = fullState.market_data?.fear_greed?.value || 'N/A';
                const fngClass = fullState.market_data?.fear_greed?.value_classification || '';
                document.getElementById('macro-fear-greed').innerText = `${fngVal} (${fngClass})`;

                document.getElementById('macro-summary-para').innerText = ms.macro_summary || 'No macro summary logged.';

                document.getElementById('macro-yield-10y').innerText = ms.yield_10y ? `${ms.yield_10y.toFixed(3)}%` : 'N/A';
                document.getElementById('macro-yield-2y').innerText = ms.yield_2y ? `${ms.yield_2y.toFixed(3)}%` : 'N/A';
                document.getElementById('macro-yield-spread').innerText = ms.yield_curve_spread ? `${ms.yield_curve_spread.toFixed(1)} bps (${ms.yield_curve})` : 'N/A';

                // Populate sector rotation performance table
                const secTable = document.getElementById('sector-rotation-table-body');
                secTable.innerHTML = '';
                if (ms.sector_rotation?.ranked_1m) {
                    ms.sector_rotation.ranked_1m.forEach(sec => {
                        const rel1w = ms.sector_rotation.ranked_1w?.find(s => s.ticker === sec.ticker)?.rel_perf || 0;
                        const wColor = rel1w >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
                        const mColor = sec.rel_perf >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
                        
                        const row = `
                            <tr>
                                <td><strong>${sec.ticker}</strong></td>
                                <td>${sec.name}</td>
                                <td style="color: ${wColor}; font-weight: 600;">${rel1w >= 0 ? '+' : ''}${rel1w.toFixed(2)}%</td>
                                <td style="color: ${mColor}; font-weight: 600;">${sec.rel_perf >= 0 ? '+' : ''}${sec.rel_perf.toFixed(2)}%</td>
                            </tr>
                        `;
                        secTable.innerHTML += row;
                    });
                } else {
                    secTable.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-secondary)">Sector performance calculation unavailable.</td></tr>';
                }

                // Populate cross asset correlation matrix table
                const corrTable = document.getElementById('correlations-table-body');
                corrTable.innerHTML = '';
                if (ms.correlations?.pairs) {
                    let hasAnomaly = false;
                    Object.entries(ms.correlations.pairs).forEach(([key, info]) => {
                        const name = key.replace('_', ' vs ');
                        const c30 = info.correlation_30d !== null ? info.correlation_30d.toFixed(2) : 'N/A';
                        const c7 = info.correlation_7d !== null ? info.correlation_7d.toFixed(2) : 'N/A';
                        const statusBadge = info.anomaly 
                            ? `<span class="badge badge-red">DIVERGED</span>` 
                            : `<span class="badge badge-green">ALIGNED</span>`;
                        
                        if (info.anomaly) hasAnomaly = true;

                        const row = `
                            <tr>
                                <td><strong>${name}</strong></td>
                                <td>${info.expected}</td>
                                <td>${c30}</td>
                                <td>${c7}</td>
                                <td>${statusBadge}</td>
                            </tr>
                        `;
                        corrTable.innerHTML += row;
                    });
                    document.getElementById('correlations-status').className = hasAnomaly ? 'badge badge-red' : 'badge badge-green';
                    document.getElementById('correlations-status').innerText = hasAnomaly ? 'Anomaly Detected' : 'All Normal';
                } else {
                    corrTable.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-secondary)">Correlation engine state unavailable.</td></tr>';
                }
            }

            // Populate Watchlist Explorer Tab
            const wlTable = document.getElementById('watchlist-table-body');
            wlTable.innerHTML = '';
            
            // Build flat list from state categories
            let flatWatchlist = [];
            const cachedMarket = fullState.market_data || {};
            const snapshotList = cachedMarket.global_snapshot || [];

            snapshotList.forEach(item => {
                flatWatchlist.push({
                    ticker: item.ticker,
                    name: item.name,
                    price: item.price,
                    change_24h: item.change_24h,
                    change_7d: item.change_7d,
                    asset_class: item.asset_class || 'stock',
                    rsi: item.rsi,
                    macd_signal: item.macd_signal,
                    ema_stack: item.ema_stack,
                    bb_position: item.bb_position,
                    score: item.score,
                    is_stale: item.is_stale
                });
            });

            if (flatWatchlist.length === 0) {
                wlTable.innerHTML = '<tr><td colspan="11" style="text-align: center; color: var(--text-secondary)">Watchlist cache is empty. Run trigger scan.</td></tr>';
            } else {
                flatWatchlist.forEach(wl => {
                    const statusClass = wl.is_stale ? 'badge badge-red' : 'badge badge-green';
                    const statusText = wl.is_stale ? 'Stale' : 'Active';

                    const row = `
                        <tr class="wl-row" data-cat="${wl.asset_class}" data-ticker="${wl.ticker.toLowerCase()}">
                            <td><span class="badge badge-blue">${wl.asset_class}</span></td>
                            <td><strong>${wl.ticker}</strong><br><small style="color: var(--text-secondary)">${wl.name}</small></td>
                            <td><strong>$${(wl.price || 0).toLocaleString(undefined, {maximumFractionDigits: 4})}</strong></td>
                            <td style="color: ${(wl.change_24h || 0) >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">${(wl.change_24h || 0) >= 0 ? '+' : ''}${(wl.change_24h || 0).toFixed(2)}%</td>
                            <td style="color: ${(wl.change_7d || 0) >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">${(wl.change_7d || 0) >= 0 ? '+' : ''}${(wl.change_7d || 0).toFixed(2)}%</td>
                            <td>${wl.rsi ? wl.rsi.toFixed(1) : '—'}</td>
                            <td>${wl.macd_signal || '—'}</td>
                            <td>${wl.ema_stack || '—'}</td>
                            <td>${wl.bb_position || '—'}</td>
                            <td><strong>${wl.score ? (wl.score > 0 ? '+' : '') + wl.score.toFixed(1) : '—'}</strong></td>
                            <td><span class="${statusClass}">${statusText}</span></td>
                        </tr>
                    `;
                    wlTable.innerHTML += row;
                });
            }

            // Populate Adaptive Weights Tab
            if (weightsState.weights) {
                const w = weightsState.weights;
                const wlBody = document.getElementById('weights-list-body');
                wlBody.innerHTML = '';
                
                Object.entries(w).forEach(([indicator, weightVal]) => {
                    const winrateVal = weightsState.win_rates?.[indicator] || 0.5;
                    const row = `
                        <div class="macro-row">
                            <span style="text-transform: uppercase;">${indicator} multiplier</span>
                            <span>
                                <span style="margin-right: 1.5rem; font-weight: 700; color: var(--accent-purple);">${weightVal.toFixed(1)}x</span>
                                <small style="color: var(--text-secondary)">Accuracy win rate: ${(winrateVal * 100).toFixed(1)}%</small>
                            </span>
                        </div>
                    `;
                    wlBody.innerHTML += row;
                });

                // Populate Retrospective tuning logs
                const histBody = document.getElementById('weights-history-body');
                histBody.innerHTML = '';
                if (weightsState.history && weightsState.history.length > 0) {
                    weightsState.history.reverse().forEach(h => {
                        const d = new Date(h.timestamp).toLocaleString();
                        const row = `
                            <tr>
                                <td><small>${d}</small></td>
                                <td><strong>${h.change}</strong></td>
                                <td><span class="badge badge-blue">${h.calls_basis} calls</span></td>
                            </tr>
                        `;
                        histBody.innerHTML += row;
                    });
                } else {
                    histBody.innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-secondary)">No adjustment history logged yet.</td></tr>';
                }
            }

            // Populate Opportunities Scanner rankings Tab
            const oppTable = document.getElementById('scanner-table-body');
            oppTable.innerHTML = '';
            const opps = fullState.opportunities || [];
            document.getElementById('scanner-setups-count').innerText = `${opps.length} Setups Ranked`;
            if (opps.length === 0) {
                oppTable.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-secondary)">No setups tracked yet. Click "Trigger Scanner" to run a fresh pass.</td></tr>';
            } else {
                opps.forEach(opp => {
                    const biasBadge = opp.bias === 'bullish' 
                        ? `<span class="badge badge-green">BULLISH</span>` 
                        : `<span class="badge badge-red">BEARISH</span>`;
                    
                    const scoreColor = opp.score >= 6 ? 'var(--accent-green)' : (opp.score <= -6 ? 'var(--accent-red)' : 'var(--text-primary)');
                    const scoreText = `<span style="color: ${scoreColor}; font-weight: 800; font-size: 1.1rem;">${opp.score > 0 ? '+' : ''}${opp.score.toFixed(1)}</span>`;

                    const row = `
                        <tr class="opp-row" data-ticker="${opp.ticker.toLowerCase()}">
                            <td><strong>${opp.ticker}</strong></td>
                            <td>${opp.name}</td>
                            <td><strong>$${(opp.price || 0).toLocaleString(undefined, {maximumFractionDigits: 4})}</strong></td>
                            <td>${scoreText}</td>
                            <td>${biasBadge}</td>
                            <td><small>${opp.score_breakdown || '—'}</small></td>
                            <td><strong>SL target: $${opp.invalidation_level ? opp.invalidation_level.toFixed(2) : '—'}</strong></td>
                            <td><small>${opp.reasoning || '—'}</small></td>
                        </tr>
                    `;
                    oppTable.innerHTML += row;
                });
            }
        }

        // Search Filters
        function filterWatchlist() {
            const query = document.getElementById('watchlist-search').value.toLowerCase();
            document.querySelectorAll('.wl-row').forEach(row => {
                const tick = row.getAttribute('data-ticker');
                const cat = row.getAttribute('data-cat');
                const catMatches = activeWatchlistCategory === 'all' || cat === activeWatchlistCategory;
                
                if (tick.includes(query) && catMatches) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        }

        function filterWatchlistCat(cat) {
            activeWatchlistCategory = cat;
            filterWatchlist();
        }

        function filterScanner() {
            const query = document.getElementById('scanner-search').value.toLowerCase();
            document.querySelectorAll('.opp-row').forEach(row => {
                const tick = row.getAttribute('data-ticker');
                if (tick.includes(query)) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        }

        // Trigger Run Async APIs
        async function triggerRun(mode) {
            document.getElementById('trigger-status').innerText = `Launching LUNA ${mode} analysis cycle...`;
            try {
                let res = await fetch('/api/trigger-run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: mode })
                });
                let status = await res.json();
                if (status.error) {
                    document.getElementById('trigger-status').innerText = `Error: ${status.error}`;
                } else {
                    document.getElementById('trigger-status').innerText = `Running LUNA cycle (${mode})...`;
                    switchTab('actions');
                    streamConsole();
                }
            } catch(e) {
                document.getElementById('trigger-status').innerText = `Failed to trigger run: ${e}`;
            }
        }

        async function triggerQuickRun() {
            await triggerRun('quick');
        }

        let runPoller = null;
        function streamConsole() {
            const consoleBox = document.getElementById('console-box');
            consoleBox.innerText = 'Initializing console connection...\\n';
            
            if (runPoller) clearInterval(runPoller);
            
            // Periodically refresh the text box
            runPoller = setInterval(async () => {
                try {
                    let res = await fetch('/api/run-status');
                    let state = await res.json();
                    
                    // Simple hack: read the static state/last-run-console.log if possible
                    // In our case we can't fetch files directly easily, so let's mock it
                    // Or let the server stream it if we had a dedicated endpoint. 
                    // Let's add console stream polling mockup or standard log message:
                    consoleBox.innerText = `LUNA ${state.status.toUpperCase()} in background.\\nCheck trading-agent.log for completed details.`;
                    
                    if (state.status !== 'running') {
                        clearInterval(runPoller);
                        document.getElementById('trigger-status').innerText = `LUNA cycle ${state.status}.`;
                        fetchData(); // reload
                    }
                } catch(e) {}
            }, 3000);
        }

        async function pollRunStatus() {
            try {
                let res = await fetch('/api/run-status');
                let state = await res.json();
                const dot = document.querySelector('.status-dot');
                const label = document.querySelector('.sidebar-footer .system-status span');
                
                if (state.status === 'running') {
                    dot.style.background = 'var(--accent-orange)';
                    dot.style.boxShadow = '0 0 10px var(--accent-orange)';
                    label.innerText = 'LUNA Running Pass';
                } else {
                    dot.style.background = 'var(--accent-green)';
                    dot.style.boxShadow = '0 0 10px var(--accent-green)';
                    label.innerText = 'Autonomous Live';
                }
            } catch(e) {}
        }
    </script>
</body>
</html>
"""

# Server instantiation Orchestrator
def start_dashboard_server(port: int = 8080):
    """Start local HTTP server and launch browser."""
    logger.info("Initializing LUNA web dashboard server...")
    
    # Try different ports if 8080 is blocked
    active_port = port
    server = None
    
    for attempt in range(5):
        try:
            handler = DashboardHTTPRequestHandler
            server = socketserver.TCPServer(("", active_port), handler)
            break
        except OSError:
            logger.warning("Port %d is already in use. Retrying on port %d...", active_port, active_port + 1)
            active_port += 1

    if server is None:
        logger.error("Could not locate a free port. Dashboard aborting.")
        return

    logger.info("LUNA Dashboard Server successfully started on http://localhost:%d", active_port)
    logger.info("Serving interactive glassmorphic UI Zero-Dependency...")
    
    # Dynamically launch standard web browser
    try:
        webbrowser.open(f"http://localhost:{active_port}")
    except Exception as e:
        logger.warning("Failed to automatically launch web browser: %s", e)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Dashboard web server shutting down gracefully.")
        server.server_close()

if __name__ == "__main__":
    start_dashboard_server()
