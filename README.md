# Folio AI: AI Portfolio Optimizer

A hackathon-ready backend that analyzes an investor’s portfolio and proposes a practical, constraint-aware rebalancing plan using a combination of deterministic optimization and AI reasoning. Built with FastAPI and LangGraph to keep the workflow transparent, auditable, and easy to integrate with a Next.js frontend.

This project is designed so that non-finance folks can understand what’s happening at every step, and developers can extend or swap components as needed.

***

## Table of Contents

- Introduction
- Features
- How It Works (High-Level)
- Architecture
- Data Models
- Project Structure
- Setup and Run
- API Usage
- Example Payload
- Example Output (What to Expect)
- Implementation Details
- Extending the App
- Limitations
- FAQ
- Contributing
- License

***

## Introduction

Portfolio Optimizer GenAI helps turn a user’s current investments and preferences into a clearer, more diversified allocation with simple explanations. It does not try to “beat the market.” Instead, it focuses on:
- Avoiding over-concentration
- Respecting user constraints (e.g., single stock cap)
- Producing actionable buy/sell/hold suggestions
- Explaining the reasoning in plain English

***

## Features

- Accepts a user’s current portfolio (tickers and quantities), cash, and preferences (risk tolerance, time horizon, constraints)
- Agent 1 (Researcher): gathers basic company context and qualitative trend signals from data tools, then summarizes the current portfolio
- Agent 2 (Optimizer): checks the analysis, applies deterministic rules (caps, diversification, optional cash bucket), and proposes a rebalancing plan
- AI-generated rationale: concise, human-readable explanation without inventing numbers
- Returns before/after metrics, target weights, buy/sell/hold actions, estimated trade values, and logs

***

## How It Works (High-Level)

1) Input: Positions, cash, and preferences (risk tolerance, time horizon, constraints)
2) Research: Collects company info (sector, proxy risk), and qualitative trend summaries
3) Analysis: Computes portfolio metrics (value, weights, sector mix, concentration)
4) Optimization: Enforces constraints (e.g., single-name cap), creates target weights, uses CASH for leftover allocation when needed
5) Explanation: AI produces a short rationale grounded in the actual numbers and constraints
6) Output: Metrics, plan, actions, and logs presented in structured JSON

***

## Architecture

- Backend: FastAPI
- Workflow Orchestration: LangGraph
- Endpoints:
  - POST /optimize/run → start a run
  - GET /optimize/status/{run_id} → check status
  - GET /optimize/result/{run_id} → fetch results
- Tools: Swappable data sources (stubbed locally for hackathon); easy to replace with real market/news/fundamentals providers
- Deterministic Optimizer:
  - Single-name caps (e.g., max 20% per stock)
  - Optional sector caps (e.g., Tech ≤ 35%)
  - Minimum position size (avoid tiny allocations)
  - Optional max number of positions
  - CASH bucket for leftover weight (keeps allocations at 100%)
- Observability: Step-by-step logs and structured outputs for easy review

***

## Data Models

PortfolioInput
- positions: list of {ticker, quantity, avg_cost?}
- cash: number
- preferences:
  - risk_tolerance: low | medium | high
  - horizon_months: integer
  - single_name_cap: number (e.g., 0.20 for 20%)
  - target_return?: number
  - sector_caps?: { sector_name: cap_fraction }
- constraints:
  - max_positions?: integer
  - min_position_size?: number (e.g., 0.01 for 1%)
  - rebalance_freq?: quarterly | semiannual | annual
  - exclude_tickers?: [string]
- user_profile:
  - country?: string
  - currency?: string (default USD)
  - allow_derivatives?: boolean

AnalysisBundle
- portfolio_metrics:
  - total_value: number
  - weights: { ticker: fraction }
  - sector_weights: { sector: fraction }
  - concentration_index: number (Herfindahl-Hirschman Index; lower is more diversified)
  - est_portfolio_beta?: number (simple proxy)
- company_digests: { ticker: short_summary }
- market_context: short qualitative summary
- issues_found: [string]

OptimizationPlan
- target_weights: { ticker_or_CASH: fraction }
- actions: [{ ticker, action: buy|sell|hold, target_weight, est_trade_value }]
- rationale: string (concise explanation grounded in the actual plan)
- risk_impact: { hhi_before, hhi_after, beta_before, beta_after, ... }
- alternatives: { conservative: {...}, growth_tilt: {...} }

***

## Project Structure

- app/
  - graph.py → LangGraph definition (agents, tools, nodes, compiled graph)
  - api.py → FastAPI application (endpoints only; imports compiled graph)
- README.md → this file

Feel free to reorganize; ensure the FastAPI module can import the compiled graph.

***

## Setup and Run

Requirements
- Python 3.10+
- LLM provider key set as an environment variable (e.g., OPENAI_API_KEY)

Install
- pip install fastapi uvicorn langgraph langchain langchain-openai pydantic

Environment
- export OPENAI_API_KEY=your_key_here

Start the server
- uvicorn app.api:app --reload

If your module path differs, update the uvicorn path accordingly.

***

## API Usage

Start a run
- POST /optimize/run with a JSON payload (see “Example Payload”)
- Response: {"run_id": "...", "state": "completed" | "running" | "failed"}

Check status
- GET /optimize/status/{run_id}

Fetch results
- GET /optimize/result/{run_id}
- Returns: analysis_bundle, optimization_plan, logs, version

Testing options
- Swagger UI: visit /docs in your browser after starting the server
- curl:
  - curl -X POST http://127.0.0.1:8000/optimize/run -H "Content-Type: application/json" -d '{...}'
- HTTPie:
  - http POST :8000/optimize/run positions:='[{"ticker":"AAPL","quantity":10},{"ticker":"MSFT","quantity":8}]' cash:=500 preferences:='{"risk_tolerance":"medium","horizon_months":24,"single_name_cap":0.2}' constraints:='{"max_positions":25,"min_position_size":0.01}' user_profile:='{"currency":"USD","allow_derivatives":false}'

***

## Example Payload

{
  "positions": [
    { "ticker": "AAPL", "quantity": 10 },
    { "ticker": "MSFT", "quantity": 8 }
  ],
  "cash": 500.0,
  "preferences": {
    "risk_tolerance": "medium",
    "horizon_months": 24,
    "single_name_cap": 0.2
  },
  "constraints": {
    "max_positions": 25,
    "min_position_size": 0.01
  },
  "user_profile": {
    "currency": "USD",
    "allow_derivatives": false
  }
}

***

## Example Output (What to Expect)

- A structured JSON including:
  - analysis_bundle:
    - portfolio_metrics: total_value, weights, sector_weights, concentration_index, est_portfolio_beta
    - company_digests: short summaries per ticker
    - market_context: concise qualitative overview
    - issues_found: e.g., “High concentration”
  - optimization_plan:
    - target_weights: allocation for each ticker, plus CASH if unallocated
    - actions: buy/sell/hold with estimated trade values
    - rationale: plain-English explanation aligned with actual numbers
    - risk_impact: before/after indicators (e.g., HHI)
    - alternatives: conservative and growth-tilt variants
  - logs: step-by-step trace of the run
  - version: backend version

Note: In development mode, market data/fundamentals/news may be stubbed. Replace with real providers for meaningful results.

***

## Implementation Details

- Input validation:
  - Ensures at least one position
  - Verifies positive investment horizon
  - Flags excluded tickers if present

- Data collection (development):
  - Prices/fundamentals/trends use stubbed values to keep the system stable
  - Easily swap in real providers later

- Analysis:
  - Computes portfolio value and weights
  - Builds sector weights from tool data
  - Calculates concentration via HHI (sum of squared weights, excluding CASH)
  - Estimates a simple beta proxy when available

- Optimization:
  - Enforces single-name caps
  - Supports optional sector caps
  - Applies minimum position size to avoid tiny allocations
  - Caps number of positions to limit turnover
  - Uses a CASH bucket to keep total allocation at 100% when caps reduce exposure
  - Generates buy/sell/hold actions with estimated trade values
  - Produces alternatives (conservative, growth-tilt) for quick what-if choices

- Explanations:
  - AI generates a short rationale based on the actual output (no invented numbers)
  - Language is concise and user-friendly

***

## Extending the App

Replace stub tools with real data
- Market data: Polygon.io, Alpha Vantage, Finnhub, Yahoo Finance
- News: NewsAPI or financial-news providers
- Fundamentals: sector, beta, valuation fields from a fundamentals API

Improve the optimizer
- Pull historical returns and estimate covariance (e.g., numpy/pandas)
- Solve constrained mean-variance with a quadratic program (e.g., cvxpy)
- Penalize turnover and incorporate transaction cost estimates
- Add tax-aware logic to avoid unnecessary realization of gains

Compliance and risk filters
- Exclude low-liquidity or penny stocks
- Allow user-defined exclusions (e.g., ESG screens)
- Add simple rule checks (e.g., sanctions lists) as a safety filter

Scale and persist
- Replace in-memory run registry with SQLite/Postgres
- Store runs/results/logs for user history
- Add background tasks/queues for long-running jobs
- Integrate tracing/metrics for observability

Frontend integration (Next.js)
- Submit POST /optimize/run from an API route or server action
- Poll GET /optimize/status/{run_id} until completed
- Render GET /optimize/result/{run_id}: show metrics, plan, rationale
- Visualize before/after weights and actions with estimated trade values

***

## Limitations

- Not investment advice; intended for educational and technical demonstration
- Development mode uses stubbed data; connect real data sources for production
- Risk proxies are simplified; real portfolios need robust covariance models and scenario analysis
- Taxes, fees, and user-specific constraints are simplified and can materially affect outcomes

***

## FAQ

- Why is CASH included in target weights?
  - When caps prevent allocating 100% to current holdings, the remainder is placed in CASH to avoid forced over-concentration. Users can later allocate CASH to diversified funds or additional names.

- I set a single-name cap of 20%. Why would a stock exceed that?
  - The optimizer enforces caps before normalization; if you see otherwise, ensure you’re on the latest version and that your inputs were parsed correctly.

- Can I add custom rules (e.g., “no fossil fuels”, “cap small-cap exposure”)?
  - Yes. Add checks in the analysis and constraints in the optimizer, and clearly surface infeasible constraints in the response.

***

## Contributing

- Fork the repository and open a PR with a clear description
- Keep the workflow transparent: avoid hiding logic in long prompts or opaque functions
- Add structured logs for new steps and preserve output formats for the frontend

***

## License

MIT License

This software is provided “as is,” without warranty of any kind. It is not financial advice and should not be used as the sole basis for investment decisions.
