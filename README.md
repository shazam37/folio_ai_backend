Portfolio Optimizer GenAI (LangGraph + FastAPI)
A hackathon-ready backend that analyzes an investor’s portfolio and suggests a practical, constraint-aware rebalancing plan using a hybrid of deterministic optimization and AI reasoning. Built with FastAPI and LangGraph to keep the workflow transparent, auditable, and easy to integrate with a Next.js frontend.

This project is designed so that non-finance folks can understand what’s happening at every step, and developers can extend or swap components as needed.

What this app does
Accepts a user’s current portfolio (tickers and quantities), cash, and preferences (risk tolerance, time horizon, constraints).

Agent 1 (Researcher):

Gathers basic company info and qualitative trend signals from the web/data tools.

Builds a clear, human-readable analysis of the current portfolio (sector exposure, concentration, rough risk proxy).

Agent 2 (Optimizer):

Checks the analysis, applies deterministic rules (caps, diversification, optional cash buffer), and proposes a rebalancing plan.

Uses AI to write a concise rationale in plain English without inventing numbers.

Outputs include:

Before/after portfolio metrics (e.g., concentration index)

Target weights, suggested buy/sell/hold actions, and estimated trade values

Explanations in simple language so users understand the “why,” not just the “what”

Why this matters (in plain English)
Diversification reduces the chance that one stock can sink the whole portfolio.

Position caps (e.g., 20% per stock) avoid overconfidence in a single name.

Clear actions and estimates help turn analysis into decisions.

Explanations avoid “black box” results, helping build trust.

This app doesn’t promise market-beating results; it aims to create a safer, more aligned portfolio based on the user’s preferences and constraints.

Architecture overview
FastAPI backend exposing simple endpoints:

POST /optimize/run → start a run

GET /optimize/status/{run_id} → check status

GET /optimize/result/{run_id} → fetch results

LangGraph orchestrates the workflow:

validate_input → fetch_market_snapshot → analyze_portfolio → optimize_portfolio → finalize

Tools gather company info, prices, and qualitative trend flags. During hackathon, these use local stubs to avoid brittle dependencies; you can easily swap in real providers later.

Deterministic optimizer enforces:

Single-name caps (e.g., max 20% per stock)

Optional sector caps (e.g., Tech ≤ 35%)

Minimum position sizes to avoid tiny, noisy positions

Optional cap on number of holdings

Cash bucket used when caps force unallocated weight (keeps total at 100%)

Observability:

Each run captures step-by-step logs and serialized outputs to make reviews and debugging easy.

Key design choices
Deterministic first, AI second: Numbers are produced by code; AI explains and critiques but never invents figures.

Transparent state: Inputs, intermediate analyses, and outputs are all returned in structured JSON.

Simple to extend: Swap stub data tools for real APIs; replace the basic optimizer with a fancier one without changing the overall flow.

Hackathon-friendly: Local storage (in-memory), minimal configuration, straightforward API for a Next.js frontend.

Data models (simplified)
PortfolioInput

positions: list of {ticker, quantity, avg_cost?}

cash

preferences: {risk_tolerance: low/medium/high, horizon_months, single_name_cap, target_return?, sector_caps?}

constraints: {max_positions?, min_position_size?, rebalance_freq?, exclude_tickers?}

user_profile: {country?, currency?, allow_derivatives?}

AnalysisBundle

portfolio_metrics: total_value, weights, sector_weights, concentration_index (HHI), estimated beta proxy

company_digests: short summaries per holding

market_context: neutral, qualitative summary

issues_found: e.g., high concentration, above-cap positions

OptimizationPlan

target_weights: final allocation per ticker (and CASH)

actions: buy/sell/hold with estimated trade values

rationale: AI-written explanation grounded in the actual plan

risk_impact: before/after concentration and proxies

alternatives: conservative and growth-tilt variants

Project structure (suggested)
app/

graph.py → LangGraph definition (agents, tools, nodes, compiled graph)

api.py → FastAPI app (endpoints only; imports compiled graph)

README.md → this file

You can rename or reorganize as needed; ensure the FastAPI module can import the compiled graph.

Setup
Requirements:

Python 3.10+

An API key for your LLM provider (e.g., OpenAI) stored as an environment variable (OPENAI_API_KEY)

Install dependencies:

pip install fastapi uvicorn langgraph langchain langchain-openai pydantic

Environment:

export OPENAI_API_KEY=your_key_here

Start the server:

uvicorn app.api:app --reload

If module paths differ, adjust the uvicorn path accordingly.

API usage
Example JSON payload:
{
"positions": [
{"ticker": "AAPL", "quantity": 10},
{"ticker": "MSFT", "quantity": 8}
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

Endpoints:

Start a run:

POST /optimize/run with the payload above

Returns: {"run_id": "...", "state": "completed" | "running" | "failed"}

Check status:

GET /optimize/status/{run_id}

Get results:

GET /optimize/result/{run_id}

Returns analysis_bundle, optimization_plan, logs, and version

Testing options:

Built-in Swagger UI at /docs

curl:

curl -X POST http://127.0.0.1:8000/optimize/run -H "Content-Type: application/json" -d '{...}'

HTTPie:

http POST :8000/optimize/run positions:='[{"ticker":"AAPL","quantity":10},{"ticker":"MSFT","quantity":8}]' cash:=500 preferences:='{"risk_tolerance":"medium","horizon_months":24,"single_name_cap":0.2}' constraints:='{"max_positions":25,"min_position_size":0.01}' user_profile:='{"currency":"USD","allow_derivatives":false}'

Implementation details
Input validation:

Enforces at least one position

Checks time horizon positive

Flags excluded tickers if present

Data collection (dev mode):

Prices default to a neutral stub value

Fundamentals and trend flags are placeholders; swap with real providers when ready

Analysis:

Calculates portfolio value and weights

Sector weights based on tool data

Concentration via HHI (sum of squared weights; lower is more diversified)

Basic risk proxy via weighted betas if available

Optimization:

Enforces single-name cap across holdings

Optional sector caps (if provided)

Minimum position size filter

Cap on number of positions (keeps turnover reasonable)

Cash bucket absorbs leftover weight when caps force down allocations

Buy/sell/hold actions include estimated trade values for clarity

Alternatives provide quick “what-if” choices

Explanations:

AI rewrites and tightens rationale using the actual plan and issues found

No invented numbers; qualitative guidance only

Extending the app
Replace stub tools with real data:

Market data: Polygon.io, Alpha Vantage, Finnhub, Yahoo Finance

News: NewsAPI or financial-news sources

Fundamentals: providers with sector, beta, and valuation fields

Improve the optimizer:

Add historical returns and compute covariance (e.g., numpy/pandas)

Use a quadratic program (e.g., cvxpy) for mean-variance with constraints

Penalize turnover and include transaction cost estimates

Add a tax-aware mode to avoid selling high-gain positions where possible

Compliance/risk filters:

Exclude penny stocks, very low liquidity names, or restricted tickers

Add ESG or custom filters as user preferences

Scalability and persistence:

Swap in-memory run registry for SQLite/Postgres

Persist runs, results, and logs for user histories

Use background tasks or a queue for long-running jobs

Add tracing/metrics for observability

Frontend integration (Next.js):

Submit POST /optimize/run from an API route or server action

Poll GET /optimize/status/{run_id} until completed

Render GET /optimize/result/{run_id}: metrics, plan, and rationale

Visualize before/after weights and show actions with trade values

Limitations
Not investment advice. This is a technical demonstration for educational purposes.

Stubbed data during development means numbers are placeholders; connect real data sources for meaningful results.

Risk proxies are simplified (e.g., HHI, basic beta). Real-world risk needs a proper covariance model and scenario analysis.

Taxes, fees, and user-specific constraints can materially change optimal actions and are simplified here.

FAQ
Why does the plan sometimes recommend “CASH”?

When caps prevent allocating 100% to current holdings, the remainder is left as cash. This is safer than forcing over-concentration. You can later allocate that cash to diversified funds or new names.

I set a single-name cap of 20%. Why did a stock show more than 20%?

That’s a bug the optimizer guards against. Ensure caps are enforced before normalization, and any excess is redistributed or moved to CASH. The provided code includes this logic.

Can I add my own constraints (e.g., “no fossil fuels”, “max 10% small-cap”)?

Yes. Add checks in the analysis and constraints in the optimizer, then surface any infeasibility clearly in the response.

Contributing
Fork and open a PR with a clear description of what you changed and why.

Keep the flow transparent: avoid burying important logic in prompts or opaque functions.

Add logs for new steps and preserve structured outputs for the UI.
