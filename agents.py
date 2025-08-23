import os
from typing import Annotated
import json

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from typing import List, Dict, Optional, TypedDict, Literal
from datetime import datetime
from pydantic import BaseModel, Field, conlist
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
from schema import *

google_api_key = "AIzaSyCt--XXbfRUxWzgiEt56_2mQY3NjoPl-BU"
tavily_api_key = "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"

os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

llm = init_chat_model("google_genai:gemini-2.5-flash")

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# tool = TavilySearch(max_results=2)
# tools = [tool]
# llm_with_tools = llm.bind_tools(tools)

# def chatbot(state: State):
#     return {"messages": [llm_with_tools.invoke(state["messages"])]}

# graph_builder.add_node("chatbot", chatbot)

# tool_node = ToolNode(tools=[tool])
# graph_builder.add_node("tools", tool_node)

# graph_builder.add_conditional_edges(
#     "chatbot",
#     tools_condition,
# )
# graph_builder.add_edge("tools", "chatbot")
# graph_builder.set_entry_point("chatbot")
# memory = InMemorySaver()
# graph = graph_builder.compile(checkpointer=memory)

# config = {"configurable": {"thread_id": "1"}}

# def stream_graph_updates(user_input: str): 
#     events = graph.stream(
#         {"messages": [{"role": "user", "content": user_input}]},
#         config,
#         stream_mode="values",
#     )
#     for event in events:
#         event["messages"][-1].pretty_print()

# if __name__ == "__main__":

#     while True:
#         try:
#             user_input = input("User: ")
#             if user_input.lower() in ["quit", "exit", "q"]:
#                 print("Goodbye!")
#                 break

#             stream_graph_updates(user_input)
#         except:
#             # fallback if input() is not available
#             user_input = "What do you know about LangGraph?"
#             print("User: " + user_input)
#             stream_graph_updates(user_input)
#             break

# In dev, we avoid DB for brevity; store runs in memory dict
RUNS: Dict[str, dict] = {}

# ---------- Schemas ----------



# ---------- LLM ----------
# OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)

# ---------- Tools (dev-safe stubs with clear seams) ----------

@tool
def fetch_prices(tickers: List[str]) -> Dict[str, float]:
    """Return latest prices for tickers. DEV: stub with generated prices; replace with real API later."""
    # In hackathon, avoid fake precision: default price = 100.0 if unknown
    return {t: 100.0 for t in tickers}

@tool
def fetch_fundamentals(tickers: List[str]) -> Dict[str, dict]:
    """Return basic fundamentals like beta, sector, market_cap. DEV: stub predictable values."""
    out = {}
    for t in tickers:
        out[t] = {"beta": 1.0, "sector": "Technology" if t[0] < "M" else "Financials",
                  "market_cap": 10e9, "pe": 20.0, "dividend_yield": 0.01}
    return out

@tool
def fetch_news_and_trends(tickers: List[str]) -> Dict[str, dict]:
    """Return concise news/trend summaries. DEV: synthetic neutral summaries."""
    return {t: {"summary": f"No major news for {t}.", "trend_flags": ["neutral"], "risk_flags": []} for t in tickers}

# ---------- Helper functions (deterministic) ----------

def log(state: GraphState, msg: str):
    state["logs"].append(f"{datetime.utcnow().isoformat()}Z | {msg}")

def compute_portfolio_metrics(p: PortfolioInput, snap: MarketSnapshot) -> PortfolioMetrics:
    # Compute total value and weights
    values = {}
    for pos in p.positions:
        price = snap.companies[pos.ticker].price or 0.0
        values[pos.ticker] = price * pos.quantity
    total_value = sum(values.values()) + p.cash
    weights = {t: (v / total_value if total_value > 0 else 0.0) for t, v in values.items()}
    # Sector weights
    sector_weights: Dict[str, float] = {}
    for pos in p.positions:
        sector = snap.companies[pos.ticker].sector or "Unknown"
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weights.get(pos.ticker, 0.0)
    # Concentration: Herfindahl-Hirschman Index
    hhi = sum(w*w for w in weights.values())
    # Simple beta aggregation
    betas = []
    for pos in p.positions:
        beta = snap.companies[pos.ticker].beta
        w = weights.get(pos.ticker, 0.0)
        if beta is not None:
            betas.append(beta * w)
    est_beta = sum(betas) if betas else None
    return PortfolioMetrics(
        total_value=total_value,
        weights=weights,
        sector_weights=sector_weights,
        concentration_index=hhi,
        est_portfolio_beta=est_beta
    )

def deterministic_optimizer(p: PortfolioInput, metrics: PortfolioMetrics, snap: MarketSnapshot) -> OptimizationPlan:
    # Strategy: risk-aware cap + sector caps + single-name cap, keep turnover modest
    single_cap = p.preferences.single_name_cap or 0.10
    sector_caps = p.preferences.sector_caps or {}
    current_w = metrics.weights.copy()

    # Start with current weights; nudge toward caps and diversify
    target = current_w.copy()

    # Enforce single-name cap
    for t, w in list(target.items()):
        if w > single_cap:
            target[t] = single_cap

    # Normalize after single-name caps
    s = sum(target.values())
    if s > 0:
        target = {t: w/s for t, w in target.items()}

    # Enforce sector caps by proportionally scaling down capped sectors
    sector_w = {}
    ticker_sector = {t: (snap.companies[t].sector or "Unknown") for t in target.keys()}
    for t, w in target.items():
        sec = ticker_sector[t]
        sector_w[sec] = sector_w.get(sec, 0.0) + w

    for sec, cap in sector_caps.items():
        if sector_w.get(sec, 0.0) > cap:
            scale = cap / sector_w[sec]
            for t in list(target.keys()):
                if ticker_sector[t] == sec:
                    target[t] *= scale
            # redistribute shaved weight proportionally to other sectors
            shaved_total = 1.0 - sum(target.values())
            if shaved_total > 0:
                for t in list(target.keys()):
                    if ticker_sector[t] != sec:
                        target[t] += shaved_total * (1.0 / (len(target)-len([x for x in target if ticker_sector[x]==sec])))

    # Optional: reduce concentration if HHI too high
    if metrics.concentration_index > 0.12:
        # push largest weights down by 10%
        sorted_t = sorted(target.items(), key=lambda x: x, reverse=True)
        for i, (t, w) in enumerate(sorted_t[:max(1, len(sorted_t)//4)]):
            target[t] = max(w * 0.9, min(0.5*single_cap, w))
        # renormalize
        s = sum(target.values())
        target = {t: w/s for t, w in target.items()}

    # Clip tiny positions below min_position_size
    min_size = (p.constraints.min_position_size or 0.01)
    for t, w in list(target.items()):
        if w < min_size:
            target[t] = 0.0
    # renormalize non-zero
    s = sum(target.values())
    if s > 0:
        target = {t: (w/s if w>0 else 0.0) for t, w in target.items()}

    # Cap number of positions
    max_pos = p.constraints.max_positions or len(target)
    if len([t for t,w in target.items() if w>0]) > max_pos:
        # keep top-k by current weight to reduce turnover
        top = sorted(target.items(), key=lambda x: metrics.weights.get(x,0), reverse=True)[:max_pos]
        keep = set([t for t,_ in top])
        target = {t: (w if t in keep else 0.0) for t,w in target.items()}
        s = sum(target.values())
        if s > 0:
            target = {t: (w/s if w>0 else 0.0) for t, w in target.items()}

    # Build actions
    actions: List[Action] = []
    for t in target.keys():
        tw = target[t]
        cw = metrics.weights.get(t, 0.0)
        if abs(tw - cw) < 0.005:
            act = "hold"
        elif tw > cw:
            act = "buy"
        else:
            act = "sell"
        est_trade_value = (tw - cw) * metrics.total_value
        actions.append(Action(ticker=t, action=act, target_weight=tw, est_trade_value=est_trade_value))

    # Risk impact (coarse)
    risk_impact = {
        "hhi_before": metrics.concentration_index,
        "hhi_after": sum([w*w for w in target.values()]),
        "beta_before": metrics.est_portfolio_beta or 1.0,
        "beta_after": metrics.est_portfolio_beta or 1.0  # unchanged without factor model
    }

    # Alternatives
    alt_conservative = {t: w*0.95 for t, w in target.items()}
    s = sum(alt_conservative.values())
    if s>0: alt_conservative = {t: w/s for t, w in alt_conservative.items()}
    alt_growth = {t: min(w*1.05, single_cap) for t, w in target.items()}
    s = sum(alt_growth.values())
    if s>0: alt_growth = {t: w/s for t, w in alt_growth.items()}

    return OptimizationPlan(
        target_weights=target,
        actions=actions,
        rationale="Targets derived via deterministic constraints (single-name and sector caps) and concentration reduction. Fine-tuned by risk tolerance.",
        risk_impact=risk_impact,
        alternatives={"conservative": alt_conservative, "growth_tilt": alt_growth},
    )

# ---------- Nodes ----------

def validate_input_node(state: GraphState) -> GraphState:
    p = state["input"]
    log(state, "Validating input")
    tickers = [pos.ticker for pos in p.positions]
    if p.constraints and p.constraints.exclude_tickers:
        for t in p.constraints.exclude_tickers:
            if t in tickers:
                state["errors"].append(f"Excluded ticker present: {t}")
    if p.preferences.horizon_months <= 0:
        state["errors"].append("Horizon must be positive months.")
    return state

def fetch_market_snapshot_node(state: GraphState) -> GraphState:
    p = state["input"]
    tickers = [pos.ticker for pos in p.positions]
    log(state, f"Fetching market snapshot for {tickers}")
    prices = fetch_prices.run({"tickers": tickers})  # using tool
    fundamentals = fetch_fundamentals.run({"tickers": tickers})
    news = fetch_news_and_trends.run({"tickers": tickers})

    companies = {}
    for t in tickers:
        comp = CompanySnap(
            ticker=t,
            price=prices.get(t),
            beta=(fundamentals.get(t) or {}).get("beta"),
            sector=(fundamentals.get(t) or {}).get("sector"),
            market_cap=(fundamentals.get(t) or {}).get("market_cap"),
            pe=(fundamentals.get(t) or {}).get("pe"),
            dividend_yield=(fundamentals.get(t) or {}).get("dividend_yield"),
            news_summary=(news.get(t) or {}).get("summary"),
            trend_flags=(news.get(t) or {}).get("trend_flags"),
            risk_flags=(news.get(t) or {}).get("risk_flags"),
        )
        companies[t] = comp
    snapshot = MarketSnapshot(as_of=datetime.utcnow().isoformat()+"Z", companies=companies)
    state["market_snapshot"] = snapshot
    log(state, "Market snapshot complete")
    return state

def analyze_portfolio_node(state: GraphState) -> GraphState:
    p = state["input"]
    snap = state["market_snapshot"]
    assert snap is not None
    log(state, "Analyzing portfolio")
    metrics = compute_portfolio_metrics(p, snap)

    # Use LLM to summarize company digests + market context based on tools' outputs
    digests = {}
    for t, info in snap.companies.items():
        digest = f"{t}: sector={info.sector}, beta={info.beta}, pe={info.pe}, news='{info.news_summary}', trends={info.trend_flags}"
        digests[t] = digest

    # Market context via LLM (but strictly qualitative; no invented numbers)
    context_prompt = f"""
You are an investment analyst. Based only on qualitative cues (sectors, betas, trend flags), summarize market context
for the given portfolio. Do not invent numbers. Portfolio sector weights: {json.dumps(metrics.sector_weights)}.
Trend flags per ticker: { {t: info.trend_flags for t, info in snap.companies.items()} }.
Risk tolerance: {p.preferences.risk_tolerance}, horizon: {p.preferences.horizon_months} months.
Return a 3-5 sentence neutral summary.
"""
    market_context = llm.invoke(context_prompt).content

    issues = []
    if metrics.concentration_index > 0.12:
        issues.append("High concentration (HHI > 0.12).")
    if any((snap.companies[t].risk_flags or []) for t in snap.companies):
        issues.append("Some holdings have risk flags in recent news.")
    if p.preferences.single_name_cap and any(w > p.preferences.single_name_cap for w in metrics.weights.values()):
        issues.append("Single-name weight above cap.")
    analysis = AnalysisBundle(
        portfolio_metrics=metrics,
        company_digests=digests,
        market_context=market_context,
        issues_found=issues
    )
    state["analysis_bundle"] = analysis
    log(state, "Analysis complete")
    return state

def optimize_portfolio_node(state: GraphState) -> GraphState:
    p = state["input"]
    analysis = state["analysis_bundle"]
    snap = state["market_snapshot"]
    assert analysis and snap
    log(state, "Optimizing portfolio with constraints")

    plan = deterministic_optimizer(p, analysis.portfolio_metrics, snap)

    # LLM to check and enrich rationale (no new numbers)
    critique_prompt = f"""
You are a portfolio manager. Review this plan and provide a concise rationale (5-7 sentences), referencing:
- risk tolerance ({p.preferences.risk_tolerance}),
- horizon ({p.preferences.horizon_months} months),
- issues found ({analysis.issues_found}),
- market context: {analysis.market_context}.
Do not add numbers not present. Plan: {json.dumps(plan.model_dump())}
"""
    rationale = llm.invoke(critique_prompt).content
    plan.rationale = rationale
    state["optimization_plan"] = plan
    log(state, "Optimization complete")
    return state

def finalize_node(state: GraphState) -> GraphState:
    log(state, "Finalizing")
    return state

# ---------- Graph compile ----------

builder = StateGraph(GraphState)
builder.add_node("validate_input", validate_input_node)
builder.add_node("fetch_market_snapshot", fetch_market_snapshot_node)
builder.add_node("analyze_portfolio", analyze_portfolio_node)
builder.add_node("optimize_portfolio", optimize_portfolio_node)
builder.add_node("finalize", finalize_node)

# Edges
builder.set_entry_point("validate_input")
builder.add_edge("validate_input", "fetch_market_snapshot")
builder.add_edge("fetch_market_snapshot", "analyze_portfolio")
builder.add_edge("analyze_portfolio", "optimize_portfolio")
builder.add_edge("optimize_portfolio", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile(checkpointer=InMemorySaver())