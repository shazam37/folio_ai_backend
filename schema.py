from typing import Annotated

from typing_extensions import TypedDict

from typing import List, Dict, Optional, TypedDict, Literal
from datetime import datetime
from pydantic import BaseModel, Field

class Position(BaseModel):
    ticker: str
    quantity: float
    avg_cost: Optional[float] = None

class Preferences(BaseModel):
    risk_tolerance: Literal["low", "medium", "high"]
    horizon_months: int
    target_return: Optional[float] = None
    sector_caps: Optional[Dict[str, float]] = None   # e.g., {"Technology": 0.35}
    single_name_cap: Optional[float] = Field(default=0.10, description="Max weight per ticker")
    tax_sensitivity: Optional[Literal["low","medium","high"]] = "medium"

class Constraints(BaseModel):
    max_positions: Optional[int] = 25
    min_position_size: Optional[float] = 0.01
    rebalance_freq: Optional[Literal["quarterly","semiannual","annual"]] = "quarterly"
    exclude_tickers: Optional[List[str]] = None

class UserProfile(BaseModel):
    country: Optional[str] = None
    currency: Optional[str] = "USD"
    allow_derivatives: Optional[bool] = False

class PortfolioInput(BaseModel):
    id: str
    positions: List[Position] = Field(..., min_items=1)
    cash: float = 0.0
    preferences: Preferences
    constraints: Optional[Constraints] = Constraints()
    user_profile: Optional[UserProfile] = UserProfile()

# Tool-level market snapshot structs (kept simple)
class CompanySnap(BaseModel):
    ticker: str
    price: Optional[float] = None
    beta: Optional[float] = None
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    pe: Optional[float] = None
    dividend_yield: Optional[float] = None
    news_summary: Optional[str] = None
    trend_flags: Optional[List[str]] = None
    risk_flags: Optional[List[str]] = None

class MarketSnapshot(BaseModel):
    as_of: str
    companies: Dict[str, CompanySnap]

class PortfolioMetrics(BaseModel):
    total_value: float
    weights: Dict[str, float]
    sector_weights: Dict[str, float]
    concentration_index: float
    est_portfolio_beta: Optional[float] = None

class AnalysisBundle(BaseModel):
    portfolio_metrics: PortfolioMetrics
    company_digests: Dict[str, str]
    market_context: str
    issues_found: List[str]

class Action(BaseModel):
    ticker: str
    action: Literal["buy","sell","hold"]
    target_weight: float
    est_trade_value: float

class OptimizationPlan(BaseModel):
    target_weights: Dict[str, float]
    actions: List[Action]
    rationale: str
    risk_impact: Dict[str, float]
    alternatives: Optional[Dict[str, Dict[str, float]]] = None

# LangGraph State
class GraphState(TypedDict):
    input: PortfolioInput
    market_snapshot: Optional[MarketSnapshot]
    analysis_bundle: Optional[AnalysisBundle]
    optimization_plan: Optional[OptimizationPlan]
    logs: List[str]
    errors: List[str]