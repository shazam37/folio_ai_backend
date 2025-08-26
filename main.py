# app/api.py
import os
import uuid
from datetime import datetime
from typing import Dict, Optional, List, Literal, TypedDict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist

# Import the compiled LangGraph from your LangGraph module
# Ensure your graph module defines a compiled `graph` object that accepts the state
# and returns the state with analysis and optimization results.
# Example: from app.graph import graph
from agents import graph  # <-- update path if needed
from schema import *



# ---------------- In-memory run registry (for hackathon/dev) ----------------

RUNS: Dict[str, dict] = {}


# ---------------- FastAPI App ----------------

app = FastAPI(
    title="Portfolio Optimizer GenAI API",
    version="0.1.0",
    description="FastAPI backend for a LangGraph-powered portfolio optimization assistant."
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] during dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/optimize/run")
def run_optimize(payload: PortfolioInput):
    """
    Kick off an optimization run. Returns a run_id that the frontend can poll.
    """
    run_id = payload.id
    initial_state: GraphState = {
        "input": payload,
        "market_snapshot": None,
        "analysis_bundle": None,
        "optimization_plan": None,
        "logs": [],
        "errors": [],
    }

    RUNS[run_id] = {
        "state": "running",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": None,
        "errors": [],
        "result": None,
        "logs": [],
    }

    try:
        # Threaded runs or background tasks can be added later; synchronous for hackathon simplicity.
        result = graph.invoke(initial_state, config={"configurable": {"thread_id": run_id}})
        RUNS[run_id]["state"] = "completed"
        RUNS[run_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"
        RUNS[run_id]["result"] = {
            "analysis_bundle": result.get("analysis_bundle"),
            "optimization_plan": result.get("optimization_plan"),
        }
        RUNS[run_id]["logs"] = result.get("logs", [])
        if result.get("errors"):
            RUNS[run_id]["errors"] = result["errors"]
    except Exception as e:
        RUNS[run_id]["state"] = "failed"
        RUNS[run_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"
        RUNS[run_id]["errors"] = [str(e)]
        raise HTTPException(status_code=500, detail=f"Run failed: {e}")

    return {"run_id": run_id, "state": RUNS[run_id]["state"]}


@app.get("/optimize/status/{run_id}")
def get_status(run_id: str):
    """
    Check the status of a submitted run.
    """
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run_id not found")
    return {
        "run_id": run_id,
        "state": run["state"],
        "created_at": run["created_at"],
        "updated_at": run["updated_at"],
        "errors": run["errors"],
    }


@app.get("/optimize/result/{run_id}")
def get_result(run_id: str):
    """
    Retrieve results of a completed run.
    """
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run_id not found")
    if run["state"] != "completed":
        raise HTTPException(status_code=409, detail="Run not completed")
    return run["result"] | {"logs": run["logs"], "version": app.version}
