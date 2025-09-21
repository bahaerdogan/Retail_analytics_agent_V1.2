from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict
import json
from datetime import datetime
from agent.rag.retrieval import retrieve_chunks
from agent.tools.sqlite_tool import SQLiteTool
from agent.haystack_modules import Router, NL2SQL, Synthesizer, Planner

class AgentState(TypedDict, total=False):
    question: str
    id: str
    format_hint: str
    route: str
    retrieved_docs: list
    constraints: dict
    sql: str
    sql_result: list
    sql_error: str
    repair_attempts: int
    final_answer: Any
    citations: list
    confidence: float
    explanation: str

def _log_event(event: Dict[str, Any]) -> None:
    try:
        payload = dict(event)
        payload["ts"] = datetime.utcnow().isoformat() + "Z"
        with open("agent_trace.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass

def router_node(state: AgentState) -> AgentState:
    question = state.get("question") or state.get("input") or ""
    state["question"] = question
    route = Router().classify(question)
    state["route"] = route
    _log_event({"node": "router", "route": route})
    return state

def retriever_node(state: AgentState) -> AgentState:
    question = state.get("question") or state.get("input") or ""
    state["question"] = question
    chunks = retrieve_chunks(question, top_k=6)
    state["retrieved_docs"] = chunks
    _log_event({"node": "retriever", "retrieved_ids": [c.get("id") for c in chunks]})
    return state

def planner_node(state: AgentState) -> AgentState:
    question = state.get("question") or state.get("input") or ""
    state["question"] = question
    constraints = Planner().extract(question, state.get("retrieved_docs", []))
    state["constraints"] = constraints
    _log_event({"node": "planner", "constraints": constraints})
    return state

def nl2sql_node(state: AgentState) -> AgentState:
    route = state.get("route")
    if not route:
        question = state.get("question") or state.get("input") or ""
        route = Router().classify(question)
        state["route"] = route
    if route in ["sql", "hybrid"]:
        sql = NL2SQL().generate(
            question=state.get("question", ""),
            schema=SQLiteTool().get_schema(),
            constraints=state.get("constraints", {})
        )
        state["sql"] = sql
    else:
        state["sql"] = ""
    _log_event({"node": "nl2sql", "has_sql": bool(state["sql"])})
    return state

def executor_node(state: AgentState) -> AgentState:
    if state.get("sql"):
        try:
            rows = SQLiteTool().execute(state["sql"])
            state["sql_result"] = rows or []
            state["sql_error"] = None
        except Exception as e:
            state["sql_result"] = []
            state["sql_error"] = str(e)
    else:
        state["sql_result"] = []
        state["sql_error"] = None
    _log_event({"node": "executor", "rows": len(state.get("sql_result", [])), "error": state.get("sql_error")})
    return state

def synthesizer_node(state: AgentState) -> AgentState:
    answer = Synthesizer().synthesize(
        question=state.get("question", ""),
        docs=state.get("retrieved_docs", []),
        sql_result=state.get("sql_result", []),
        sql=state.get("sql", ""),
        constraints=state.get("constraints", {}),
        format_hint=state.get("format_hint", ""),
    )
    state["final_answer"] = answer["final_answer"]
    state["citations"] = answer["citations"]
    state["confidence"] = answer["confidence"]
    state["explanation"] = answer["explanation"]
    _log_event({"node": "synthesizer", "final_type": type(state["final_answer"]).__name__})
    return state

def repair_node(state: AgentState) -> AgentState:
    empty_result = False
    if state.get("sql"):
        rows = state.get("sql_result", [])
        if not rows:
            empty_result = True
        elif len(rows) == 1 and isinstance(rows[0], tuple) and len(rows[0]) == 1 and rows[0][0] is None:
            empty_result = True

    repair_attempts = state.get("repair_attempts") or 0
    if (state.get("sql_error") or empty_result) and repair_attempts < 2:
        state["repair_attempts"] = repair_attempts + 1
        sql = NL2SQL().generate(
            question=state.get("question", ""),
            schema=SQLiteTool().get_schema(),
            constraints=state.get("constraints", {}),
            attempt=state["repair_attempts"]
        )
        state["sql"] = sql
        _log_event({"node": "repair", "attempt": state["repair_attempts"]})
        return executor_node(state)
    _log_event({"node": "repair", "attempt": repair_attempts, "skipped": True})
    return state

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("nl2sql", nl2sql_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("repair", repair_node)

    workflow.set_entry_point("router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "planner")
    workflow.add_edge("planner", "nl2sql")
    workflow.add_edge("nl2sql", "executor")
    workflow.add_edge("executor", "repair")
    workflow.add_edge("repair", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()