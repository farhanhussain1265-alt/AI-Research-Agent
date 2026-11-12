from backend.graph.state import State


def route_next(state:State)->dict:
    return "research" if state["needs_research"] else "orchestrator"