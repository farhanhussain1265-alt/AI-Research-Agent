from backend.schemas.router_decision import RouterDecision
from backend.graph.state import State
from backend.services.llm_services import model
from backend.prompts.router_system import ROUTER_SYSTEM
from langchain_core.messages import SystemMessage,HumanMessage

def router_node(state:State)->dict:
    decider=model.with_structured_output(RouterDecision)

    decision=decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date:{state['as_of']}")
        ]
    )
    if decision.mode=="open_book":
        recency_days=7
    elif decision.mode=="hybrid":
        recency_days=45
    else:
        recency_days=365

    return {
        "needs_research":decision.needs_research,
        "mode":decision.mode,
        "queries":decision.queries,
        "recency_days":recency_days
    }