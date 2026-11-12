from backend.graph.state import State
from typing import List,Dict
from backend.tools.tavily import _tavily_search
from backend.services.llm_services import model
from backend.schemas.evidence_pack import EvidenceItem,EvidencePack
from langchain_core.messages import HumanMessage,SystemMessage
from backend.prompts.research_system import RESEARCH_SYSTEM
from datetime import date, timedelta
from backend.services.research_service import _iso_to_date

# ---Research Node----
def research_node(state:State)->dict:
    queries=(state.get("queries") or []) or []
    raw:List[dict]=[]

    for q in queries:
        raw.extend(_tavily_search(q,max_results=6))

    if not raw:
        return {"evidence":[]}

    extractor=model.with_structured_output(EvidencePack)
    pack=extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}",
                    f"Recency days: {state['recency_days']}",
                    f"Raw results: \n{raw}"
                )
            )
        ]
    )
    dedup={}
    for e in pack.evidence:
        if e.url:
            dedup[e.url]=e
    evidence=list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}