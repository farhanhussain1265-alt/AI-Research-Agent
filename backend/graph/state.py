from pydantic import BaseModel,Field
from typing import TypedDict,List,Dict,Annotated,Literal,Optional
from backend.schemas.evidence_pack import EvidenceItem
from backend.schemas.plan import Plan
import operator

class State(TypedDict):

    topic:str

    # routing /research
    node:str
    needs_research:bool
    queries:List[str]
    evidence:List[EvidenceItem]

    plan:Optional[Plan]

    # recency
    as_of:str
    recency_days:int

    # workers
    sections:Annotated[List[tuple[int,str]],operator.add] # task_id,section_md

    # reducer/image
    merged_md:str
    md_with_placeholder:List[dict]

    final:str

