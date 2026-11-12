from pydantic import BaseModel,Field
from typing import TypedDict,List,Dict,Annotated,Optional

class EvidenceItem(BaseModel):
    title:str
    url:str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class EvidencePack(BaseModel):
    evidence:List[EvidenceItem]=Field(default_factory=list)
    
