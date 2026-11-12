from pydantic import BaseModel,Field
from typing import TypedDict,List,Dict,Literal,Annotated


class Task(BaseModel):
    id:int
    title:str
    goal:str=Field(...,description="One sentence describing what the reader should do/understand.")
    bullets:List[str]=Field(...,min_length=3,max_length=5)
    target_words:int=Field(...,description="Target words (120-450).")

    tags:List[str]=Field(default_factory=list)
    require_research:bool=False
    require_citations:bool=False
    require_code:bool=False
    