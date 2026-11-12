from typing import TypedDict,List,Dict,Annotated,Literal,Optional
from pydantic import BaseModel,Field


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder:str=Field(...,description="e.g.[[IMAGE_1]]")
    filename:str=Field(...,description="Saved under image/, e.g. qkv_flow.png")
    alt:str
    caption:str
    prompt:str=Field(...,description="Prompt to send to image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"

class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)