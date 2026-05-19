from pydantic import BaseModel, Field
from typing import List

class Chapter(BaseModel):
    name: str                            = Field(description="章名称")
    description: str                     = Field(description="章的描述")
    learning_objectives : List[str]      = Field(default_factory=list, description="学习目标列表")
    
class Outline(BaseModel):
    chapters : List[Chapter]             = Field(description="章列表")