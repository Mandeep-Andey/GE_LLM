from pydantic import BaseModel, Field
from typing import List

class Interaction(BaseModel):
    character_1: str
    character_2: str
    # THE CHANGE: The summary is back, and we've added a field for the quote.
    interaction_summary: str = Field(..., description="A brief, 10-15 word summary of the interaction.")
    quote: str = Field(..., description="The exact sentence or phrase from the text that contains the interaction.")

class LLMInteractionOutput(BaseModel):
    interactions: List[Interaction]