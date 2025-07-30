from pydantic import BaseModel, Field
from typing import List, Literal

# Define controlled vocabularies for strong validation
InteractionType = Literal["Direct Dialogue", "Physical Action", "Observation", "Memory/Reference"]

class Interaction(BaseModel):
    """
    Defines the rich but focused data structure for a single character interaction.
    """
    character_1: str
    character_2: str
    interaction_type: InteractionType
    evidence_snippet: str = Field(..., description="A short, 3-5 word phrase from the text that proves the interaction.")

class LLMInteractionOutput(BaseModel):
    interactions: List[Interaction]