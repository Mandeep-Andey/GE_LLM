from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# Define controlled vocabularies for strong validation. This forces the LLM to be consistent.
InteractionType = Literal["Direct Dialogue", "Physical Action", "Observation", "Memory/Reference"]
SentimentType = Literal["Positive", "Negative", "Neutral", "Ambiguous"]

class Interaction(BaseModel):
    """
    Defines the rich data structure for a single character interaction.
    """
    character_1: str
    character_2: str
    interaction_type: InteractionType
    # THE CHANGE: Location is now optional, which is more robust.
    location: Optional[str] = Field(None, description="The inferred setting of the interaction, if mentioned.")
    sentiment: SentimentType
    evidence_snippet: str = Field(..., description="A short, 3-5 word phrase from the text that proves the interaction.")

class LLMInteractionOutput(BaseModel):
    interactions: List[Interaction]