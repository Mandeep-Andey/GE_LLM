from typing import List

class PromptManager:
    def __init__(self, canonical_character_list: List[str]):
        self.character_list_str = ",\n".join(f'    "{name}"' for name in canonical_character_list)
        self.initial_prompt_template = self._build_initial_prompt()

    def _build_initial_prompt(self) -> str:
        # THE DEFINITIVE FIX: This prompt uses the direct, machine-like structure you
        # recommended, but is updated to request the rich data fields we need.
        return f"""
**TASK:**
Generate a JSON object that lists all direct character interactions in the `## PARAGRAPH TO ANALYZE ##`.

**STRICT RULES:**
1.  **OUTPUT FORMAT:** Respond ONLY with a single, valid JSON object. Do not include any text, explanations, or markdown.
2.  **JSON SCHEMA:** The JSON must be `{{ "interactions": [ {{ "character_1": "string", "character_2": "string", "interaction_type": "string", "evidence_snippet": "string" }} ] }}`.
3.  **CANONICAL NAMES:** You MUST use the exact names from the `## VALID CHARACTERS ##` list.
4.  **INTERACTION TYPE:** For the `interaction_type` field, you MUST choose ONE value from: ["Direct Dialogue", "Physical Action", "Observation", "Memory/Reference"].
5.  **EVIDENCE SNIPPET:** The `evidence_snippet` MUST be a short, 3-5 word phrase copied EXACTLY from the text that proves the interaction.
6.  **CONTEXT:** Use our chat history to understand who pronouns refer to. Only extract interactions from the most recent paragraph.

**EXAMPLE OF A SINGLE TURN:**
---
## MY INPUT PARAGRAPH ##
"Come here, Dorothea," said Celia, her voice filled with warmth. Mr. Brooke, standing in the doorway of the library, watched them both with a smile.

## YOUR JSON OUTPUT ##
{{
  "interactions": [
    {{
      "character_1": "Celia Brooke",
      "character_2": "Dorothea Brooke",
      "interaction_type": "Direct Dialogue",
      "evidence_snippet": "Come here, Dorothea"
    }},
    {{
      "character_1": "Mr. Arthur Brooke",
      "character_2": "Dorothea Brooke",
      "interaction_type": "Observation",
      "evidence_snippet": "watched them both"
    }}
  ]
}}
---

Here is the definitive list of characters for our entire session.
## VALID CHARACTERS ##
[
{self.character_list_str}
]

I will now provide the first paragraph. Process it and provide the JSON output.
"""

    def get_initial_prompt(self) -> str:
        """Returns the initial, detailed instruction prompt."""
        return self.initial_prompt_template

    def format_paragraph_prompt(self, paragraph_text: str) -> str:
        """Formats a subsequent paragraph for the ongoing chat."""
        return f"Process this next paragraph according to all rules I have provided.\n\n---\n{paragraph_text}"