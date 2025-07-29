from typing import List

class PromptManager:
    def __init__(self, canonical_character_list: List[str]):
        self.character_list_str = ",\n".join(f'    "{name}"' for name in canonical_character_list)

    def create_interaction_prompt(self, paragraph_text: str, active_characters: List[str] = None) -> str:
        context_block = ""
        if active_characters:
            active_chars_str = ", ".join(f'"{name}"' for name in active_characters)
            context_block = f"""**CONTEXT - Characters most recently mentioned in the scene:**
[{active_chars_str}]
"""
        # THE FIX: The prompt now explicitly asks for the summary and quote, and the example reflects this.
        return f"""
**TASK:**
Generate a JSON object listing all direct character interactions in the `## PARAGRAPH TO ANALYZE ##`.

**STRICT RULES:**
1.  **OUTPUT FORMAT:** Respond with a single, valid JSON object only. No explanations.
2.  **JSON SCHEMA:** The JSON must be `{{ "interactions": [ {{ "character_1": "string", "character_2": "string", "interaction_summary": "string", "quote": "string" }} ] }}`.
3.  **CANONICAL NAMES:** You MUST use the exact names from the `## VALID CHARACTERS ##` list.
4.  **QUOTE ACCURACY:** The `quote` field MUST be the exact, unmodified sentence or phrase from the text that shows the interaction.

**EXAMPLE:**
---
## EXAMPLE INPUT ##
CONTEXT - Characters most recently mentioned in the scene:
[ "Dorothea Brooke", "Celia Brooke" ]

PARAGRAPH TO ANALYZE:
Mr. Brooke watched them both. "Come here, Dorothea," she said, with some satisfaction.

## EXAMPLE JSON OUTPUT ##
{{
  "interactions": [
    {{
      "character_1": "Mr. Arthur Brooke",
      "character_2": "Dorothea Brooke",
      "interaction_summary": "Mr. Brooke observes Dorothea.",
      "quote": "Mr. Brooke watched them both."
    }},
    {{
      "character_1": "Mr. Arthur Brooke",
      "character_2": "Celia Brooke",
      "interaction_summary": "Mr. Brooke observes Celia.",
      "quote": "Mr. Brooke watched them both."
    }},
    {{
      "character_1": "Celia Brooke",
      "character_2": "Dorothea Brooke",
      "interaction_summary": "Celia speaks to Dorothea, expressing satisfaction.",
      "quote": "\\"Come here, Dorothea,\\" she said, with some satisfaction."
    }}
  ]
}}
---

**DATA FOR CURRENT TASK:**
---
{context_block}
## VALID CHARACTERS ##
[
{self.character_list_str}
]

## PARAGRAPH TO ANALYZE ##
{paragraph_text}
---

**YOUR JSON OUTPUT:**
"""