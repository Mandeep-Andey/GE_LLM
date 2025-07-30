from typing import List

class PromptManager:
    def __init__(self, canonical_character_list: List[str]):
        self.character_list_str = ",\n".join(f'    "{name}"' for name in canonical_character_list)

    def create_interaction_prompt(self, paragraph_text: str, active_characters: List[str] = None) -> str:
        context_block = ""
        if active_characters:
            active_chars_str = ", ".join(f'"{name}"' for name in active_characters)
            context_block = f"""**CONTEXT - Characters recently mentioned in the scene:**
[{active_chars_str}]
"""
        return f"""
**TASK:**
Generate a JSON object that lists all direct character interactions in the `## PARAGRAPH TO ANALYZE ##`.

**STRICT RULES:**
1.  **OUTPUT FORMAT:** Respond ONLY with a single, valid JSON object.
2.  **JSON SCHEMA:** The JSON must be `{{ "interactions": [ {{ "character_1": "string", "character_2": "string", "interaction_type": "string", "evidence_snippet": "string" }} ] }}`.
3.  **CANONICAL NAMES:** You MUST use the exact names from the `## VALID CHARACTERS ##` list.
4.  **INTERACTION TYPE:** You MUST choose ONE value from: ["Direct Dialogue", "Physical Action", "Observation", "Memory/Reference"].
5.  **EVIDENCE SNIPPET:** The `evidence_snippet` MUST be a short, 3-5 word phrase copied EXACTLY from the text.
6.  **FOCUS:** Use the `CONTEXT` to help identify pronouns, but only extract interactions explicitly present in the `## PARAGRAPH TO ANALYZE ##`.

**EXAMPLE:**
---
## EXAMPLE INPUT ##
CONTEXT - Characters recently mentioned in the scene:
[ "Dorothea Brooke", "Celia Brooke" ]

PARAGRAPH TO ANALYZE:
Mr. Brooke watched them both. "Come here, Dorothea," she said, with some satisfaction.

## EXAMPLE JSON OUTPUT ##
{{
  "interactions": [
    {{
      "character_1": "Mr. Arthur Brooke",
      "character_2": "Dorothea Brooke",
      "interaction_type": "Observation",
      "evidence_snippet": "watched them both"
    }},
    {{
      "character_1": "Celia Brooke",
      "character_2": "Dorothea Brooke",
      "interaction_type": "Direct Dialogue",
      "evidence_snippet": "Come here, Dorothea"
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