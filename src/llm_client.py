import requests
import json
from typing import List, Optional, Dict
from pydantic import ValidationError
from src.schemas import LLMInteractionOutput, Interaction


class LLMClient:
    """
    A client to handle communication with a local Ollama LLM instance.
    This version includes a resilient parser that validates interactions individually
    to maximize data recovery from imperfect LLM outputs.
    """

    def __init__(self, host: str):
        self.api_url = f"{host}/api/generate"
        print(f"LLM Client initialized for Ollama GENERATE server at {self.api_url}")

    def _heal_interaction_keys(self, interaction_dict: Dict) -> Dict:
        """Fixes common key typos from the LLM before validation."""
        healed_dict = {}
        for key, value in interaction_dict.items():
            # Remove trailing commas or spaces from keys
            cleaned_key = key.strip().replace(',', '')
            healed_dict[cleaned_key] = value
        return healed_dict

    def get_llm_response(self, model_name: str, prompt: str) -> Optional[LLMInteractionOutput]:
        """
        Sends a prompt to Ollama and resiliently parses the response,
        validating each interaction individually.
        """
        try:
            payload = {"model": model_name, "prompt": prompt, "stream": False, "format": "json"}
            response = requests.post(self.api_url, json=payload, timeout=600)
            response.raise_for_status()

            response_data = response.json()
            json_string = response_data.get("response", "{}")

            # 1. First, parse the raw string into a basic Python dictionary.
            raw_data = json.loads(json_string)

            # 2. Extract the list of interactions. If it's not there, it's a major failure.
            unvalidated_interactions = raw_data.get("interactions")
            if unvalidated_interactions is None:
                print(
                    f"\nWARNING: LLM response was valid JSON but missing the required 'interactions' key. Output ignored.")
                print(f"--- LLM Raw Output ---\n{json_string}\n--------------------")
                return None

            # 3. Iterate and validate each interaction individually.
            valid_interactions: List[Interaction] = []
            for interaction_dict in unvalidated_interactions:
                try:
                    # First, try to heal any common key typos
                    healed_dict = self._heal_interaction_keys(interaction_dict)
                    # Now, validate the single interaction against the Interaction schema
                    validated_interaction = Interaction.model_validate(healed_dict)
                    valid_interactions.append(validated_interaction)
                except ValidationError as e:
                    print(f"\nWARNING: Skipping one malformed interaction object. Details:\n{e}")
                    print(f"--- Invalid Interaction Object ---\n{interaction_dict}\n--------------------")
                    continue  # Skip this bad interaction and continue to the next one

            # 4. Reassemble the final, fully validated Pydantic object.
            return LLMInteractionOutput(interactions=valid_interactions)

        except requests.exceptions.RequestException as e:
            print(f"\nERROR: Could not connect to Ollama server. Details: {e}")
            return None
        except json.JSONDecodeError as e:
            # This catches cases where the LLM's entire output is not even valid JSON
            print(f"\nERROR: LLM output was not valid JSON. Details: {e}")
            print(f"--- LLM Raw Output ---\n{json_string}\n--------------------")
            return None