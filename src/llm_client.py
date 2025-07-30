import requests
import json
from typing import Optional
from pydantic import ValidationError, PydanticUserError
from src.schemas import LLMInteractionOutput


class LLMClient:
    def __init__(self, host: str):
        self.api_url = f"{host}/api/generate"  # Reverted to the stateless endpoint
        print(f"LLM Client initialized for Ollama GENERATE server at {self.api_url}")

    def get_llm_response(self, model_name: str, prompt: str) -> Optional[LLMInteractionOutput]:
        try:
            payload = {"model": model_name, "prompt": prompt, "stream": False, "format": "json"}
            response = requests.post(self.api_url, json=payload, timeout=600)
            response.raise_for_status()

            response_data = response.json()
            json_string = response_data.get("response", "{}")

            # The "self-healing" logic to fix common LLM key typos
            if '"interaction, type":' in json_string:
                json_string = json_string.replace('"interaction, type":', '"interaction_type":')

            validated_output = LLMInteractionOutput.model_validate_json(json_string)
            return validated_output

        except requests.exceptions.RequestException as e:
            print(f"\nERROR: Could not connect to Ollama server. Details: {e}")
            return None
        except (ValidationError, PydanticUserError, json.JSONDecodeError) as e:
            print(f"\nERROR: LLM output failed validation. Details:\n{e}")
            print(f"--- LLM Raw Output ---\n{json_string}\n--------------------")
            return None