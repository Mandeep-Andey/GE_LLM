import requests
import json
from typing import List, Optional
from pydantic import ValidationError
from src.schemas import LLMInteractionOutput


class LLMClient:
    def __init__(self, host: str):
        self.api_url = f"{host}/api/generate"
        print(f"LLM Client initialized for Ollama GENERATE server at {self.api_url}")

    def _extract_json_objects(self, text: str) -> List[str]:
        """
        A resilient parser that finds and extracts one or more complete JSON objects
        from a raw string.
        """
        json_objects = []
        brace_level = 0
        current_obj_start = -1

        for i, char in enumerate(text):
            if char == '{':
                if brace_level == 0:
                    current_obj_start = i
                brace_level += 1
            elif char == '}':
                if brace_level > 0:
                    brace_level -= 1
                    if brace_level == 0 and current_obj_start != -1:
                        json_objects.append(text[current_obj_start: i + 1])
                        current_obj_start = -1
        return json_objects

    def get_llm_response(self, model_name: str, prompt: str) -> List[LLMInteractionOutput]:
        """
        Sends a prompt to Ollama and resiliently parses the response,
        handling single or multiple JSON objects.

        Returns:
            A list of validated LLMInteractionOutput objects. The list will be
            empty if no valid objects could be parsed.
        """
        try:
            payload = {"model": model_name, "prompt": prompt, "stream": False, "format": "json"}
            response = requests.post(self.api_url, json=payload, timeout=600)
            response.raise_for_status()

            response_data = response.json()
            raw_output_string = response_data.get("response", "")

            if not raw_output_string:
                return []

            # Use the resilient parser to find all potential JSON objects in the response
            json_strings = self._extract_json_objects(raw_output_string)

            if not json_strings:
                print(f"\nWARNING: LLM produced a non-JSON response.")
                print(f"--- LLM Raw Output ---\n{raw_output_string}\n--------------------")
                return []

            validated_responses = []
            for json_str in json_strings:
                try:
                    # Validate each found JSON object against our Pydantic schema
                    validated_output = LLMInteractionOutput.model_validate_json(json_str)
                    validated_responses.append(validated_output)
                except ValidationError as e:
                    print(
                        f"\nWARNING: A JSON object from the LLM failed Pydantic validation. Skipping it. Details:\n{e}")
                    print(f"--- Invalid JSON Object ---\n{json_str}\n--------------------")

            return validated_responses

        except requests.exceptions.RequestException as e:
            print(f"\nERROR: Could not connect to Ollama server. Details: {e}")
            return []