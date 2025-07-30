import requests
import json
from typing import Optional, List, Dict
from pydantic import ValidationError
from src.schemas import LLMInteractionOutput


class LLMClient:
    """
    A stateful client to handle conversational chains with a local Ollama LLM instance.
    """

    def __init__(self, host: str):
        self.api_url = f"{host}/api/chat"  # NOTE: We use the /api/chat endpoint now
        self.conversation_history: List[Dict[str, str]] = []
        print(f"LLM Client initialized for Ollama CHAT server at {self.api_url}")

    def start_new_chat(self):
        """Resets the conversation history for a new chapter."""
        self.conversation_history = []
        # print("New chat session started.")

    def send_paragraph(self, model_name: str, prompt: str) -> Optional[LLMInteractionOutput]:
        """
        Sends the next paragraph in the conversation and gets a structured response.
        """
        if not prompt:
            return None

        # Add the new user prompt to the history
        self.conversation_history.append({"role": "user", "content": prompt})

        try:
            payload = {
                "model": model_name,
                "messages": self.conversation_history,
                "stream": False,
                "format": "json"
            }

            response = requests.post(self.api_url, json=payload, timeout=600)
            response.raise_for_status()

            response_data = response.json()

            # The actual message content is nested differently in the chat API
            assistant_message = response_data.get("message", {})
            json_string = assistant_message.get("content", "{}")

            # Add the assistant's valid response to the history for the next turn
            self.conversation_history.append({"role": "assistant", "content": json_string})

            validated_output = LLMInteractionOutput.model_validate_json(json_string)
            return validated_output

        except requests.exceptions.RequestException as e:
            print(f"\nERROR: Could not connect to Ollama server. Details: {e}")
            # Remove the failed user prompt from history
            self.conversation_history.pop()
            return None
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"\nERROR: LLM output failed validation. Terminating chat for this chapter. Details:\n{e}")
            print(f"--- LLM Raw Output ---\n{json_string}\n--------------------")
            # The chat is now "poisoned". We reset the history to start fresh on the next paragraph.
            self.start_new_chat()
            return None