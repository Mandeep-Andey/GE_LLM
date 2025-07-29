import time
import json
import re
from src.llm_client import LLMClient
from src.prompt_manager import PromptManager
from src.character_mapper import CharacterMapper

if __name__ == "__main__":
    print("--- LLM Context-Awareness Sandbox Test ---")

    MODEL_TO_TEST = "qwen3:8b"  # Use the better model for this quality test

    # The specific paragraphs from Chapter 1 that demonstrate the problem
    paragraph1 = "Early in the day Dorothea had returned from the infant school which she had set going in the village, and was taking her usual place in the pretty sitting-room which divided the bedrooms of the sisters, bent on finishing a plan for some buildings (a kind of work which she delighted in), when Celia, who had been watching her with a hesitating desire to propose something, said—"
    paragraph2 = "“Dorothea, dear, if you don’t mind—if you are not very busy—suppose we looked at mamma’s jewels to-day, and divided them? It is exactly six months to-day since uncle gave them to you, and you have not looked at them yet.”"

    # Setup our tools
    character_mapper = CharacterMapper(file_path="./char_alias.json")
    canonical_names = character_mapper.all_canonical_names
    prompt_manager = PromptManager(canonical_character_list=canonical_names)
    llm_client = LLMClient()

    # --- TEST 1: Process Paragraph 2 WITHOUT context ---
    print("\n" + "=" * 50)
    print("--- 1. TESTING PARAGRAPH 2 (NO CONTEXT) ---")
    print("--- EXPECTED TO FAIL OR BE INCOMPLETE ---")
    print("=" * 50)

    prompt_no_context = prompt_manager.create_interaction_prompt(paragraph2)
    response_no_context = llm_client.get_llm_response(MODEL_TO_TEST, prompt_no_context)

    if response_no_context:
        print(response_no_context.model_dump_json(indent=2))
        print(f"\nAnalysis: Found {len(response_no_context.interactions)} interactions.")
    else:
        print("\nAnalysis: Model failed to return a valid response.")

    # --- TEST 2: Process Paragraph 2 WITH context ---
    print("\n" + "=" * 50)
    print("--- 2. TESTING PARAGRAPH 2 (WITH CONTEXT) ---")
    print("--- EXPECTED TO SUCCEED ---")
    print("=" * 50)

    # Simulate the "Active Character Buffer" by identifying characters from Paragraph 1
    active_characters = ["Dorothea Brooke", "Celia Brooke"]
    print(f"Providing context buffer: {active_characters}")

    prompt_with_context = prompt_manager.create_interaction_prompt(paragraph2, active_characters=active_characters)
    response_with_context = llm_client.get_llm_response(MODEL_TO_TEST, prompt_with_context)

    if response_with_context:
        print(response_with_context.model_dump_json(indent=2))
        print(f"\nAnalysis: Found {len(response_with_context.interactions)} interactions.")
    else:
        print("\nAnalysis: Model failed to return a valid response.")