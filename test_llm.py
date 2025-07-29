import time
import json
from src.llm_client import LLMClient
from src.prompt_manager import PromptManager
from src.character_mapper import CharacterMapper
from src.data_preprocessor import load_books
import re

if __name__ == "__main__":
    # --- 1. SETUP ---
    print("--- LLM Sandbox Test ---")

    # IMPORTANT: Use the exact names of your local Ollama models here.
    # Common names are 'gemma:2b', 'gemma:7b', 'gemma2:9b', etc.
    # I will assume 'gemma:2b' for the smaller model and 'gemma2:9b' for the larger,
    # as these are standard. Please adjust if your names are different.
    MODELS_TO_TEST = ["gemma3:4b", "gemma3:12b"]  # <-- ADJUST THESE NAMES IF NEEDED

    # Load a single chapter for the test case
    print("Loading test chapter...")
    all_books_raw = load_books("./data/Middlemarch")
    book_1_raw = all_books_raw.get("book_1.txt", "")
    chapter_pattern = re.compile(r'^\s*Chapter\s*\d+\s*', re.MULTILINE)
    chapters_raw = chapter_pattern.split(book_1_raw)[1:]
    test_chapter_text = chapters_raw[0]  # Use Chapter 1

    # Load character list to inject into the prompt
    character_mapper = CharacterMapper(file_path="./char_alias.json")
    canonical_names = character_mapper.all_canonical_names

    # Prepare the prompt and the client
    prompt_manager = PromptManager(canonical_character_list=canonical_names)
    prompt = prompt_manager.create_interaction_prompt(test_chapter_text)

    llm_client = LLMClient()

    # --- 2. RUN TESTS ---
    for model_name in MODELS_TO_TEST:
        print("\n" + "=" * 50)
        print(f"--- TESTING MODEL: {model_name} ---")
        print("=" * 50)

        start_time = time.time()
        response_data = llm_client.get_llm_response(model_name, prompt)
        end_time = time.time()

        print(f"Time Taken: {end_time - start_time:.2f} seconds")

        if response_data:
            print("\n--- LLM Response (Parsed JSON) ---")
            # Use json.dumps for pretty printing
            print(json.dumps(response_data, indent=2))

            num_interactions = len(response_data.get("interactions", []))
            print(f"\nAnalysis: Found {num_interactions} interactions.")
        else:
            print("\nAnalysis: Model failed to return a valid response.")