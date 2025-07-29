import re
import sys
from pathlib import Path
import json
from collections import deque

from tqdm import tqdm

from src.settings import Settings
from src.data_preprocessor import load_books
from src.character_mapper import CharacterMapper
from src.prompt_manager import PromptManager
from src.llm_client import LLMClient

if __name__ == "__main__":
    print("--- LLM-Based NLP Extraction Pipeline Started ---")
    settings = Settings(config_path="config.yaml")

    print(f"\n--- Loading Book: {settings.TARGET_BOOK_FILENAME} ---")
    all_books_raw = load_books(settings.BOOKS_DIR)
    target_book_raw = all_books_raw.get(settings.TARGET_BOOK_FILENAME)
    if not target_book_raw:
        print(f"FATAL: Target book not found.")
        sys.exit(1)

    chapter_pattern = re.compile(r'^\s*Chapter\s*\d+\s*', re.MULTILINE)
    chapters_raw = chapter_pattern.split(target_book_raw)[1:]
    print(f"Found {len(chapters_raw)} chapters.")

    character_mapper = CharacterMapper(file_path=str(settings.CHARACTER_FILE))
    prompt_manager = PromptManager(canonical_character_list=character_mapper.all_canonical_names)
    llm_client = LLMClient(host=settings.LLM_HOST)

    BOOK_RESULTS_DIR = settings.RESULTS_DIR / Path(settings.TARGET_BOOK_FILENAME).stem
    BOOK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for i, chapter_text in enumerate(chapters_raw):
        chapter_output_path = BOOK_RESULTS_DIR / f"chapter_{i:03d}.json"

        if chapter_output_path.exists():
            print(f"Skipping Chapter {i + 1} as its result file already exists.")
            continue

        print(f"\n--- Processing Chapter {i + 1}/{len(chapters_raw)} ---")
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', chapter_text) if p.strip()]

        all_chapter_interactions = []

        # THE FIX: Implement the Active Character Buffer
        # A deque is a list-like object optimized for adding/removing from the ends.
        active_character_buffer = deque(maxlen=5)  # Remember the last 5 characters

        for paragraph in tqdm(paragraphs, desc=f"Chapter {i + 1} Paragraphs"):
            # Create the prompt with the current context buffer
            prompt = prompt_manager.create_interaction_prompt(paragraph, list(active_character_buffer))
            llm_response = llm_client.get_llm_response(settings.LLM_MODEL, prompt)

            if llm_response and llm_response.interactions:
                interactions = [interaction.model_dump() for interaction in llm_response.interactions]
                all_chapter_interactions.extend(interactions)

                # Update the buffer with characters found in this paragraph's interactions
                for interaction in interactions:
                    if interaction['character_1'] not in active_character_buffer:
                        active_character_buffer.append(interaction['character_1'])
                    if interaction['character_2'] not in active_character_buffer:
                        active_character_buffer.append(interaction['character_2'])

        with open(chapter_output_path, 'w', encoding='utf-8') as f:
            json.dump({"interactions": all_chapter_interactions}, f, indent=2)

        print(f"Saved {len(all_chapter_interactions)} interactions for Chapter {i + 1} to {chapter_output_path}")

    print("\n\n--- LLM Extraction Complete ---")