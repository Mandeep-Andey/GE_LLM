import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import sys
from pathlib import Path
import json
from collections import deque

import nltk
import ssl
import torch
from tqdm import tqdm
import argparse
import shutil

from src.settings import Settings
from src.data_preprocessor import load_books
from src.character_mapper import CharacterMapper
from src.prompt_manager import PromptManager
from src.llm_client import LLMClient

if __name__ == "__main__":
    # --- COMMAND-LINE ARGUMENT PARSING ---
    # THE FIX: Added the missing argparse setup block.
    parser = argparse.ArgumentParser(description="Run the LLM-based NLP extraction pipeline.")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="If set, deletes the entire 'llm_results' directory for a clean slate."
    )
    args = parser.parse_args()

    # --- 1. SETUP & ENVIRONMENT CHECK ---
    print("--- Conversational LLM Extraction Pipeline Started ---")

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    print("Setting up environment: Downloading NLTK resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print("Setup complete.")

    settings = Settings(config_path="config.yaml")

    RESULTS_DIR = settings.RESULTS_DIR
    if args.force_rerun and RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- 2. DATA LOADING ---
    print(f"\n--- Loading Book: {settings.TARGET_BOOK_FILENAME} ---")
    all_books_raw = load_books(settings.BOOKS_DIR)
    target_book_raw = all_books_raw.get(settings.TARGET_BOOK_FILENAME)
    if not target_book_raw:
        sys.exit(f"FATAL: Target book '{settings.TARGET_BOOK_FILENAME}' not found.")

    chapter_pattern = re.compile(r'^\s*Chapter\s*\d+\s*', re.MULTILINE)
    chapters_raw = chapter_pattern.split(target_book_raw)[1:]
    print(f"Found {len(chapters_raw)} chapters.")

    # --- 3. LLM PROCESSING ---
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
        sentences = nltk.sent_tokenize(chapter_text)
        all_chapter_interactions = []
        active_character_buffer = deque(maxlen=5)
        step_size = settings.CHUNK_SIZE - settings.CHUNK_OVERLAP
        num_chunks = max(1, (len(sentences) - settings.CHUNK_OVERLAP + step_size - 1) // step_size)

        for j in tqdm(range(num_chunks), desc=f"Chapter {i + 1} Chunks"):
            start = j * step_size
            end = start + settings.CHUNK_SIZE
            chunk_text = " ".join(sentences[start:end])

            if not chunk_text.strip(): continue

            prompt = prompt_manager.create_interaction_prompt(chunk_text, list(active_character_buffer))
            llm_responses = llm_client.get_llm_response(settings.LLM_MODEL, prompt)

            for response in llm_responses:
                if response and response.interactions:
                    interactions = [interaction.model_dump() for interaction in response.interactions]
                    all_chapter_interactions.extend(interactions)
                    for interaction in interactions:
                        if interaction['character_1'] not in active_character_buffer:
                            active_character_buffer.append(interaction['character_1'])
                        if interaction['character_2'] not in active_character_buffer:
                            active_character_buffer.append(interaction['character_2'])

        unique_interactions_set = {tuple(sorted(d.items())) for d in all_chapter_interactions}
        deduplicated_interactions = [dict(t) for t in unique_interactions_set]

        with open(chapter_output_path, 'w', encoding='utf-8') as f:
            json.dump({"interactions": deduplicated_interactions}, f, indent=2)

        print(
            f"Saved {len(deduplicated_interactions)} unique interactions for Chapter {i + 1} to {chapter_output_path}")

    print("\n\n--- LLM Extraction Complete ---")