import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import sys
from pathlib import Path
import json
from collections import deque

import nltk
import ssl
from tqdm import tqdm
import argparse
import shutil
from transformers import AutoTokenizer

from src.settings import Settings
from src.data_preprocessor import load_books
from src.character_mapper import CharacterMapper
from src.prompt_manager import PromptManager
from src.llm_client import LLMClient


def create_adaptive_chunks(sentences: list[str], tokenizer, token_limit: int, overlap_sentences: int) -> list[str]:
    chunks = []
    current_chunk_sentences = deque()
    current_token_count = 0

    for sentence in sentences:
        sentence_token_count = len(tokenizer.encode(sentence))

        if current_token_count + sentence_token_count > token_limit and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            # Create the overlap by preserving the last few sentences
            overlapped_sentences = []
            for _ in range(min(overlap_sentences, len(current_chunk_sentences))):
                overlapped_sentences.insert(0, current_chunk_sentences.pop())
            current_chunk_sentences = deque(overlapped_sentences)
            current_token_count = len(tokenizer.encode(" ".join(current_chunk_sentences)))

        current_chunk_sentences.append(sentence)
        current_token_count += sentence_token_count

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LLM-based NLP extraction pipeline.")
    parser.add_argument("--force-rerun", action="store_true", help="Deletes 'llm_results' for a clean slate.")
    args = parser.parse_args()

    # --- 1. SETUP ---
    print("--- LLM Extraction Pipeline with Adaptive Chunking Started ---")
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt', quiet=True)

    settings = Settings(config_path="config.yaml")

    RESULTS_DIR = settings.RESULTS_DIR
    if args.force_rerun and RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- 2. DATA LOADING & PREP ---
    print(f"\n--- Loading all books from '{settings.BOOKS_DIR}' ---")
    all_books_raw = load_books(settings.BOOKS_DIR)
    print(f"Found {len(all_books_raw)} books to process.")

    counting_tokenizer = AutoTokenizer.from_pretrained(settings.FAST_TOKENIZER)
    character_mapper = CharacterMapper(file_path=str(settings.CHARACTER_FILE))
    prompt_manager = PromptManager(canonical_character_list=character_mapper.all_canonical_names)
    llm_client = LLMClient(host=settings.LLM_HOST)

    # --- 3. LLM PROCESSING ---
    for book_filename, book_text in all_books_raw.items():
        book_name = Path(book_filename).stem
        print(f"\n\n--- Processing Book: {book_name} ---")
        BOOK_RESULTS_DIR = RESULTS_DIR / book_name
        BOOK_RESULTS_DIR.mkdir(exist_ok=True)

        chapter_pattern = re.compile(r'^\s*Chapter\s*\d+\s*', re.MULTILINE)
        chapters_raw = chapter_pattern.split(book_text)[1:]
        print(f"Found {len(chapters_raw)} chapters.")

        for i, chapter_text in enumerate(chapters_raw):
            chapter_output_path = BOOK_RESULTS_DIR / f"chapter_{i:03d}.json"
            if chapter_output_path.exists():
                print(f"Skipping Chapter {i + 1} as its result file already exists.")
                continue

            print(f"\n--- Processing Chapter {i + 1}/{len(chapters_raw)} ---")
            sentences = nltk.sent_tokenize(chapter_text)
            chunks = create_adaptive_chunks(sentences, counting_tokenizer, settings.CHUNK_TOKEN_LIMIT,
                                            settings.CHUNK_OVERLAP_SENTENCES)

            all_chapter_interactions = []
            active_character_buffer = deque(maxlen=5)

            for chunk_text in tqdm(chunks, desc=f"Chapter {i + 1} Chunks"):
                prompt = prompt_manager.create_interaction_prompt(chunk_text, list(active_character_buffer))

                # THE FIX: The client returns a single object or None, not a list.
                llm_response = llm_client.get_llm_response(settings.LLM_MODEL, prompt)

                # THE FIX: We no longer loop. We just check if the single response is valid.
                if llm_response and llm_response.interactions:
                    interactions = [interaction.model_dump() for interaction in llm_response.interactions]
                    all_chapter_interactions.extend(interactions)
                    for interaction in interactions:
                        if interaction['character_1'] not in active_character_buffer:
                            active_character_buffer.append(interaction['character_1'])
                        if interaction['character_2'] not in active_character_buffer:
                            active_character_buffer.append(interaction['character_2'])

            # THE FIX: Simpler, more robust deduplication.
            # Convert each dict to a string to make it hashable for the set.
            seen = set()
            deduplicated_interactions = []
            for interaction in all_chapter_interactions:
                # Create a unique key for the interaction, ignoring order of characters
                key_part1 = tuple(sorted((interaction['character_1'], interaction['character_2'])))
                key_part2 = interaction['evidence_snippet']
                interaction_key = (key_part1, key_part2)

                if interaction_key not in seen:
                    deduplicated_interactions.append(interaction)
                    seen.add(interaction_key)

            with open(chapter_output_path, 'w', encoding='utf-8') as f:
                json.dump({"interactions": deduplicated_interactions}, f, indent=2)

            print(f"Saved {len(deduplicated_interactions)} unique interactions for Chapter {i + 1}")

    print("\n\n--- LLM Extraction Complete ---")