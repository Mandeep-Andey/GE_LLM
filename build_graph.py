import json
from pathlib import Path
from typing import List, Tuple
import networkx as nx
from tqdm import tqdm
import argparse
import sys

from src.settings import Settings
from src.character_mapper import CharacterMapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a graph artifact from pre-computed NLP results.")
    parser.add_argument("book_name", type=str, help="The name of the book to process (e.g., 'book_1').")
    args = parser.parse_args()

    settings = Settings(config_path="config.yaml")
    print(f"\n--- Graph Builder Started for Book: {args.book_name} ---")

    BOOK_RESULTS_DIR = settings.RESULTS_DIR / args.book_name
    if not BOOK_RESULTS_DIR.exists():
        print(f"FATAL: No results found. Please run 'run_llm_extraction.py' first.")
        sys.exit(1)

    character_mapper = CharacterMapper(file_path=str(settings.CHARACTER_FILE))

    G = nx.Graph()

    result_files = sorted(BOOK_RESULTS_DIR.glob("*.json"))
    for file_path in tqdm(result_files, desc=f"Verifying and Building Edges for {args.book_name}"):
        with open(file_path, 'r', encoding='utf-8') as f:
            chapter_data = json.load(f)

        for interaction in chapter_data.get("interactions", []):
            char1_raw = interaction.get("character_1")
            char2_raw = interaction.get("character_2")

            # "Trust, but Verify" step
            char1 = character_mapper.get_canonical_name(char1_raw)
            char2 = character_mapper.get_canonical_name(char2_raw)

            if not (char1 and char2 and char1 != char2):
                continue

            # THE DEFINITIVE FIX: Sanitize the data and provide default values.
            # Use .get() with a default string for every attribute.
            interaction_details = {
                "type": interaction.get("interaction_type", "Unknown"),
                "sentiment": interaction.get("sentiment", "Neutral"),
                "location": interaction.get("location", "Unknown"),
                "evidence": interaction.get("evidence_snippet", "N/A")
            }

            if G.has_edge(char1, char2):
                G[char1][char2]['weight'] += 1
                G[char1][char2]['details'].append(interaction_details)
            else:
                G.add_edge(char1, char2, weight=1, details=[interaction_details])

    # Convert the 'details' list to a JSON string for GML compatibility
    for u, v, data in G.edges(data=True):
        data['details'] = json.dumps(data['details'])

    graph_output_filename = f"{args.book_name}_graph.gml"
    graph_output_path = settings.GRAPH_ARTIFACTS_DIR / graph_output_filename
    settings.GRAPH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    nx.write_gml(G, str(graph_output_path))
    print(f"\nGraph building complete. Graph artifact with rich edge data saved to {graph_output_path}")