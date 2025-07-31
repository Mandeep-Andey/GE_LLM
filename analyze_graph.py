import json
from pathlib import Path
from typing import List, Tuple, Dict
import networkx as nx
from tqdm import tqdm
import argparse
import sys
from collections import defaultdict
import shutil

from src.settings import Settings
from src.character_mapper import CharacterMapper
from src.graph_manager import GraphManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build, analyze, and report on a character network graph.")
    parser.add_argument("book_name", type=str, help="The name of the book to process (e.g., 'book_1').")
    args = parser.parse_args()

    settings = Settings(config_path="config.yaml")
    print(f"\n--- Graph Analysis Pipeline Started for Book: {args.book_name} ---")

    BOOK_RESULTS_DIR = settings.RESULTS_DIR / args.book_name
    if not BOOK_RESULTS_DIR.exists():
        print(f"FATAL: No results found for '{args.book_name}'. Please run 'run_llm_extraction.py' first.")
        sys.exit(1)

    # --- 1. BUILD GRAPH DATA FROM LLM RESULTS ---
    print("\n--- Phase 1: Verifying and Building Edges ---")
    character_mapper = CharacterMapper(file_path=str(settings.CHARACTER_FILE))

    # THE CHANGE: Store edges on a per-chapter basis
    all_edges_by_chapter: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    result_files = sorted(BOOK_RESULTS_DIR.glob("*.json"))
    for file_path in tqdm(result_files, desc=f"Processing chapters for {args.book_name}"):
        with open(file_path, 'r', encoding='utf-8') as f:
            chapter_data = json.load(f)

        chapter_index = int(file_path.stem.split('_')[-1])

        for interaction in chapter_data.get("interactions", []):
            char1_raw = interaction.get("character_1")
            char2_raw = interaction.get("character_2")
            char1 = character_mapper.get_canonical_name(char1_raw)
            char2 = character_mapper.get_canonical_name(char2_raw)

            if char1 and char2 and char1 != char2:
                all_edges_by_chapter[chapter_index].append(tuple(sorted((char1, char2))))

    # --- 2. INITIALIZE GRAPH MANAGER WITH AGGREGATED DATA ---
    # Flatten the chapter data into a single list for the main graph
    all_edges_flat = [edge for chapter_edges in all_edges_by_chapter.values() for edge in chapter_edges]
    graph_manager = GraphManager(all_edges_flat)

    # --- 3. GENERATE AND SAVE REPORTS ---
    print("\n--- Phase 2: Generating Analysis Reports ---")
    BOOK_REPORT_DIR = Path("./analysis_reports") / args.book_name
    if BOOK_REPORT_DIR.exists():
        shutil.rmtree(BOOK_REPORT_DIR)
    BOOK_REPORT_DIR.mkdir(parents=True)

    # Generate the main, book-level report
    full_report_text = graph_manager.generate_full_analysis_report(top_n=settings.TOP_N_ANALYSIS)

    # Generate the new chapter-wise report
    chapter_report_text = graph_manager.generate_chapter_wise_report(all_edges_by_chapter)

    # Combine and save the reports
    final_report = f"{full_report_text}\n\n{chapter_report_text}"
    report_path = BOOK_REPORT_DIR / "analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(final_report)
    print(f"Detailed text analysis saved to {report_path}")
    print("\n" + final_report)

    # --- 4. GENERATE VISUALIZATION ---
    viz_filename = f"{args.book_name}_network.html"
    viz_path = BOOK_REPORT_DIR / viz_filename
    graph_manager.save_interactive_visualization(output_path=viz_path)

    print(f"\nAnalysis complete. All reports are in the '{BOOK_REPORT_DIR}' directory.")