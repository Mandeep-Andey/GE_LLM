#!/bin/bash

# --- Master Analysis Script ---
# This script builds the graph artifacts and generates the final analysis
# for all books that have been processed by `run_llm_extraction.py`.
# It assumes you are running it from the project's root directory.

# Exit immediately if a command fails
set -e

# --- Configuration ---
# The directory where the `run_llm_extraction.py` script saves its results.
NLP_RESULTS_DIR="./llm_results"
# The command to run your python scripts (e.g., "python" or "uv run")
RUN_COMMAND="uv run"

# --- Main Logic ---
echo "--- Starting Full Analysis Pipeline for All Books ---"

# Check if the NLP results directory exists
if [ ! -d "$NLP_RESULTS_DIR" ]; then
    echo "ERROR: The directory '$NLP_RESULTS_DIR' was not found."
    echo "Please run the 'run_llm_extraction.py' script first."
    exit 1
fi

# Find all the book subdirectories (e.g., book_1, book_2) that have been processed
# and loop through them.
for book_dir in "$NLP_RESULTS_DIR"/*/; do
    # Check if it's a directory
    if [ -d "$book_dir" ]; then
        # Get just the name of the directory (e.g., "book_1")
        book_name=$(basename "$book_dir")

        echo ""
        echo "======================================================"
        echo "               Processing: $book_name"
        echo "======================================================"

        # --- Phase 1: Build the Graph Artifact ---
        echo "\n--> Step 1: Building graph artifact for $book_name..."
        $RUN_COMMAND build_graph.py "$book_name"

        # --- Phase 2: Analyze the Graph and Generate Reports ---
        echo "\n--> Step 2: Analyzing graph and generating reports for $book_name..."
        $RUN_COMMAND analyze_graph.py "$book_name"
    fi
done

echo ""
echo "--- Full Analysis Pipeline Complete ---"
echo "All reports and visualizations have been generated in the 'analysis_reports' directory."