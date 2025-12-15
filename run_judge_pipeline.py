#!/usr/bin/env python3
"""
Run Judge Pipeline - Validates extracted interactions using Gemini as a judge LLM.

This script takes the raw LLM extraction results and passes them through
the Gemini judge for quality scoring and filtering.

Usage:
    uv run run_judge_pipeline.py book_1
    uv run run_judge_pipeline.py book_1 --sample-rate 0.2  # Judge 20% sample
    uv run run_judge_pipeline.py book_1 --threshold 0.7     # Accept threshold
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import logging

from src.settings import Settings
from src.gemini_judge import (
    GeminiJudge, 
    InteractionToJudge, 
    JudgedInteraction,
    Verdict,
    filter_by_verdict
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_chapter_text(books_dir: Path, book_name: str, chapter_idx: int) -> str:
    """Load the original chapter text for context."""
    # Try to load from the book file and extract the chapter
    book_file = books_dir / f"{book_name}.txt"
    if not book_file.exists():
        return ""
    
    import re
    with open(book_file, 'r', encoding='utf-8') as f:
        book_text = f.read()
    
    chapter_pattern = re.compile(r'^\\s*Chapter\\s*\\d+\\s*', re.MULTILINE)
    chapters = chapter_pattern.split(book_text)[1:]
    
    if chapter_idx < len(chapters):
        return chapters[chapter_idx][:3000]  # First 3000 chars
    return ""


def run_judge_pipeline(
    book_name: str,
    sample_rate: float = 1.0,
    accept_threshold: float = 0.7,
    reject_threshold: float = 0.3
):
    """
    Run the judge pipeline on extracted interactions.
    
    Args:
        book_name: Name of the book to process (e.g., 'book_1')
        sample_rate: Fraction of interactions to judge (0.0-1.0)
        accept_threshold: Minimum score to auto-accept
        reject_threshold: Maximum score to auto-reject
    """
    # Load settings
    settings = Settings(config_path="config.yaml")
    
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set!")
        logger.info("Set it with: export GEMINI_API_KEY='your-api-key'")
        return
    
    # Initialize judge
    judge = GeminiJudge(api_key=api_key, model_name="gemini-1.5-pro")
    
    # Setup paths
    results_dir = settings.RESULTS_DIR / book_name
    judged_dir = settings.RESULTS_DIR / f"{book_name}_judged"
    judged_dir.mkdir(exist_ok=True)
    
    if not results_dir.exists():
        logger.error(f"No results found for '{book_name}'. Run extraction first.")
        return
    
    # Process each chapter
    result_files = sorted(results_dir.glob("*.json"))
    
    all_stats = {
        'total_interactions': 0,
        'judged_interactions': 0,
        'accepted': 0,
        'rejected': 0,
        'needs_review': 0,
        'avg_confidence': 0.0,
        'avg_aggregate_score': 0.0
    }
    
    for file_path in tqdm(result_files, desc=f"Judging {book_name}"):
        logger.info(f"Processing {file_path.name}")
        
        # Load interactions
        with open(file_path, 'r', encoding='utf-8') as f:
            chapter_data = json.load(f)
        
        interactions = chapter_data.get('interactions', [])
        all_stats['total_interactions'] += len(interactions)
        
        if not interactions:
            continue
        
        # Sample if requested
        import random
        if sample_rate < 1.0:
            sample_size = max(1, int(len(interactions) * sample_rate))
            sampled_indices = set(random.sample(range(len(interactions)), sample_size))
            interactions_to_judge = [interactions[i] for i in sampled_indices]
        else:
            interactions_to_judge = interactions
            sampled_indices = set(range(len(interactions)))
        
        # Get chapter text for context
        chapter_idx = int(file_path.stem.split('_')[-1])
        chapter_text = load_chapter_text(settings.BOOKS_DIR, book_name, chapter_idx)
        
        # Prepare for judging
        to_judge = [
            InteractionToJudge(
                character_1=i['character_1'],
                character_2=i['character_2'],
                interaction_type=i['interaction_type'],
                evidence_snippet=i['evidence_snippet'],
                surrounding_context=chapter_text
            )
            for i in interactions_to_judge
        ]
        
        # Judge interactions
        evaluations = judge.judge_batch(to_judge, batch_size=10)
        
        # Build judged interactions
        judged_interactions = []
        confidence_scores = []
        aggregate_scores = []
        
        for interaction, evaluation in zip(interactions_to_judge, evaluations):
            if evaluation:
                agg_score = judge.compute_aggregate_score(evaluation)
                judged = JudgedInteraction(
                    character_1=interaction['character_1'],
                    character_2=interaction['character_2'],
                    interaction_type=interaction['interaction_type'],
                    evidence_snippet=interaction['evidence_snippet'],
                    verdict=evaluation.verdict,
                    confidence_score=evaluation.confidence_score,
                    aggregate_score=agg_score,
                    reasoning=evaluation.reasoning
                )
                judged_interactions.append(judged)
                confidence_scores.append(evaluation.confidence_score)
                aggregate_scores.append(agg_score)
        
        # Filter by verdict
        filtered = filter_by_verdict(
            judged_interactions,
            accept_threshold=accept_threshold,
            reject_threshold=reject_threshold
        )
        
        # Update stats
        all_stats['judged_interactions'] += len(judged_interactions)
        all_stats['accepted'] += len(filtered['accepted'])
        all_stats['rejected'] += len(filtered['rejected'])
        all_stats['needs_review'] += len(filtered['needs_review'])
        
        if confidence_scores:
            all_stats['avg_confidence'] += sum(confidence_scores)
            all_stats['avg_aggregate_score'] += sum(aggregate_scores)
        
        # Save judged results
        output_data = {
            'chapter': file_path.stem,
            'sample_rate': sample_rate,
            'total_original': len(chapter_data.get('interactions', [])),
            'total_judged': len(judged_interactions),
            'accepted': [j.model_dump() for j in filtered['accepted']],
            'rejected': [j.model_dump() for j in filtered['rejected']],
            'needs_review': [j.model_dump() for j in filtered['needs_review']]
        }
        
        output_path = judged_dir / file_path.name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str)
    
    # Finalize stats
    if all_stats['judged_interactions'] > 0:
        all_stats['avg_confidence'] /= all_stats['judged_interactions']
        all_stats['avg_aggregate_score'] /= all_stats['judged_interactions']
    
    # Print summary
    print("\n" + "=" * 60)
    print("JUDGE PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Book:                  {book_name}")
    print(f"Sample Rate:           {sample_rate:.0%}")
    print(f"Accept Threshold:      {accept_threshold}")
    print(f"Reject Threshold:      {reject_threshold}")
    print("-" * 60)
    print(f"Total Interactions:    {all_stats['total_interactions']}")
    print(f"Judged Interactions:   {all_stats['judged_interactions']}")
    print(f"  - Accepted:          {all_stats['accepted']} ({all_stats['accepted']/max(1,all_stats['judged_interactions']):.1%})")
    print(f"  - Rejected:          {all_stats['rejected']} ({all_stats['rejected']/max(1,all_stats['judged_interactions']):.1%})")
    print(f"  - Needs Review:      {all_stats['needs_review']} ({all_stats['needs_review']/max(1,all_stats['judged_interactions']):.1%})")
    print("-" * 60)
    print(f"Avg Confidence Score:  {all_stats['avg_confidence']:.4f}")
    print(f"Avg Aggregate Score:   {all_stats['avg_aggregate_score']:.4f}")
    print("=" * 60)
    print(f"\nJudged results saved to: {judged_dir}")
    
    # Save summary stats
    stats_path = judged_dir / "summary_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Gemini judge on extracted interactions"
    )
    parser.add_argument(
        "book_name",
        type=str,
        help="Name of the book to process (e.g., 'book_1')"
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Fraction of interactions to judge (default: 1.0 = all)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Accept threshold for aggregate score (default: 0.7)"
    )
    parser.add_argument(
        "--reject-threshold",
        type=float,
        default=0.3,
        help="Reject threshold for aggregate score (default: 0.3)"
    )
    
    args = parser.parse_args()
    
    run_judge_pipeline(
        book_name=args.book_name,
        sample_rate=args.sample_rate,
        accept_threshold=args.threshold,
        reject_threshold=args.reject_threshold
    )
