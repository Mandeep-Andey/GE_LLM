"""
Evaluation Metrics Module - Computes precision, recall, F1, and agreement metrics.

This module provides tools for evaluating the extraction pipeline against
gold-standard annotations, computing inter-annotator agreement, and
generating quality reports for research publication.
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import math


@dataclass
class AnnotatedInteraction:
    """A single interaction with annotation metadata."""
    character_1: str
    character_2: str
    interaction_type: str
    evidence_snippet: str
    annotator_id: Optional[str] = None
    annotation_source: str = "system"  # "system", "human", "gold"


@dataclass
class EvaluationResult:
    """Results from comparing predictions to gold standard."""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    
    # Detailed breakdowns
    precision_by_type: Dict[str, float] = field(default_factory=dict)
    recall_by_type: Dict[str, float] = field(default_factory=dict)
    f1_by_type: Dict[str, float] = field(default_factory=dict)
    
    # Error analysis
    false_positive_examples: List[AnnotatedInteraction] = field(default_factory=list)
    false_negative_examples: List[AnnotatedInteraction] = field(default_factory=list)


@dataclass
class AgreementMetrics:
    """Inter-annotator agreement metrics."""
    cohens_kappa: float
    percent_agreement: float
    krippendorff_alpha: Optional[float] = None
    
    # Per-category agreement
    agreement_by_type: Dict[str, float] = field(default_factory=dict)


def normalize_interaction(interaction: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Normalize an interaction for comparison.
    
    Returns a tuple of (sorted_characters, interaction_type) where
    characters are sorted alphabetically to handle order-independence.
    """
    chars = tuple(sorted([
        interaction['character_1'].lower().strip(),
        interaction['character_2'].lower().strip()
    ]))
    itype = interaction['interaction_type'].lower().strip()
    return (chars[0], chars[1], itype)


def normalize_interaction_flexible(interaction: Dict[str, Any]) -> Tuple[str, str]:
    """
    Normalize interaction for flexible matching (ignores interaction type).
    
    Use this for evaluating character pair extraction independent of classification.
    """
    chars = tuple(sorted([
        interaction['character_1'].lower().strip(),
        interaction['character_2'].lower().strip()
    ]))
    return chars


class InteractionEvaluator:
    """
    Evaluates extracted interactions against gold standard annotations.
    
    Supports multiple matching strategies:
    - Strict: Characters + interaction type must match
    - Flexible: Only characters must match (ignores type)
    - Evidence-aware: Also compares evidence snippets
    """
    
    def __init__(self, gold_annotations: List[Dict[str, Any]]):
        """
        Initialize with gold standard annotations.
        
        Args:
            gold_annotations: List of interaction dicts from human annotation
        """
        self.gold = gold_annotations
        self.gold_normalized = {normalize_interaction(g) for g in gold_annotations}
        self.gold_flexible = {normalize_interaction_flexible(g) for g in gold_annotations}
    
    def evaluate_strict(self, predictions: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Strict evaluation: both characters AND interaction type must match.
        """
        pred_normalized = {normalize_interaction(p) for p in predictions}
        
        true_positives = len(self.gold_normalized & pred_normalized)
        false_positives = len(pred_normalized - self.gold_normalized)
        false_negatives = len(self.gold_normalized - pred_normalized)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Compute per-type metrics
        precision_by_type, recall_by_type, f1_by_type = self._compute_per_type_metrics(predictions)
        
        # Collect error examples
        fp_examples = [AnnotatedInteraction(**p, annotation_source="prediction") 
                      for p in predictions 
                      if normalize_interaction(p) not in self.gold_normalized][:10]
        
        fn_examples = [AnnotatedInteraction(**g, annotation_source="gold") 
                      for g in self.gold 
                      if normalize_interaction(g) not in pred_normalized][:10]
        
        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision_by_type=precision_by_type,
            recall_by_type=recall_by_type,
            f1_by_type=f1_by_type,
            false_positive_examples=fp_examples,
            false_negative_examples=fn_examples
        )
    
    def evaluate_flexible(self, predictions: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Flexible evaluation: only character pairs must match (ignores type).
        
        Useful for evaluating entity extraction independent of classification.
        """
        pred_flexible = {normalize_interaction_flexible(p) for p in predictions}
        
        true_positives = len(self.gold_flexible & pred_flexible)
        false_positives = len(pred_flexible - self.gold_flexible)
        false_negatives = len(self.gold_flexible - pred_flexible)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives
        )
    
    def _compute_per_type_metrics(self, predictions: List[Dict[str, Any]]) -> Tuple[Dict, Dict, Dict]:
        """Compute precision/recall/F1 broken down by interaction type."""
        types = ["direct dialogue", "physical action", "observation", "memory/reference"]
        
        precision_by_type = {}
        recall_by_type = {}
        f1_by_type = {}
        
        for itype in types:
            gold_of_type = {normalize_interaction(g) for g in self.gold 
                          if g['interaction_type'].lower() == itype}
            pred_of_type = {normalize_interaction(p) for p in predictions 
                          if p['interaction_type'].lower() == itype}
            
            if not gold_of_type and not pred_of_type:
                continue
                
            tp = len(gold_of_type & pred_of_type)
            fp = len(pred_of_type - gold_of_type)
            fn = len(gold_of_type - pred_of_type)
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            
            precision_by_type[itype] = p
            recall_by_type[itype] = r
            f1_by_type[itype] = f
        
        return precision_by_type, recall_by_type, f1_by_type


def compute_cohens_kappa(
    annotations_1: List[Dict[str, Any]],
    annotations_2: List[Dict[str, Any]],
    all_possible_pairs: List[Tuple[str, str]]
) -> float:
    """
    Compute Cohen's Kappa for inter-annotator agreement.
    
    Args:
        annotations_1: First annotator's interactions
        annotations_2: Second annotator's interactions  
        all_possible_pairs: List of all character pairs that could be annotated
        
    Returns:
        Cohen's Kappa coefficient (-1 to 1, where 1 is perfect agreement)
    """
    set_1 = {normalize_interaction_flexible(a) for a in annotations_1}
    set_2 = {normalize_interaction_flexible(a) for a in annotations_2}
    
    # For each possible pair, both annotators make a binary decision (interaction exists or not)
    agreements = 0
    total = len(all_possible_pairs)
    
    if total == 0:
        return 0.0
    
    p_yes_1 = len(set_1) / total  # Proportion annotator 1 says "yes"
    p_yes_2 = len(set_2) / total  # Proportion annotator 2 says "yes"
    
    for pair in all_possible_pairs:
        in_1 = pair in set_1
        in_2 = pair in set_2
        if in_1 == in_2:
            agreements += 1
    
    p_observed = agreements / total
    
    # Expected agreement by chance
    p_expected = (p_yes_1 * p_yes_2) + ((1 - p_yes_1) * (1 - p_yes_2))
    
    if p_expected == 1:
        return 1.0  # Perfect agreement
    
    kappa = (p_observed - p_expected) / (1 - p_expected)
    return kappa


def compute_agreement_metrics(
    annotator_results: Dict[str, List[Dict[str, Any]]],
    all_possible_pairs: List[Tuple[str, str]]
) -> AgreementMetrics:
    """
    Compute inter-annotator agreement across multiple annotators.
    
    Args:
        annotator_results: Dict mapping annotator_id -> list of interactions
        all_possible_pairs: All character pairs that could be annotated
        
    Returns:
        AgreementMetrics with kappa and percent agreement
    """
    annotator_ids = list(annotator_results.keys())
    
    if len(annotator_ids) < 2:
        return AgreementMetrics(cohens_kappa=1.0, percent_agreement=1.0)
    
    # Compute pairwise kappas and average
    kappas = []
    for i, ann1 in enumerate(annotator_ids):
        for ann2 in annotator_ids[i+1:]:
            kappa = compute_cohens_kappa(
                annotator_results[ann1],
                annotator_results[ann2],
                all_possible_pairs
            )
            kappas.append(kappa)
    
    avg_kappa = sum(kappas) / len(kappas) if kappas else 0
    
    # Compute percent agreement (at least N-1 annotators agree)
    pair_counts = defaultdict(int)
    for annotations in annotator_results.values():
        for ann in annotations:
            pair = normalize_interaction_flexible(ann)
            pair_counts[pair] += 1
    
    n_annotators = len(annotator_ids)
    agreements = sum(1 for count in pair_counts.values() if count >= n_annotators - 1)
    total_unique = len(pair_counts)
    percent_agreement = agreements / total_unique if total_unique > 0 else 1.0
    
    return AgreementMetrics(
        cohens_kappa=avg_kappa,
        percent_agreement=percent_agreement
    )


def generate_evaluation_report(
    result: EvaluationResult,
    agreement: Optional[AgreementMetrics] = None
) -> str:
    """
    Generate a human-readable evaluation report.
    """
    lines = [
        "=" * 60,
        "INTERACTION EXTRACTION EVALUATION REPORT",
        "=" * 60,
        "",
        "## Overall Metrics",
        f"  Precision:       {result.precision:.4f}",
        f"  Recall:          {result.recall:.4f}",
        f"  F1 Score:        {result.f1_score:.4f}",
        "",
        f"  True Positives:  {result.true_positives}",
        f"  False Positives: {result.false_positives}",
        f"  False Negatives: {result.false_negatives}",
        "",
    ]
    
    if result.f1_by_type:
        lines.extend([
            "## Per-Type Metrics",
            f"  {'Type':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}",
            "  " + "-" * 50,
        ])
        for itype in result.f1_by_type:
            p = result.precision_by_type.get(itype, 0)
            r = result.recall_by_type.get(itype, 0)
            f = result.f1_by_type.get(itype, 0)
            lines.append(f"  {itype:<20} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
        lines.append("")
    
    if agreement:
        lines.extend([
            "## Inter-Annotator Agreement",
            f"  Cohen's Kappa:     {agreement.cohens_kappa:.4f}",
            f"  Percent Agreement: {agreement.percent_agreement:.4f}",
            "",
        ])
    
    if result.false_positive_examples:
        lines.extend([
            "## Sample False Positives (System extracted, not in gold)",
        ])
        for ex in result.false_positive_examples[:5]:
            lines.append(f"  - {ex.character_1} <-> {ex.character_2} ({ex.interaction_type})")
        lines.append("")
    
    if result.false_negative_examples:
        lines.extend([
            "## Sample False Negatives (In gold, system missed)",
        ])
        for ex in result.false_negative_examples[:5]:
            lines.append(f"  - {ex.character_1} <-> {ex.character_2} ({ex.interaction_type})")
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


# --- File I/O utilities ---

def load_gold_annotations(file_path: str) -> List[Dict[str, Any]]:
    """Load gold annotations from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('interactions', data) if isinstance(data, dict) else data


def save_evaluation_report(result: EvaluationResult, output_path: str):
    """Save evaluation report to file."""
    report = generate_evaluation_report(result)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)


def export_for_error_analysis(result: EvaluationResult, output_path: str):
    """Export detailed error analysis to JSON for further investigation."""
    data = {
        'metrics': {
            'precision': result.precision,
            'recall': result.recall,
            'f1_score': result.f1_score,
        },
        'false_positives': [
            {
                'character_1': ex.character_1,
                'character_2': ex.character_2,
                'interaction_type': ex.interaction_type,
                'evidence_snippet': ex.evidence_snippet
            }
            for ex in result.false_positive_examples
        ],
        'false_negatives': [
            {
                'character_1': ex.character_1,
                'character_2': ex.character_2,
                'interaction_type': ex.interaction_type,
                'evidence_snippet': ex.evidence_snippet
            }
            for ex in result.false_negative_examples
        ]
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
