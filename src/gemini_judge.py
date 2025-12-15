"""
Gemini Judge LLM - Validates and scores extracted character interactions.

This module implements the "LLM-as-Judge" pattern using Google's Gemini API
to provide a second opinion on extracted interactions, enabling quality filtering
and confidence scoring.
"""

import google.generativeai as genai
import json
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Verdict(str, Enum):
    """Possible verdicts from the judge."""
    ACCEPT = "accept"           # Interaction is valid
    REJECT = "reject"           # Interaction is invalid/hallucinated
    NEEDS_REVIEW = "needs_review"  # Uncertain, flag for human review


class JudgeEvaluation(BaseModel):
    """Structured evaluation from the judge LLM."""
    verdict: Verdict
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    factuality_score: float = Field(..., ge=0.0, le=1.0, description="Does evidence support the interaction?")
    character_validity_score: float = Field(..., ge=0.0, le=1.0, description="Are characters plausibly present?")
    type_accuracy_score: float = Field(..., ge=0.0, le=1.0, description="Is interaction type correct?")
    reasoning: str = Field(..., description="Brief explanation for the verdict")
    suggested_correction: Optional[str] = Field(None, description="Suggested fix if needs_review")


@dataclass
class InteractionToJudge:
    """Input structure for judging an interaction."""
    character_1: str
    character_2: str
    interaction_type: str
    evidence_snippet: str
    surrounding_context: str  # The paragraph/chunk the interaction was extracted from


class GeminiJudge:
    """
    Judge LLM using Google's Gemini API.
    
    Implements rate limiting and batch processing to work within free tier limits.
    Free tier: 15 RPM (requests per minute), 1M tokens/day
    """
    
    # Rate limiting settings for free tier
    REQUESTS_PER_MINUTE = 15
    MIN_REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE  # 4 seconds between requests
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        """
        Initialize the Gemini Judge.
        
        Args:
            api_key: Google AI API key
            model_name: Gemini model to use (gemini-1.5-pro recommended for judging)
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.last_request_time = 0
        logger.info(f"Gemini Judge initialized with model: {model_name}")
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            sleep_time = self.MIN_REQUEST_INTERVAL - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _create_judge_prompt(self, interactions: List[InteractionToJudge]) -> str:
        """Create the evaluation prompt for the judge."""
        
        interactions_json = json.dumps([asdict(i) for i in interactions], indent=2)
        
        return f"""You are an expert literary analyst acting as a judge to validate character interactions extracted from George Eliot's novel "Middlemarch".

## YOUR TASK
Evaluate each extracted interaction and determine if it is:
1. **ACCEPT** - The interaction is valid and accurately extracted
2. **REJECT** - The interaction is invalid, hallucinated, or misattributed  
3. **NEEDS_REVIEW** - Uncertain; requires human verification

## EVALUATION CRITERIA
For each interaction, score these dimensions (0.0 to 1.0):

1. **Factuality** (40% weight): Does the evidence_snippet actually support this interaction occurring?
   - 1.0 = Clear, direct evidence
   - 0.5 = Implied or indirect evidence
   - 0.0 = No evidence or contradicted by text

2. **Character Validity** (20% weight): Are both characters plausibly present in this scene?
   - 1.0 = Both explicitly mentioned nearby
   - 0.5 = One mentioned, other implied
   - 0.0 = Character couldn't be in this scene

3. **Type Accuracy** (40% weight): Is the interaction_type classification correct?
   - "Direct Dialogue" = Characters speaking to each other
   - "Physical Action" = Physical contact or action toward each other
   - "Observation" = One character watching/noticing another
   - "Memory/Reference" = One character thinking about/mentioning another

## INTERACTIONS TO EVALUATE
{interactions_json}

## RESPONSE FORMAT
Respond with a JSON array. For each interaction (in order), provide:
```json
[
  {{
    "verdict": "accept|reject|needs_review",
    "confidence_score": 0.0-1.0,
    "factuality_score": 0.0-1.0,
    "character_validity_score": 0.0-1.0,
    "type_accuracy_score": 0.0-1.0,
    "reasoning": "Brief explanation",
    "suggested_correction": null or "Suggested fix if needs_review"
  }}
]
```

Respond ONLY with the JSON array, no other text.
"""

    def judge_single(self, interaction: InteractionToJudge) -> Optional[JudgeEvaluation]:
        """
        Evaluate a single interaction.
        
        Args:
            interaction: The interaction to evaluate
            
        Returns:
            JudgeEvaluation or None if the request failed
        """
        results = self.judge_batch([interaction])
        return results[0] if results else None
    
    def judge_batch(self, interactions: List[InteractionToJudge], 
                    batch_size: int = 10) -> List[Optional[JudgeEvaluation]]:
        """
        Evaluate a batch of interactions.
        
        Processes in sub-batches to stay within context limits and reduce API calls.
        
        Args:
            interactions: List of interactions to evaluate
            batch_size: Number of interactions per API call (default 10)
            
        Returns:
            List of JudgeEvaluations (None for failed evaluations)
        """
        all_results = []
        
        for i in range(0, len(interactions), batch_size):
            batch = interactions[i:i + batch_size]
            logger.info(f"Judging batch {i//batch_size + 1}/{(len(interactions) + batch_size - 1)//batch_size}")
            
            self._rate_limit()
            
            try:
                prompt = self._create_judge_prompt(batch)
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.1,  # Low temperature for consistent judgments
                    )
                )
                
                # Parse response
                evaluations_raw = json.loads(response.text)
                
                for eval_dict in evaluations_raw:
                    try:
                        # Convert verdict string to enum
                        eval_dict['verdict'] = Verdict(eval_dict['verdict'].lower())
                        evaluation = JudgeEvaluation(**eval_dict)
                        all_results.append(evaluation)
                    except Exception as e:
                        logger.warning(f"Failed to parse evaluation: {e}")
                        all_results.append(None)
                        
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                # Add None for each interaction in this failed batch
                all_results.extend([None] * len(batch))
        
        return all_results
    
    def compute_aggregate_score(self, evaluation: JudgeEvaluation) -> float:
        """
        Compute weighted aggregate score from individual dimension scores.
        
        Weights: Factuality 40%, Character Validity 20%, Type Accuracy 40%
        """
        return (
            evaluation.factuality_score * 0.4 +
            evaluation.character_validity_score * 0.2 +
            evaluation.type_accuracy_score * 0.4
        )


class JudgedInteraction(BaseModel):
    """An interaction with its judge evaluation attached."""
    # Original interaction fields
    character_1: str
    character_2: str
    interaction_type: str
    evidence_snippet: str
    
    # Judge evaluation fields
    verdict: Verdict
    confidence_score: float
    aggregate_score: float
    reasoning: str
    
    # Metadata
    needs_human_review: bool = False


def filter_by_verdict(
    interactions: List[JudgedInteraction],
    accept_threshold: float = 0.7,
    reject_threshold: float = 0.3
) -> Dict[str, List[JudgedInteraction]]:
    """
    Filter interactions into accept/reject/review buckets based on scores.
    
    Args:
        interactions: List of judged interactions
        accept_threshold: Minimum aggregate score to auto-accept
        reject_threshold: Maximum aggregate score to auto-reject
        
    Returns:
        Dictionary with 'accepted', 'rejected', 'needs_review' lists
    """
    result = {
        'accepted': [],
        'rejected': [],
        'needs_review': []
    }
    
    for interaction in interactions:
        if interaction.verdict == Verdict.REJECT or interaction.aggregate_score < reject_threshold:
            result['rejected'].append(interaction)
        elif interaction.verdict == Verdict.ACCEPT and interaction.aggregate_score >= accept_threshold:
            result['accepted'].append(interaction)
        else:
            interaction.needs_human_review = True
            result['needs_review'].append(interaction)
    
    return result


# --- Convenience function for integration with existing pipeline ---

def judge_chapter_results(
    judge: GeminiJudge,
    chapter_interactions: List[Dict[str, Any]],
    chapter_text: str
) -> List[JudgedInteraction]:
    """
    Convenience function to judge all interactions from a chapter.
    
    Args:
        judge: GeminiJudge instance
        chapter_interactions: List of interaction dicts from LLM extraction
        chapter_text: The original chapter text for context
        
    Returns:
        List of JudgedInteraction objects
    """
    # Convert to InteractionToJudge format
    to_judge = []
    for interaction in chapter_interactions:
        to_judge.append(InteractionToJudge(
            character_1=interaction['character_1'],
            character_2=interaction['character_2'],
            interaction_type=interaction['interaction_type'],
            evidence_snippet=interaction['evidence_snippet'],
            surrounding_context=chapter_text[:2000]  # First 2000 chars as context
        ))
    
    # Get evaluations
    evaluations = judge.judge_batch(to_judge)
    
    # Combine into JudgedInteraction objects
    judged = []
    for interaction, evaluation in zip(chapter_interactions, evaluations):
        if evaluation:
            judged.append(JudgedInteraction(
                character_1=interaction['character_1'],
                character_2=interaction['character_2'],
                interaction_type=interaction['interaction_type'],
                evidence_snippet=interaction['evidence_snippet'],
                verdict=evaluation.verdict,
                confidence_score=evaluation.confidence_score,
                aggregate_score=judge.compute_aggregate_score(evaluation),
                reasoning=evaluation.reasoning
            ))
    
    return judged
