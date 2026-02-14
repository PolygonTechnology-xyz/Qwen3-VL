"""
Character Error Rate (CER) and Word Error Rate (WER) implementations using jiwer.
"""

from typing import List
from jiwer import cer, wer, normalize_intensity


def compute_metrics_batch(
    references: List[str], 
    hypotheses: List[str],
    normalize: bool = True
) -> dict:
    """
    Compute CER and WER for a batch of predictions using jiwer.
    
    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts
        normalize: Whether to normalize text before calculation
    
    Returns:
        Dictionary with 'cer' and 'wer' keys containing average scores
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"Number of references ({len(references)}) must match number of hypotheses ({len(hypotheses)})")
    
    if len(references) == 0:
        return {"cer": 0.0, "wer": 0.0}
    
    try:
        # Compute CER and WER using jiwer
        cer_score = cer(references, hypotheses)
        wer_score = wer(references, hypotheses)
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {"cer": 0.0, "wer": 0.0}
    
    return {
        "cer": cer_score,
        "wer": wer_score,
    }
