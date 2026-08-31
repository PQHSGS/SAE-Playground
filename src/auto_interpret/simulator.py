from typing import List, Tuple
import math
from src.auto_interpret.sample_collector import ActivatingSnippet


class FeatureSimulator:
    """
    Evaluates explanation quality by measuring correlation between
    simulated activations and actual dictionary activations.
    """

    @staticmethod
    def compute_faithfulness_score(
        actual_activations: List[float],
        predicted_scores: List[float],
    ) -> float:
        """
        Computes Spearman/Pearson rank correlation between predicted score and actual activation.
        """
        if len(actual_activations) < 2:
            return 0.0

        n = len(actual_activations)
        mean_act = sum(actual_activations) / n
        mean_pred = sum(predicted_scores) / n

        num = sum((a - mean_act) * (p - mean_pred) for a, p in zip(actual_activations, predicted_scores))
        denom_a = math.sqrt(sum((a - mean_act) ** 2 for a in actual_activations))
        denom_p = math.sqrt(sum((p - mean_pred) ** 2 for p in predicted_scores))

        if denom_a * denom_p == 0:
            return 0.0

        return max(0.0, num / (denom_a * denom_p))
