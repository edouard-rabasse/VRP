from .vrp_instance import VRPInstance
from .iteration_result import IterationResult
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class ClassificationCriteria:
    """Configuration for classification criteria."""

    classifier_score_threshold: Optional[float] = None
    entropy_score_threshold: Optional[float] = None
    max_violations: Optional[int] = None
    min_cost_improvement: Optional[float] = None
    max_iterations: Optional[int] = None
    require_valid: bool = True

    # Advanced criteria
    entropy_trend_threshold: Optional[float] = None  # Rate of entropy decrease
    classifier_trend_threshold: Optional[float] = None  # Rate of classifier improvement
    cost_plateau_tolerance: Optional[float] = (
        None  # Stop if cost hasn't improved by this much
    )

    # Combination logic
    combination_mode: str = "AND"  # "AND", "OR", "WEIGHTED"
    weights: Optional[Dict[str, float]] = None


@dataclass
class ClassificationResult:
    """Result of VRP instance classification."""

    iteration_result: Optional[IterationResult] = None
    satisfied_criteria: list = None
    classification_score: float = 0.0
    reason: str = ""

    @property
    def valid(self) -> bool:
        """Whether a valid iteration was found."""
        return self.iteration_result is not None

    @property
    def first_time_valid(self) -> bool:
        """Whether this is the first time valid iteration."""
        return self.iteration_result is not None and getattr(
            self.iteration_result, "first_time_valid", False
        )


class EnhancedVRPClassifier:
    """Enhanced classifier for VRP instances with multiple criteria."""

    def __init__(self, criteria: ClassificationCriteria = None):
        self.criteria = criteria or ClassificationCriteria()

        # Predefined classifier configurations
        self.presets = {
            "aggressive": ClassificationCriteria(
                classifier_score_threshold=0.5,
                entropy_score_threshold=4,
                max_iterations=50,
                require_valid=True,
                combination_mode="OR",
            ),
        }

    def classify(
        self, instance: VRPInstance, criteria: ClassificationCriteria = None
    ) -> ClassificationResult:
        """Classify VRP instance using specified criteria."""
        criteria = criteria or self.criteria

        best_iteration = None
        best_score = -np.inf
        satisfied_criteria_list = []

        for i, iteration in enumerate(instance.iterations):
            # Check individual criteria
            criteria_results = self._evaluate_criteria(iteration, instance, i, criteria)

            # Combine criteria based on mode
            if criteria.combination_mode == "AND":
                satisfies_all = all(criteria_results.values())
                if satisfies_all and (
                    best_iteration is None or iteration.iter < best_iteration.iter
                ):
                    best_iteration = iteration
                    satisfied_criteria_list = [
                        k for k, v in criteria_results.items() if v
                    ]
                    break

            elif criteria.combination_mode == "OR":
                satisfies_any = any(criteria_results.values())
                if satisfies_any:
                    score = sum(criteria_results.values())
                    if score > best_score:
                        best_score = score
                        best_iteration = iteration
                        satisfied_criteria_list = [
                            k for k, v in criteria_results.items() if v
                        ]
                    break

        reason = self._generate_reason(satisfied_criteria_list, best_iteration)

        return ClassificationResult(
            iteration_result=best_iteration,
            satisfied_criteria=satisfied_criteria_list,
            classification_score=best_score if best_score != -np.inf else 0.0,
            reason=reason,
        )

    def _evaluate_criteria(
        self,
        iteration: IterationResult,
        instance: VRPInstance,
        iter_idx: int,
        criteria: ClassificationCriteria,
    ) -> Dict[str, bool]:
        """Evaluate all criteria for a single iteration."""
        results = {}

        # Basic threshold criteria
        if criteria.classifier_score_threshold is not None:
            results["classifier_score"] = (
                iteration.classifier_score is not None
                and iteration.classifier_score <= criteria.classifier_score_threshold
            )

        if criteria.entropy_score_threshold is not None:
            results["entropy_score"] = (
                iteration.entropy_score is not None
                and iteration.entropy_score <= criteria.entropy_score_threshold
            )

        if criteria.max_violations is not None:
            results["violations"] = (
                iteration.number_of_violations is not None
                and iteration.number_of_violations <= criteria.max_violations
            )

        if criteria.max_iterations is not None:
            results["max_iterations"] = iteration.iter <= criteria.max_iterations

        if criteria.require_valid:
            results["valid"] = iteration.valid

        # # Advanced trend-based criteria
        # if criteria.entropy_trend_threshold is not None:
        #     entropy_trend = self._calculate_entropy_trend(instance, iter_idx)
        #     results["entropy_trend"] = entropy_trend <= criteria.entropy_trend_threshold

        # if criteria.classifier_trend_threshold is not None:
        #     classifier_trend = self._calculate_classifier_trend(instance, iter_idx)
        #     results["classifier_trend"] = (
        #         classifier_trend >= criteria.classifier_trend_threshold
        #     )

        # if criteria.cost_plateau_tolerance is not None:
        #     cost_improvement = self._calculate_cost_improvement(instance, iter_idx)
        #     results["cost_plateau"] = (
        #         cost_improvement >= criteria.cost_plateau_tolerance
        # )

        return results

    # def _calculate_entropy_trend(
    #     self, instance: VRPInstance, iter_idx: int, window: int = 3
    # ) -> float:
    #     """Calculate entropy trend over recent iterations."""
    #     if iter_idx < window:
    #         return 0.0

    #     recent_entropies = []
    #     for i in range(max(0, iter_idx - window + 1), iter_idx + 1):
    #         if (
    #             i < len(instance.iterations)
    #             and instance.iterations[i].entropy_score is not None
    #         ):
    #             recent_entropies.append(instance.iterations[i].entropy_score)

    #     if len(recent_entropies) < 2:
    #         return 0.0

    #     # Simple linear trend (could use regression for more accuracy)
    #     return (recent_entropies[-1] - recent_entropies[0]) / len(recent_entropies)

    # def _calculate_classifier_trend(
    #     self, instance: VRPInstance, iter_idx: int, window: int = 3
    # ) -> float:
    #     """Calculate classifier score trend over recent iterations."""
    #     if iter_idx < window:
    #         return 0.0

    #     recent_scores = []
    #     for i in range(max(0, iter_idx - window + 1), iter_idx + 1):
    #         if (
    #             i < len(instance.iterations)
    #             and instance.iterations[i].classifier_score is not None
    #         ):
    #             recent_scores.append(instance.iterations[i].classifier_score)

    #     if len(recent_scores) < 2:
    #         return 0.0

    #     # Improvement means decreasing classifier score (lower is better)
    #     return (recent_scores[0] - recent_scores[-1]) / len(recent_scores)

    # def _calculate_cost_improvement(
    #     self, instance: VRPInstance, iter_idx: int, window: int = 5
    # ) -> float:
    #     """Calculate cost improvement over recent iterations."""
    #     if iter_idx < window:
    #         return 1.0  # Assume good improvement early on

    #     recent_costs = []
    #     for i in range(max(0, iter_idx - window + 1), iter_idx + 1):
    #         if (
    #             i < len(instance.iterations)
    #             and instance.iterations[i].solver_cost is not None
    #         ):
    #             recent_costs.append(instance.iterations[i].solver_cost)

    #     if len(recent_costs) < 2:
    #         return 1.0

    #     # Calculate relative improvement
    #     initial_cost = recent_costs[0]
    #     final_cost = recent_costs[-1]

    #     if initial_cost == 0:
    #         return 0.0

    #     return (initial_cost - final_cost) / initial_cost

    def _calculate_weighted_score(
        self, criteria_results: Dict[str, bool], weights: Dict[str, float]
    ) -> float:
        """Calculate weighted score for criteria combination."""
        total_score = 0.0
        total_weight = 0.0

        for criterion, satisfied in criteria_results.items():
            weight = weights.get(criterion, 1.0)
            if satisfied:
                total_score += weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _generate_reason(
        self, satisfied_criteria: list, best_iteration: Optional[IterationResult]
    ) -> str:
        """Generate human-readable reason for classification."""
        if best_iteration is None:
            return "No iteration satisfied the specified criteria"

        if not satisfied_criteria:
            return f"Selected iteration {best_iteration.iter} with minimal criteria satisfaction"

        criteria_str = ", ".join(satisfied_criteria)
        return f"Iteration {best_iteration.iter} satisfied: {criteria_str}"

    def classify_with_preset(
        self, instance: VRPInstance, preset_name: str
    ) -> ClassificationResult:
        """Classify using a predefined preset."""
        if preset_name not in self.presets:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {list(self.presets.keys())}"
            )

        return self.classify(instance, self.presets[preset_name])


# Convenience functions for backward compatibility
def classify_vrp_instance(
    instance: VRPInstance, threshold: float = 0.5
) -> ClassificationResult:
    """Simple classifier using only classifier score threshold."""
    criteria = ClassificationCriteria(
        classifier_score_threshold=threshold, require_valid=True, combination_mode="AND"
    )
    classifier = EnhancedVRPClassifier(criteria)
    return classifier.classify(instance)


def classify_vrp_instance_multi(
    instance: VRPInstance,
    classifier_threshold: float = 0.5,
    entropy_threshold: float = 0.3,
    max_violations: int = 5,
) -> ClassificationResult:
    """Multi-criteria classifier."""
    criteria = ClassificationCriteria(
        classifier_score_threshold=classifier_threshold,
        entropy_score_threshold=entropy_threshold,
        max_violations=max_violations,
        require_valid=True,
        combination_mode="AND",
    )
    classifier = EnhancedVRPClassifier(criteria)
    return classifier.classify(instance)
