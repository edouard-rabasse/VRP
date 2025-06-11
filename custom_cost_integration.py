# custom_cost_integration.py: Helper functions for integrating custom costs with the VRP solver

import os
import csv
from typing import List, Dict, Tuple, Optional


def create_custom_cost_file_from_flagged_arcs(
    flagged_arcs: List,
    instance_number: int,
    base_walking_cost: float = 1.0,
    flagged_walking_multiplier: float = 2.0,
    base_driving_cost: float = 1.0,
    flagged_driving_multiplier: float = 1.5,
    output_dir: str = "MSH/MSH/customCosts",
) -> str:
    """
    Create a custom cost file from flagged arcs that can be read by the Java VRP solver.

    Args:
        flagged_arcs: List of flagged arcs from PyTorch model
        instance_number: Instance number for file naming
        base_walking_cost: Base multiplier for walking costs
        flagged_walking_multiplier: Multiplier for walking costs on flagged arcs
        base_driving_cost: Base multiplier for driving costs
        flagged_driving_multiplier: Multiplier for driving costs on flagged arcs
        output_dir: Directory to save the custom cost file

    Returns:
        Path to the created custom cost file
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    custom_cost_file = os.path.join(output_dir, f"custom_costs_{instance_number}.txt")

    print(f"[CustomCost] Creating custom cost file: {custom_cost_file}")
    print(f"[CustomCost] Flagged arcs count: {len(flagged_arcs)}")

    # Track unique arcs to avoid duplicates
    processed_arcs = set()

    with open(custom_cost_file, "w") as f:
        f.write("# Custom arc costs based on PyTorch model flagging\n")
        f.write("# Format: tail;head;mode;cost\n")
        f.write("# mode: 1=driving, 2=walking\n")
        f.write(
            "# Multipliers: walking_flagged={}, driving_flagged={}\n".format(
                flagged_walking_multiplier, flagged_driving_multiplier
            )
        )
        f.write("\n")

        # Process flagged arcs
        for arc in flagged_arcs:
            # Extract arc information - adjust based on your arc data structure
            if hasattr(arc, "tail") and hasattr(arc, "head"):
                tail, head = arc.tail, arc.head
            elif isinstance(arc, dict):
                tail, head = arc.get("tail"), arc.get("head")
            elif isinstance(arc, (list, tuple)) and len(arc) >= 2:
                tail, head = arc[0], arc[1]
            else:
                print(
                    f"[CustomCost] Warning: Could not extract tail/head from arc: {arc}"
                )
                continue

            # Create unique arc identifier
            arc_key = f"{tail}_{head}"
            if arc_key in processed_arcs:
                continue
            processed_arcs.add(arc_key)

            # Add custom costs for both walking and driving modes
            f.write(f"{tail};{head};1;{flagged_driving_multiplier}\n")  # Driving cost
            f.write(f"{tail};{head};2;{flagged_walking_multiplier}\n")  # Walking cost

    print(
        f"[CustomCost] Written {len(processed_arcs)} unique flagged arcs to {custom_cost_file}"
    )
    return custom_cost_file


def read_arc_usage_from_solution(solution_file: str) -> List[Dict]:
    """
    Read arc usage from a VRP solution file.

    Args:
        solution_file: Path to the solution file (e.g., Arcs_6_1.txt)

    Returns:
        List of dictionaries containing arc information
    """
    arcs = []

    if not os.path.exists(solution_file):
        print(f"[CustomCost] Warning: Solution file not found: {solution_file}")
        return arcs

    try:
        with open(solution_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse arc format: typically "tail -> head" or similar
                if "->" in line:
                    parts = line.split("->")
                    if len(parts) == 2:
                        tail = parts[0].strip()
                        head = parts[1].strip()

                        # Try to convert to integers if possible
                        try:
                            tail = int(tail)
                            head = int(head)
                        except ValueError:
                            pass

                        arcs.append(
                            {"tail": tail, "head": head, "line": line_num, "raw": line}
                        )
                else:
                    # Try other parsing methods based on your file format
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            tail = int(parts[0])
                            head = int(parts[1])
                            arcs.append(
                                {
                                    "tail": tail,
                                    "head": head,
                                    "line": line_num,
                                    "raw": line,
                                }
                            )
                        except ValueError:
                            continue

    except Exception as e:
        print(f"[CustomCost] Error reading solution file {solution_file}: {e}")

    print(f"[CustomCost] Read {len(arcs)} arcs from {solution_file}")
    return arcs


def compare_solutions(
    solution_before: str, solution_after: str, instance_number: int
) -> Dict[str, any]:
    """
    Compare two VRP solutions to analyze the impact of custom costs.

    Args:
        solution_before: Path to solution file before custom costs
        solution_after: Path to solution file after custom costs
        instance_number: Instance number for reporting

    Returns:
        Dictionary with comparison results
    """

    print(f"[CustomCost] Comparing solutions for instance {instance_number}")

    arcs_before = read_arc_usage_from_solution(solution_before)
    arcs_after = read_arc_usage_from_solution(solution_after)

    # Convert to sets for comparison
    arcs_before_set = {(arc["tail"], arc["head"]) for arc in arcs_before}
    arcs_after_set = {(arc["tail"], arc["head"]) for arc in arcs_after}

    # Find differences
    arcs_removed = arcs_before_set - arcs_after_set
    arcs_added = arcs_after_set - arcs_before_set
    arcs_common = arcs_before_set & arcs_after_set

    # Calculate similarity
    total_unique_arcs = len(arcs_before_set | arcs_after_set)
    similarity = len(arcs_common) / total_unique_arcs if total_unique_arcs > 0 else 0

    results = {
        "instance": instance_number,
        "arcs_before_count": len(arcs_before),
        "arcs_after_count": len(arcs_after),
        "arcs_removed": list(arcs_removed),
        "arcs_added": list(arcs_added),
        "arcs_common_count": len(arcs_common),
        "similarity": similarity,
        "change_percentage": (1 - similarity) * 100,
    }

    print(f"[CustomCost] Solution comparison results:")
    print(f"  Arcs before: {results['arcs_before_count']}")
    print(f"  Arcs after: {results['arcs_after_count']}")
    print(f"  Arcs removed: {len(arcs_removed)}")
    print(f"  Arcs added: {len(arcs_added)}")
    print(f"  Similarity: {similarity:.3f} ({(1-similarity)*100:.1f}% change)")

    return results


def extract_objective_from_log(log_file: str) -> Optional[float]:
    """
    Extract the objective value from a VRP solver log file.

    Args:
        log_file: Path to the solver log file

    Returns:
        Objective value if found, None otherwise
    """

    if not os.path.exists(log_file):
        return None

    try:
        with open(log_file, "r") as f:
            content = f.read()

        # Look for common objective value patterns
        patterns = [
            "Total Cost:",
            "Objective:",
            "Best solution:",
            "Final cost:",
            "Optimal value:",
        ]

        lines = content.split("\n")
        for line in lines:
            for pattern in patterns:
                if pattern in line:
                    # Extract numeric value after the pattern
                    parts = line.split(pattern)
                    if len(parts) > 1:
                        try:
                            # Extract the first number found
                            import re

                            numbers = re.findall(r"\d+\.?\d*", parts[1])
                            if numbers:
                                return float(numbers[0])
                        except ValueError:
                            continue

        return None

    except Exception as e:
        print(f"[CustomCost] Error reading log file {log_file}: {e}")
        return None


def create_experiment_report(
    instance_number: int, iterations: List[Dict], output_file: str = None
) -> str:
    """
    Create a detailed experiment report.

    Args:
        instance_number: Instance number
        iterations: List of iteration results
        output_file: Optional output file path

    Returns:
        Path to the created report file
    """

    if output_file is None:
        output_file = f"experiment_report_instance_{instance_number}.md"

    with open(output_file, "w") as f:
        f.write(f"# VRP Iterative Optimization Report - Instance {instance_number}\n\n")

        f.write("## Summary\n")
        f.write(f"- Total iterations: {len(iterations)}\n")

        if iterations:
            best_obj = min(
                iter_result["objective_value"]
                for iter_result in iterations
                if iter_result["objective_value"] != float("inf")
            )
            worst_obj = max(
                iter_result["objective_value"]
                for iter_result in iterations
                if iter_result["objective_value"] != float("inf")
            )

            f.write(f"- Best objective: {best_obj:.2f}\n")
            f.write(f"- Worst objective: {worst_obj:.2f}\n")
            f.write(
                f"- Improvement: {((worst_obj - best_obj) / worst_obj * 100):.2f}%\n"
            )

            total_time = sum(iter_result["time"] for iter_result in iterations)
            f.write(f"- Total time: {total_time:.2f} seconds\n")

        f.write("\n## Iteration Details\n\n")
        f.write(
            "| Iteration | Objective | Time (s) | Flagged Arcs | Solver Success |\n"
        )
        f.write(
            "|-----------|-----------|----------|--------------|----------------|\n"
        )

        for i, iter_result in enumerate(iterations, 1):
            f.write(
                f"| {i} | {iter_result['objective_value']:.2f} | "
                f"{iter_result['time']:.2f} | {iter_result['flagged_arcs_count']} | "
                f"{iter_result['solver_success']} |\n"
            )

        f.write("\n## Convergence Analysis\n")
        if len(iterations) > 1:
            objectives = [
                iter_result["objective_value"]
                for iter_result in iterations
                if iter_result["objective_value"] != float("inf")
            ]
            if len(objectives) > 1:
                improvements = []
                for i in range(1, len(objectives)):
                    if objectives[i - 1] != 0:
                        improvement = (
                            (objectives[i - 1] - objectives[i])
                            / objectives[i - 1]
                            * 100
                        )
                        improvements.append(improvement)
                        f.write(f"- Iteration {i+1}: {improvement:.2f}% improvement\n")

        f.write("\n---\n")
        f.write(f"Report generated automatically by VRP optimization pipeline.\n")

    print(f"[CustomCost] Experiment report saved to: {output_file}")
    return output_file


# Example usage and testing functions
if __name__ == "__main__":
    print("Testing custom cost integration functions...")

    # Test creating a custom cost file with dummy flagged arcs
    dummy_flagged_arcs = [
        {"tail": 1, "head": 2},
        {"tail": 2, "head": 3},
        {"tail": 3, "head": 1},
    ]

    cost_file = create_custom_cost_file_from_flagged_arcs(
        flagged_arcs=dummy_flagged_arcs,
        instance_number=6,
        flagged_walking_multiplier=2.5,
        flagged_driving_multiplier=1.3,
    )

    print(f"Created test custom cost file: {cost_file}")

    # Test reading the created file
    if os.path.exists(cost_file):
        with open(cost_file, "r") as f:
            print("Custom cost file contents:")
            print(f.read())

    print("Custom cost integration functions tested successfully!")
