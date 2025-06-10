# test_iterative_optimization.py: Test script for the optimized VRP pipeline

import sys
import os

# Add the current directory to the Python path so we can import the pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.optimized_pipeline import OptimizedVRPPipeline


def test_single_instance():
    """Test the optimized pipeline on a single instance."""

    print("Testing OptimizedVRPPipeline on instance 6...")

    # Initialize the pipeline (loads model once)
    pipeline = OptimizedVRPPipeline()

    # Test single arc flagging
    print("\n=== Testing Single Arc Flagging ===")
    flagged_arcs, flagged_coordinates = pipeline.flag_arcs(6, "1")
    print(f"Flagged {len(flagged_arcs)} arcs")  # Test VRP solver with new Java command
    print("\n=== Testing VRP Solver with Gurobi Integration ===")
    solver_success = pipeline.run_vrp_solver(6, "configurationCustomCosts.xml", "1")
    print(f"VRP solver success: {solver_success}")

    # Test objective parsing
    print("\n=== Testing Objective Parsing ===")
    objective = pipeline.parse_objective_value(6)
    print(f"Parsed objective value: {objective}")


def test_iterative_optimization():
    """Test the full iterative optimization process."""

    print("Testing full iterative optimization...")

    # Initialize the pipeline (loads model once)
    pipeline = OptimizedVRPPipeline()

    # Run iterative optimization with smaller parameters for testing
    results = pipeline.iterative_optimization(
        instance_number=6,
        max_iterations=3,  # Reduced for testing
        convergence_threshold=0.05,  # Relaxed for testing
    )

    # Print detailed results
    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Instance: {results['instance']}")
    print(f"Total iterations: {len(results['iterations'])}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Best objective: {results['best_objective']}")
    print(f"Best iteration: {results['best_iteration']}")
    print(f"Converged: {results['converged']}")

    print("\nIteration breakdown:")
    for i, iter_result in enumerate(results["iterations"]):
        print(f"  Iteration {i+1}:")
        print(f"    Objective: {iter_result['objective_value']}")
        print(f"    Time: {iter_result['time']:.2f}s")
        print(f"    Flagged arcs: {iter_result['flagged_arcs_count']}")
        print(f"    Solver success: {iter_result['solver_success']}")


def benchmark_model_loading():
    """Benchmark the difference between loading model each time vs. reusing."""

    import time

    print("Benchmarking model loading performance...")

    # Test 1: Load model each time (like original pipeline)
    print("\n=== Test 1: Loading model each iteration ===")
    times_with_loading = []

    for i in range(3):
        start_time = time.perf_counter()

        # Create new pipeline (loads model)
        pipeline = OptimizedVRPPipeline()

        # Flag arcs
        flagged_arcs, _ = pipeline.flag_arcs(6, "1")

        end_time = time.perf_counter()
        iteration_time = end_time - start_time
        times_with_loading.append(iteration_time)

        print(
            f"  Iteration {i+1}: {iteration_time:.2f}s (flagged {len(flagged_arcs)} arcs)"
        )

    # Test 2: Reuse loaded model
    print("\n=== Test 2: Reusing loaded model ===")
    times_with_reuse = []

    # Load model once
    pipeline = OptimizedVRPPipeline()

    for i in range(3):
        start_time = time.perf_counter()

        # Flag arcs (reuses loaded model)
        flagged_arcs, _ = pipeline.flag_arcs(6, "1")

        end_time = time.perf_counter()
        iteration_time = end_time - start_time
        times_with_reuse.append(iteration_time)

        print(
            f"  Iteration {i+1}: {iteration_time:.2f}s (flagged {len(flagged_arcs)} arcs)"
        )

    # Compare results
    avg_with_loading = sum(times_with_loading) / len(times_with_loading)
    avg_with_reuse = sum(times_with_reuse) / len(times_with_reuse)
    speedup = avg_with_loading / avg_with_reuse

    print(f"\n=== BENCHMARK RESULTS ===")
    print(f"Average time with model loading: {avg_with_loading:.2f}s")
    print(f"Average time with model reuse: {avg_with_reuse:.2f}s")
    print(f"Speedup factor: {speedup:.2f}x")
    print(f"Time saved per iteration: {avg_with_loading - avg_with_reuse:.2f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the optimized VRP pipeline")
    parser.add_argument(
        "--test",
        choices=["single", "iterative", "benchmark", "all"],
        default="all",
        help="Which test to run",
    )

    args = parser.parse_args()

    try:
        if args.test in ["single", "all"]:
            test_single_instance()
            print("\n" + "=" * 50)

        if args.test in ["iterative", "all"]:
            test_iterative_optimization()
            print("\n" + "=" * 50)

        if args.test in ["benchmark", "all"]:
            benchmark_model_loading()
            print("\n" + "=" * 50)

        print("All tests completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
