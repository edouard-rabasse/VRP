# test_java_command.py: Test the updated Java command integration with Gurobi

import subprocess
import os
import time
from pathlib import Path


def test_java_command_direct():
    """Test the Java command directly without the pipeline."""
    print("=== Testing Java Command Integration ===")

    # Test the exact command format
    msh_dir = "MSH/MSH"
    instance_number = 2  # Use a smaller instance for testing

    command = [
        "java",
        "-Xmx14000m",
        "-Djava.library.path=C:\\gurobi1201\\win64\\bin",
        "-cp",
        "bin;C:\\gurobi1201\\win64\\lib\\gurobi.jar",
        "main.Main_customCosts",
        f"Coordinates_{instance_number}.txt",
        f"Arcs_{instance_number}_1.txt",
        "configurationCustomCosts.xml",
    ]

    print(f"Working directory: {os.path.abspath(msh_dir)}")
    print(f"Command: {' '.join(command)}")
    # Check if required files exist
    coord_file = os.path.join(
        msh_dir, "instances", f"Coordinates_{instance_number}.txt"
    )
    arc_file = os.path.join(
        msh_dir, "results", "configuration1", f"Arcs_{instance_number}_1.txt"
    )
    config_file = os.path.join(msh_dir, "configurationCustomCosts.xml")

    print(f"\nChecking required files:")
    print(
        f"  Coordinates file: {coord_file} - {'EXISTS' if os.path.exists(coord_file) else 'MISSING'}"
    )
    print(
        f"  Arcs file: {arc_file} - {'EXISTS' if os.path.exists(arc_file) else 'MISSING'}"
    )
    print(
        f"  Config file: {config_file} - {'EXISTS' if os.path.exists(config_file) else 'MISSING'}"
    )

    # Check Gurobi paths
    gurobi_lib = "C:\\gurobi1201\\win64\\lib\\gurobi.jar"
    gurobi_bin = "C:\\gurobi1201\\win64\\bin"
    print(
        f"  Gurobi JAR: {gurobi_lib} - {'EXISTS' if os.path.exists(gurobi_lib) else 'MISSING'}"
    )
    print(
        f"  Gurobi bin: {gurobi_bin} - {'EXISTS' if os.path.exists(gurobi_bin) else 'MISSING'}"
    )

    # Check compiled classes
    main_class = os.path.join(msh_dir, "bin", "main", "Main_customCosts.class")
    print(
        f"  Main class: {main_class} - {'EXISTS' if os.path.exists(main_class) else 'MISSING'}"
    )

    if not all(
        [
            os.path.exists(coord_file),
            os.path.exists(arc_file),
            os.path.exists(config_file),
        ]
    ):
        print("\nERROR: Missing required files. Cannot proceed with test.")
        return False

    try:
        print(f"\nExecuting command...")
        start_time = time.perf_counter()

        result = subprocess.run(
            command,
            cwd=msh_dir,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout for test
        )

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        print(f"Execution completed in {execution_time:.2f} seconds")
        print(f"Return code: {result.returncode}")

        if result.stdout:
            print(f"\nStandard Output:")
            print(result.stdout[:1000])  # First 1000 chars
            if len(result.stdout) > 1000:
                print("... (output truncated)")

        if result.stderr:
            print(f"\nStandard Error:")
            print(result.stderr[:1000])  # First 1000 chars
            if len(result.stderr) > 1000:
                print("... (error output truncated)")

        # Check for output files
        results_dir = os.path.join(msh_dir, "results", "configurationCustomCosts")
        if os.path.exists(results_dir):
            print(f"\nResults directory contents:")
            for item in os.listdir(results_dir):
                file_path = os.path.join(results_dir, item)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"  {item} ({file_size} bytes)")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("Command timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"Error executing command: {e}")
        return False


def test_pipeline_integration():
    """Test the updated pipeline with the new Java command."""
    print("\n=== Testing Pipeline Integration ===")

    try:
        from optimized_vrp_pipeline import OptimizedVRPPipeline

        # Create pipeline (skip model loading for this test)
        print("Creating pipeline...")

        # Test just the VRP solver part
        pipeline = OptimizedVRPPipeline()

        # Test solver execution
        print("Testing VRP solver execution...")
        instance_number = 2
        success = pipeline.run_vrp_solver(
            instance_number, "configurationCustomCosts.xml", "1"
        )

        if success:
            print("VRP solver execution successful!")

            # Test objective parsing
            print("Testing objective value parsing...")
            objective = pipeline.parse_objective_value(instance_number)
            print(f"Parsed objective value: {objective}")

            return True
        else:
            print("VRP solver execution failed!")
            return False

    except Exception as e:
        print(f"Error testing pipeline: {e}")
        return False


def main():
    """Run all tests."""
    print("Java Command Integration Test")
    print("=" * 50)

    # Test 1: Direct Java command
    success1 = test_java_command_direct()

    # Test 2: Pipeline integration
    success2 = test_pipeline_integration()

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Direct Java command test: {'PASSED' if success1 else 'FAILED'}")
    print(f"Pipeline integration test: {'PASSED' if success2 else 'FAILED'}")

    if success1 and success2:
        print("\nAll tests passed! The Java command integration is working correctly.")
    else:
        print("\nSome tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
