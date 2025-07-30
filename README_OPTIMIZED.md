# Optimized VRP Iterative Pipeline

This repository contains an optimized iterative pipeline that combines PyTorch-based arc flagging with Vehicle Routing Problem (VRP) solving. The main improvement is that the PyTorch model is loaded once and reused across iterations, significantly improving performance.

## Files Overview

### Core Pipeline Files

- `iterative_optimization.py` - Main optimized pipeline class
- `custom_cost_integration.py` - Helper functions for custom cost file creation
- `run_vrp_solver.sh` - Shell script to run iteratvie VRP solver

### Java VRP Solver Integration

- `MSH/MSH/src/split/SplitWithEdgeConstraints.java` - Modified with custom cost calculation
- `MSH/MSH/src/distanceMatrices/CustomArcCostMatrix.java` - Custom cost storage
- `MSH/MSH/src/pulseStructures/PulseHandlerCC.java` - Pulse algorithm with custom costs

## Quick Start

### 1. Basic Usage

```python
from iterative_optimization import OptimizedVRPPipeline

# Initialize pipeline (loads model once)
pipeline = OptimizedVRPPipeline()

# Run iterative optimization
results = pipeline.iterative_optimization(
    instance_number=6,
    max_iterations=5,
    convergence_threshold=0.01
)

print(f"Best objective: {results['best_objective']}")
print(f"Converged: {results['converged']}")
```

### 2. Single Arc Flagging

```python
# Flag arcs without full optimization
flagged_arcs, flagged_coordinates = pipeline.flag_arcs(6, "1")
print(f"Flagged {len(flagged_arcs)} problematic arcs")
```

### 3. Custom Cost File Creation

```python
from custom_cost_integration import create_custom_cost_file_from_flagged_arcs

# Create custom cost file from flagged arcs
cost_file = create_custom_cost_file_from_flagged_arcs(
    flagged_arcs=flagged_arcs,
    instance_number=6,
    flagged_walking_multiplier=2.0,
    flagged_driving_multiplier=1.5
)
```



## Bug Fixes Applied

1. **Walking Cost Calculation**: Fixed loop indexing in `calculateTotalCost()` method
2. **Model Reuse**: Eliminated repeated model loading for significant performance gain
3. **Arc Processing**: Improved error handling and debugging output
4. **Java Command Integration**: Fixed VRP solver execution to use proper Gurobi integration

### Java Command Integration Fix

**Problem**: The original pipeline used a simple JAR execution command, but the VRP solver requires proper Gurobi integration.

**Solution**: Updated the `run_vrp_solver` method to use the correct Java command format:

```bash
java -Xmx14000m "-Djava.library.path=C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_customCosts Coordinates_X.txt Arcs_X_1.txt configurationCustomCosts.xml
```

**Key Changes**:
- Proper Gurobi classpath and library path configuration
- Enhanced error reporting with stdout/stderr capture
- File validation checks for required inputs
- Support for different arc file suffixes

### Prerequisites

1. **Java 8+** with proper PATH configuration
2. **Gurobi Optimizer** (version 12.01) installed at `C:\gurobi1201\`
3. **Python 3.8+** with required packages:
   ```bash
   pip install torch torchvision matplotlib hydra-core
   ```

### Testing Java Integration

Before running the full pipeline, test the Java command integration:

```bash
python test_java_command.py
```

This will validate:

- All required files exist
- Gurobi installation paths are correct
- Java command executes properly
- Result parsing works correctly

## Testing

### Run All Tests

```bash
python test_iterative_optimization.py --test all
```

### Run Specific Tests

```bash
# Test single instance processing
python test_iterative_optimization.py --test single

# Test full iterative optimization
python test_iterative_optimization.py --test iterative

# Benchmark model loading performance
python test_iterative_optimization.py --test benchmark
```

For complete documentation and usage examples, see the individual Python files which contain detailed docstrings and comments.
