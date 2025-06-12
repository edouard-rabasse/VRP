# File: src/main.py
from src.pipeline.optimized_pipeline import OptimizedVRPPipeline

if __name__ == "__main__":
    pipeline = OptimizedVRPPipeline()
    res = pipeline.iterative_optimization(instance=7, max_iter=10)
    print(res)
