# File: src/main.py
from src.pipeline.optimized_pipeline import OptimizedVRPPipeline
from src.pipeline.config import override_java_param
from time import time

if __name__ == "__main__":
    pipeline = OptimizedVRPPipeline()

    for threshold in [0.00002]:
        for walking in [10]:
            for multiplier in [1,0.5,0.1]:
                # Override Java parameters for the MSH solver
                print(
                    f"Running with walking cost: {walking}, multiplier: {multiplier}, threshold: {threshold}"
                )
                override_java_param(
                    "MSH/MSH/config/configurationCustomCosts2.xml",
                    {
                        "DEFAULT_WALK_COST": walking,
                        "CUSTOM_COST_MULTIPLIER": multiplier,
                    },
                )
                nb_equal = 0
                nb_total = 0
                nb_valid = 0
                average_diff = 0.0
                nb_converged = 0
                nb_iter = 0

                start = time()

                for i in range(1, 80):

                    res = pipeline.iterative_optimization(
                        instance=i, max_iter=100, thresh=threshold
                    )
                    nb_equal += res["equal"]
                    nb_total += 1
                    nb_valid += res["valid"]
                    if res["valid"]:
                        average_diff += res["cost_delta"]
                    if res["converged"]:
                        nb_converged += 1
                        nb_iter += res["number_iter"]

                end = time()
                total_time = end - start

                nb_diff = nb_total - nb_valid
                average_diff /= nb_valid if nb_valid > 0 else 0.0

                # save results in a file
                with open(
                    f"output/results_{walking}_{multiplier}_{threshold}.txt", "w"
                ) as f:
                    f.write(f"Walking cost: {walking}\n")
                    f.write(f"Multiplier: {multiplier}\n")
                    f.write(f"Threshold: {threshold}\n")
                    f.write(f"Total instances: {nb_total}\n")
                    f.write(f"Valid instances: {nb_valid}\n")
                    f.write(f"Equal instances: {nb_equal}\n")
                    f.write(f"Average cost difference: {average_diff:.2f}\n")
                    f.write(f"Converged instances: {nb_converged}\n")
                    f.write(
                        f"Average iterations for converged instances: {nb_iter / nb_converged if nb_converged > 0 else 0:.2f}\n"
                    )
                    f.write(f"Total time taken: {total_time:.2f} seconds\n")
