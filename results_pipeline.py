from src.results.instance_loader import InstanceCSVLoader

import sys


def main(path, range=range(1, 40)):
    """
    Main function to load instances and compute mean summaries.

    Args:
        path (str): Path to the directory containing instance files.
        range (range): Range of instance numbers to process.
    """
    loader = InstanceCSVLoader(base_path=path)
    loader.load_multiple_instances(list(range))

    summaries = loader.get_mean_summaries()
    print("Mean summaries for instances:")
    print(summaries)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "output/10_1_2e-05"  # Default path if not provided
    main(path)
