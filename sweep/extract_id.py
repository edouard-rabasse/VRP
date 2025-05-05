import subprocess
import sys
import re

def create_sweep(sweep_config_path):
    try:
        # Run wandb sweep and capture stdout/stderr
        result = subprocess.run(
            ["wandb", "sweep", sweep_config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )

        output = result.stdout

        # Try to find sweep ID using regex
        match = re.search(r"Creating sweep with ID: ([a-z0-9]+)", output)
        if match:
            sweep_id = match.group(1)
            print(sweep_id)
            return

        # Try fallback: extract from URL
        match = re.search(r"sweeps/([a-z0-9]+)", output)
        if match:
            sweep_id = match.group(1)
            print(sweep_id)
            return

        print("❌ ERROR: Could not extract sweep ID from output.")
        print("Full output:\n", output)

    except subprocess.CalledProcessError as e:
        print("❌ ERROR while running wandb sweep:", e.output)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_sweep.py path/to/sweep.yaml")
        sys.exit(1)

    sweep_yaml = sys.argv[1]
    create_sweep(sweep_yaml)
