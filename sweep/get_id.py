import subprocess
import re
import os

# Liste des modèles = noms des fichiers sweep_<model>.yaml
models = ["vgg", "resnet", "deit_tiny", "multi", "cnn", "MFCN"]
sweep_dir = "./sweep"
output_file = os.path.join(sweep_dir, "sweep_ids.txt")

with open(output_file, "w") as out:
    for model in models:
        yaml_path = os.path.join(sweep_dir, f"sweep_{model}.yaml")
        print(f"🔄 Creating sweep for {model} from {yaml_path}")

        try:
            result = subprocess.run(
                ["wandb", "sweep", yaml_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False
            )
            output = result.stdout

            # Extraction du sweep ID (via "Creating sweep with ID:" ou URL)
            match = re.search(r"Creating sweep with ID:\s*([a-z0-9]+)", output)
            if not match:
                match = re.search(r"sweeps/([a-z0-9]+)", output)

            if match:
                sweep_id = match.group(1)
                out.write(f"{model}={sweep_id}\n")
                print(f"✅ {model}: {sweep_id}")
            else:
                print(f"❌ Failed to extract sweep ID for {model}")
                print(output)

        except Exception as e:
            print(f"❌ Error running wandb sweep for {model}: {e}")
