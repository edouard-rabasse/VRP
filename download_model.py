from huggingface_hub import hf_hub_download
import os


def fetch_checkpoint(
    repo_id: str,
    filename: str,
    cache_dir: str = "checkpoints",
    revision: str = "main",
    token: str | None = None,
) -> str:
    """
    Télécharge un fichier depuis un repo Hugging Face et retourne le chemin local.
    """
    os.makedirs(cache_dir, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        token=token,  # None si public
    )
    return local_path


if __name__ == "__main__":
    checkpoint_path = fetch_checkpoint(
        repo_id="Eddedc/resnet50_VRP",
        # filename="resnet_64_40_7_config7_new_r7f3tthy.pth",
        filename="vgg_32_50_7_config7_new_e8rnnn1b.pth",
        cache_dir="checkpoints",
        revision="main",
        token=None,
    )
