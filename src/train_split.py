# test_train.py this scripts takes two folders and divides it into train and test sets

import os
import shutil
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig


def split_dataset(
    src_dir: str, dst_dir: str, train_ratio: float = 0.8, random_state: int = 42
):
    """
    Splits images in src_dir/<class> into train/test sets and copies them
    into dst_dir/train/<class> and dst_dir/test/<class>.

    :param src_dir: root folder containing one subfolder per class
    :param dst_dir: where to create `train/` and `test/` subfolders
    :param train_ratio: fraction of images to put in the training set
    :param random_state: seed for reproducibility
    """
    # find all class subdirectories
    classes = sorted(
        [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    )

    for cls in classes:
        class_src = os.path.join(src_dir, cls)
        images = [
            f
            for f in os.listdir(class_src)
            if os.path.isfile(os.path.join(class_src, f))
        ]
        if len(images) == 0:
            # exit function
            return None
        print(f"Found {len(images)} images in class '{cls}'")
        # split into train/test
        train_imgs, test_imgs = train_test_split(
            images, train_size=train_ratio, random_state=random_state, shuffle=True
        )

        for subset, filenames in (("train", train_imgs), ("test", test_imgs)):
            subset_dir = os.path.join(dst_dir, subset, cls)
            os.makedirs(subset_dir, exist_ok=True)
            for fname in filenames:
                src_path = os.path.join(class_src, fname)
                dst_path = os.path.join(subset_dir, fname)
                shutil.copy2(src_path, dst_path)


@hydra.main(config_path="../config/plot/", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    # Example usage:
    # your original data in "./data/" with subfolders label0/, label1/, etc.
    # will be split into "./splits/train/..." and "./splits/test/..."
    src_dirs = ["mask_removed_color", "mask_removed", "mask_classic"]
    input_dir = cfg.mask_output_folder
    output_dir = cfg.mask_split_output_folder

    for src_dir in src_dirs:

        split_dataset(
            src_dir=input_dir + f"{src_dir}",
            dst_dir=output_dir + f"{src_dir}/",
            train_ratio=cfg.train_ratio,  # e.g. 75% train, 25% test
            random_state=cfg.random_state,
        )

    src_dir = cfg.output_folder
    dst_dir = output_dir
    split_dataset(
        src_dir=src_dir,
        dst_dir=dst_dir,
        train_ratio=cfg.train_ratio,  # e.g. 75% train, 25% test
        random_state=cfg.random_state,
    )


if __name__ == "__main__":
    main()
