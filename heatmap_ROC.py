"""
heatmap_ROC.py
This script processes heatmap metrics for VRP route analysis, generating ROC curves and other performance metrics.
"""

import os
import torch
import pandas as pd
from src.models import load_model
from src.visualization import HeatmapMetric
import hydra
from omegaconf import DictConfig
from collections import defaultdict
from src.benchmark.plot_ROC import plot_metrics_by_threshold
import matplotlib.pyplot as plt


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # ── Initialisation modèle ─────────────────────────────────────────────
    cfg.load_model = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.model.name, device, cfg.model).eval()
    print(f"[Viz] Model loaded: {cfg.model.name}")

    # ── Initialisation HeatmapMetric ───────────────────────────────────────
    heatmap_metric = HeatmapMetric(cfg, model)

    # ── Paramètres ─────────────────────────────────────────────────────────
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    input_dir = cfg.arcs.coord_in_dir  # dossier contenant les fichiers à traiter
    filenames = ["Coordinates_" + str(i) + ".txt" for i in range(1, 80)]
    print(f"[Viz] Found {len(filenames)} files to process.")

    # ── Résultats par seuil ────────────────────────────────────────────────
    all_results = defaultdict(list)

    for fname in filenames:
        print(f"[Viz] Processing {fname}")
        try:
            results = heatmap_metric.process_image(fname, thresholds)
            for metrics in results:
                th = metrics["threshold"]
                all_results[th].append(metrics)
        except Exception as e:
            print(f"[Error] Skipping {fname} due to: {e}")

    # ── Résumé global des métriques ────────────────────────────────────────
    for th in thresholds:
        if th not in all_results:
            continue
        df = pd.DataFrame(all_results[th])
        print(f"\n[Summary] Threshold {th:.2f}")
        print(
            df[
                ["precision", "recall", "f1", "false_positive_rate", "correct_best"]
            ].mean()
        )

    # ── Optionnel : Sauvegarde CSV ─────────────────────────────────────────
    full_df = pd.concat(
        [pd.DataFrame(v) for v in all_results.values()], axis=0, ignore_index=True
    )

    # remove entries with nb_diff == 0

    output_csv = os.path.join("output/", "heatmap_metrics.csv")
    full_df.to_csv(output_csv, index=False)
    print(f"\n✅ Résultats sauvegardés dans : {output_csv}")

    full_df = full_df[full_df["nb_diff"] > 0]

    plot_metrics_by_threshold(
        full_df, folder="output/", filename=f"heatmap_ROC_{cfg.model.name}.png"
    )


if __name__ == "__main__":
    main()
