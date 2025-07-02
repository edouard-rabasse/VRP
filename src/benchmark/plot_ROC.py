import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics_by_threshold(
    full_df, folder="results/", filename="metrics_by_threshold.png"
):
    """
    Trace les courbes des métriques (précision, rappel, F1-score, taux de faux positifs)
    en fonction du seuil (threshold) à partir des résultats agrégés.
    """

    # Calculer la moyenne des métriques par seuil
    mean_df = (
        full_df.groupby("threshold")[
            ["precision", "recall", "f1", "false_positive_rate"]
        ]
        .mean()
        .reset_index()
    )

    print("\n📊 Moyennes par seuil :")
    print(mean_df)
    # Tracer les courbes des métriques en fonction du seuil
    plt.figure(figsize=(10, 6))
    plt.plot(mean_df["threshold"], mean_df["precision"], marker="o", label="Precision")
    plt.plot(mean_df["threshold"], mean_df["recall"], marker="o", label="Recall")
    plt.plot(mean_df["threshold"], mean_df["f1"], marker="o", label="F1-Score")
    plt.plot(
        mean_df["threshold"],
        mean_df["false_positive_rate"],
        marker="o",
        label="False Positive Rate",
    )

    plt.xlabel("Seuil (threshold)")
    plt.ylabel("Score")
    plt.title("Métriques en fonction du seuil")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Sauvegarde ou affichage
    plt.savefig(os.path.join(folder, filename))
    # plt.show()
