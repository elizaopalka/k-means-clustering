"""
plot.py

All plotting functions for the clustering pipeline. Kept separate from
cluster.py so the modeling code has no matplotlib dependency.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_risk_histograms(merged, out_path):
    n_clusters = merged["cluster"].nunique()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, cluster_id in enumerate(sorted(merged["cluster"].unique())):
        subset = merged[merged["cluster"] == cluster_id]["risk_score"].dropna()
        axes[i].hist(subset, bins=30, color=sns.color_palette("muted")[i % 10], alpha=0.85, edgecolor="white")
        mean_val = subset.mean()
        axes[i].axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.3f}")
        axes[i].set_title(f"Cluster {cluster_id} — risk score distribution")
        axes[i].set_xlabel("% sentences with match")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()

    for j in range(n_clusters, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Risk score distribution by cluster", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_scatter(merged, explained_variance, out_path):
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(
        data=merged, x="PC1", y="PC2", hue="cluster", size="risk_score",
        sizes=(10, 120), palette="muted", alpha=0.5, ax=ax,
    )
    ax.set_title("K-Means clusters — colored by cluster, sized by risk score")
    ax.set_xlabel("PC1 (TF-IDF term-frequency space)")
    ax.set_ylabel("PC2 (TF-IDF term-frequency space)")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.text(
        0.5, -0.03,
        f"2D projection explains {explained_variance:.1%} of variance | "
        "point size = % of document sentences flagged",
        ha="center", fontsize=9, color="gray",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_risk_bar(cluster_stats, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        cluster_stats["cluster"], cluster_stats["mean_risk"],
        color=sns.color_palette("muted", len(cluster_stats)), edgecolor="white",
    )
    for bar, z in zip(bars, cluster_stats["z_score"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"z={z:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean risk score (% sentences matched)")
    ax.set_title("Average risk score per cluster (z-score annotated)")
    ax.set_xticks(cluster_stats["cluster"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_exposure_clusters(exposure_clustered, out_path):
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(
        data=exposure_clustered, x="PC1", y="PC2", hue="cluster",
        palette="muted", alpha=0.6, s=25, ax=ax,
    )
    ax.set_title("K-Means clusters — document-level exposure statistics (numeric features)")
    ax.set_xlabel("PC1 (standardized numeric feature space)")
    ax.set_ylabel("PC2 (standardized numeric feature space)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
