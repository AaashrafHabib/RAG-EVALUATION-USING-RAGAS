import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_radar_chart(df: pd.DataFrame, output_path: str = "data/results/radar_chart.png"):
    """Create a radar chart comparing all configurations across RAGAS metrics."""
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    num_metrics = len(metrics)

    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(df), 10)))

    for idx, row in df.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=row["config_name"], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], size=12)
    ax.set_ylim(0, 1)
    ax.set_title("RAGAS Metrics Comparison Across Configurations", size=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Radar chart saved to {output_path}")


def plot_heatmap(df: pd.DataFrame, output_path: str = "data/results/heatmap.png"):
    """Create a heatmap of RAGAS scores by configuration."""
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    heatmap_data = df.set_index("config_name")[metrics]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("RAGAS Scores Heatmap", size=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {output_path}")


def plot_latency_vs_quality(df: pd.DataFrame, output_path: str = "data/results/latency_quality.png"):
    """Scatter plot: average latency vs. average RAGAS score."""
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    df["avg_ragas_score"] = df[metrics].mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df["avg_latency"],
        df["avg_ragas_score"],
        s=100,
        c=range(len(df)),
        cmap="viridis",
        edgecolors="black",
        linewidth=0.5,
    )

    for idx, row in df.iterrows():
        ax.annotate(
            row["config_name"],
            (row["avg_latency"], row["avg_ragas_score"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel("Average Latency (seconds)", size=12)
    ax.set_ylabel("Average RAGAS Score", size=12)
    ax.set_title("Latency vs. Quality Trade-off", size=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Latency vs Quality chart saved to {output_path}")


def plot_bar_comparison(df: pd.DataFrame, output_path: str = "data/results/bar_comparison.png"):
    """Grouped bar chart comparing all metrics across configurations."""
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(df))
    width = 0.2

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df[metric], width, label=metric.replace("_", " ").title())

    ax.set_xlabel("Configuration", size=12)
    ax.set_ylabel("Score", size=12)
    ax.set_title("RAGAS Metrics by Configuration", size=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["config_name"], rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bar comparison saved to {output_path}")


def generate_all_plots(results_csv: str, output_dir: str = "data/results"):
    """Generate all visualization plots from a results CSV."""
    df = pd.read_csv(results_csv)
    df = df.dropna(subset=["faithfulness"])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_radar_chart(df, str(output_path / "radar_chart.png"))
    plot_heatmap(df, str(output_path / "heatmap.png"))
    plot_latency_vs_quality(df, str(output_path / "latency_quality.png"))
    plot_bar_comparison(df, str(output_path / "bar_comparison.png"))

    print("\nAll plots generated successfully!")
