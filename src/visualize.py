"""
This module generates 4 key plots to help validate the pipeline output:
    1. Hypergraph: Shows the top coordinated clusters with node colors (Profile Prior).
    2. Score Distribution: Histogram of coordination scores with null threshold overlay.
    3. Temporal Burst: Rug plot comparing top cluster vs. organic baseline.
    4. Size vs. Score: Scatter plot identifying the "sweet spot" of coordination.

All plots are saved to a specified output directory for easy inspection.
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from matplotlib.patches import Patch

from src.config import Config

# Set a clean, professional plotting style
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 12


#HYPERGRAPH VISUALIZATION
def plot_hypergraph(
    graph: nx.Graph,
    scored_clusters: List[Dict[str, Any]],
    profile_priors: Dict[str, float],
    top_k: int = 3,
    save_path: Optional[str] = None,
    tier: str = "dev"
) -> None:
    """
    Plot the subgraph containing the top K coordinated clusters.

    Node colors:
        - Red: High suspicion (Profile Prior > 0.7)
        - Blue: Low suspicion (Profile Prior < 0.3)
        - Purple/Mixed: Moderate suspicion (0.3 – 0.7)

    Edge width: Strength of coordination signal.

    Args:
        graph: Weighted NetworkX graph from sniffers.py.
        scored_clusters: Scored clusters from scorer.py.
        profile_priors: dict[account_id] -> suspicion score.
        top_k: Number of top coordinated clusters to display.
        save_path: If provided, saves the figure to this path.
        tier: "dev" or "eval" (used for title).
    """
    # Filter to coordinated clusters and take top K
    coord_clusters = [c for c in scored_clusters if c.get("is_coordinated", False)]
    coord_clusters.sort(key=lambda x: x["coordination_score"], reverse=True)
    top_clusters = coord_clusters[:top_k]

    if not top_clusters:
        print("No coordinated clusters to plot in hypergraph.")
        return

    # Collect all account IDs from the top clusters
    nodes_to_keep = set()
    for cluster in top_clusters:
        nodes_to_keep.update(cluster["account_ids"])

    # Extract subgraph
    subgraph = graph.subgraph(nodes_to_keep).copy()

    if subgraph.number_of_nodes() < 2:
        print("Subgraph too small to visualize.")
        return

    # Compute node colors based on profile prior
    node_colors = []
    for node in subgraph.nodes():
        prior = profile_priors.get(node, 0.5)
        # Red (high) to Blue (low), with Purple in between
        if prior > 0.7:
            node_colors.append("red")
        elif prior < 0.3:
            node_colors.append("blue")
        else:
            node_colors.append("purple")

    # Edge widths based on weight
    if subgraph.edges():
        edge_weights = [subgraph[u][v]["weight"] for u, v in subgraph.edges()]
        max_weight = max(edge_weights) if edge_weights else 1.0
        edge_widths = [w / max_weight * 3 + 0.5 for w in edge_weights]
    else:
        edge_widths = []

    # Compute layout
    try:
        pos = nx.spring_layout(subgraph, seed=42, k=0.5)
    except Exception:
        # Fallback for disconnected graphs
        pos = nx.random_layout(subgraph, seed=42)

    # Create figure
    plt.figure(figsize=(12, 10))

    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph, pos,
        node_size=300,
        node_color=node_colors,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5
    )

    # Draw edges
    if edge_widths:
        nx.draw_networkx_edges(
            subgraph, pos,
            width=edge_widths,
            alpha=0.6,
            edge_color="gray"
        )

    # Draw labels
    nx.draw_networkx_labels(
        subgraph, pos,
        font_size=8,
        font_weight="bold",
        font_color="black"
    )

    # Legend
    legend_elements = [
        Patch(facecolor="red", alpha=0.7, label="High Suspicion (Profile Prior > 0.7)"),
        Patch(facecolor="purple", alpha=0.7, label="Mixed / Unknown (Profile Prior ~ 0.5)"),
        Patch(facecolor="blue", alpha=0.7, label="Low Suspicion (Profile Prior < 0.3)"),
    ]

    plt.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.title(
        f"Hypergraph of Top {len(top_clusters)} Coordinated Clusters (Tier: {tier.upper()})\n"
        f"Node Color: Red = Suspicious Profile | Blue = Organic Profile\n"
        f"Edge Width: Strength of Coordination Signal",
        fontsize=14
    )
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Hypergraph saved to: {save_path}")

    plt.show()
    plt.close()


#SCORE DISTRIBUTION
def plot_score_distribution(
    scored_clusters: List[Dict[str, Any]],
    null_threshold: float = 0.5,
    save_path: Optional[str] = None,
    tier: str = "dev"
) -> None:
    """
    Plot a histogram of coordination scores with the null threshold overlay.

    Args:
        scored_clusters: Scored clusters from scorer.py.
        null_threshold: The calibrated threshold value.
        save_path: If provided, saves the figure to this path.
        tier: "dev" or "eval" (used for title).
    """
    if not scored_clusters:
        print("⚠️  No clusters to plot for score distribution.")
        return

    scores = [c["coordination_score"] for c in scored_clusters]
    coord_flags = [c.get("is_coordinated", False) for c in scored_clusters]

    coord_scores = [s for s, f in zip(scores, coord_flags) if f]
    non_coord_scores = [s for s, f in zip(scores, coord_flags) if not f]

    plt.figure(figsize=(12, 6))

    # Histograms
    if non_coord_scores:
        plt.hist(
            non_coord_scores,
            bins=20,
            alpha=0.6,
            label="Not Coordinated (Noise)",
            color="blue",
            edgecolor="black"
        )
    if coord_scores:
        plt.hist(
            coord_scores,
            bins=20,
            alpha=0.8,
            label="Coordinated (Signal)",
            color="red",
            edgecolor="black"
        )

    # Null threshold line
    plt.axvline(
        x=null_threshold,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Null Threshold ({null_threshold:.3f})"
    )

    plt.xlabel("Coordination Score", fontsize=12)
    plt.ylabel("Number of Clusters", fontsize=12)
    plt.title(
        f"Score Distribution vs. Null Threshold (Tier: {tier.upper()})\n"
        f"Signal: {len(coord_scores)} clusters | Noise: {len(non_coord_scores)} clusters",
        fontsize=14
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Score distribution saved to: {save_path}")

    plt.show()
    plt.close()

# TEMPORAL BURST COMPARISON
def plot_temporal_burst(
    scored_clusters: List[Dict[str, Any]],
    posts_dict: Dict[str, dict],
    save_path: Optional[str] = None,
    tier: str = "dev"
) -> None:
    """
    Plot a rug comparison of the top coordinated cluster's timestamps vs. organic baseline.

    Args:
        scored_clusters: Scored clusters from scorer.py.
        posts_dict: dict[post_id] -> post record.
        save_path: If provided, saves the figure to this path.
        tier: "dev" or "eval" (used for title).
    """
    # Get the top coordinated cluster
    coord_clusters = [c for c in scored_clusters if c.get("is_coordinated", False)]
    if not coord_clusters:
        print("⚠️  No coordinated clusters to plot temporal burst.")
        return

    top_cluster = max(coord_clusters, key=lambda x: x["coordination_score"])
    top_post_ids = top_cluster["post_ids"]

    # Get timestamps of the top cluster
    top_timestamps = []
    for pid in top_post_ids:
        dt = posts_dict[pid].get("created_at")
        if dt is not None:
            top_timestamps.append(dt.timestamp())

    if len(top_timestamps) < 2:
        print("⚠️  Top cluster has insufficient timestamps.")
        return

    # Get a random sample of organic posts (non-coordinated clusters)
    non_coord_clusters = [c for c in scored_clusters if not c.get("is_coordinated", False)]
    if non_coord_clusters:
        random_clusters = random.sample(non_coord_clusters, min(3, len(non_coord_clusters)))
        baseline_timestamps = []
        for cluster in random_clusters:
            for pid in cluster["post_ids"][:10]:  # Up to 10 per cluster
                dt = posts_dict[pid].get("created_at")
                if dt is not None:
                    baseline_timestamps.append(dt.timestamp())
    else:
        # Fallback: random sample of all posts
        all_post_ids = list(posts_dict.keys())
        sample_ids = random.sample(all_post_ids, min(20, len(all_post_ids)))
        baseline_timestamps = []
        for pid in sample_ids:
            dt = posts_dict[pid].get("created_at")
            if dt is not None:
                baseline_timestamps.append(dt.timestamp())

    if not baseline_timestamps:
        print("⚠️  No baseline timestamps available.")
        return

    # Normalize to minutes relative to the first timestamp
    ref_time = min(top_timestamps + baseline_timestamps)
    top_minutes = [(t - ref_time) / 60 for t in top_timestamps]
    baseline_minutes = [(t - ref_time) / 60 for t in baseline_timestamps]

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.scatter(
        top_minutes,
        [1] * len(top_minutes),
        marker="|",
        s=200,
        color="red",
        alpha=0.8,
        linewidths=2
    )
    ax1.set_title(
        f"Top Coordinated Cluster (Score: {top_cluster['coordination_score']:.4f})",
        fontsize=12
    )
    ax1.set_ylabel("Posts")
    ax1.set_ylim(0.8, 1.2)
    ax1.set_yticks([])

    ax2.scatter(
        baseline_minutes,
        [1] * len(baseline_minutes),
        marker="|",
        s=200,
        color="blue",
        alpha=0.5,
        linewidths=1
    )
    ax2.set_title("Baseline Organic Posts (Random Sample)", fontsize=12)
    ax2.set_xlabel("Time (minutes relative to first post)")
    ax2.set_ylabel("Posts")
    ax2.set_ylim(0.8, 1.2)
    ax2.set_yticks([])

    plt.suptitle(f"Temporal Burst Comparison (Tier: {tier.upper()})", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Temporal burst comparison saved to: {save_path}")

    plt.show()
    plt.close()

#CLUSTER SIZE VS. SCORE
def plot_size_vs_score(
    scored_clusters: List[Dict[str, Any]],
    null_threshold: float = 0.5,
    save_path: Optional[str] = None,
    tier: str = "dev"
) -> None:
    """
    Scatter plot: number of posts per cluster vs coordination_score,
    colored by is_coordinated.

    Args:
        scored_clusters: Scored clusters from scorer.py.
        null_threshold: The calibrated threshold value.
        save_path: If provided, saves the figure to this path.
        tier: "dev" or "eval" (used for title).
    """
    if not scored_clusters:
        print("⚠️  No clusters to plot.")
        return

    sizes = [len(c["post_ids"]) for c in scored_clusters]
    scores = [c["coordination_score"] for c in scored_clusters]
    colors = ["red" if c.get("is_coordinated", False) else "blue" for c in scored_clusters]

    plt.figure(figsize=(12, 7))

    plt.scatter(
        scores,
        sizes,
        c=colors,
        alpha=0.7,
        s=80,
        edgecolors="black",
        linewidths=0.5
    )

    # Null threshold line
    plt.axvline(
        x=null_threshold,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Null Threshold ({null_threshold:.3f})"
    )

    # Annotate the top cluster
    if scored_clusters:
        top = max(scored_clusters, key=lambda x: x["coordination_score"])
        plt.annotate(
            f"Top Cluster\nScore: {top['coordination_score']:.3f}",
            xy=(top["coordination_score"], len(top["post_ids"])),
            xytext=(top["coordination_score"] + 0.05, len(top["post_ids"]) + 2),
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=10,
            ha="left"
        )

    plt.xlabel("Coordination Score", fontsize=12)
    plt.ylabel("Number of Posts in Cluster", fontsize=12)
    plt.title(
        f"Cluster Size vs. Coordination Score (Tier: {tier.upper()})\n"
        "Red = Flagged Coordinated | Blue = Marked Noise",
        fontsize=14
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Size-vs-score plot saved to: {save_path}")

    plt.show()
    plt.close()



# ORCHESTRATOR

def run_visualizations(
    scored_clusters: List[Dict[str, Any]],
    graph: nx.Graph,
    bundle: Any,  # DataBundle
    null_threshold: Optional[float] = None,
    output_dir: str = "figures",
    tier: str = "dev"
) -> None:
    """
    Main orchestration function: generates all 4 visualizations.

    Args:
        scored_clusters: Output from scorer.calibrate_clusters().
        graph: Weighted NetworkX graph from sniffers.py.
        bundle: DataBundle from data_loader.py.
        null_threshold: The calibrated threshold. If None, attempts to infer.
        output_dir: Directory to save the figures.
        tier: "dev" or "eval".
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Infer null threshold if not provided
    if null_threshold is None:
        coord_scores = [c["coordination_score"] for c in scored_clusters if c.get("is_coordinated", False)]
        non_coord_scores = [c["coordination_score"] for c in scored_clusters if not c.get("is_coordinated", False)]
        if coord_scores and non_coord_scores:
            null_threshold = (min(coord_scores) + max(non_coord_scores)) / 2
        else:
            null_threshold = Config.PERCENTILE_THRESHOLD / 100.0  # fallback

    print("\n" + "=" * 60)
    print(f"📊 GENERATING VISUALIZATIONS (Tier: {tier.upper()})")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Using null threshold: {null_threshold:.4f}")

    #Hypergraph
    hypergraph_path = os.path.join(output_dir, f"hypergraph_{tier}.png")
    plot_hypergraph(
        graph,
        scored_clusters,
        bundle.profile_priors,
        top_k=3,
        save_path=hypergraph_path,
        tier=tier
    )

    #Score Distribution
    score_dist_path = os.path.join(output_dir, f"score_distribution_{tier}.png")
    plot_score_distribution(
        scored_clusters,
        null_threshold,
        save_path=score_dist_path,
        tier=tier
    )

    #Temporal Burst
    burst_path = os.path.join(output_dir, f"temporal_burst_{tier}.png")
    plot_temporal_burst(
        scored_clusters,
        bundle.posts_dict,
        save_path=burst_path,
        tier=tier
    )

    #Size vs. Score
    size_path = os.path.join(output_dir, f"size_vs_score_{tier}.png")
    plot_size_vs_score(
        scored_clusters,
        null_threshold,
        save_path=size_path,
        tier=tier
    )

    print("\n✅ All visualizations complete.")
    print(f"📂 Figures saved to: {os.path.abspath(output_dir)}")
    print("💡 Use these plots to validate your manual inspection:")
    print("   - Red nodes in the graph should represent suspicious, tightly-connected accounts.")
    print("   - The score distribution should show a clear separation between signal and noise.")
    print("   - The temporal burst plot should show the coordinated cluster firing in a tight window.")
    print("   - The size-vs-score plot should show small, dense clusters scoring high.")