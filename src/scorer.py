"""
This module takes candidate clusters from sniffers.py and:
    1. Computes a temporal burst score (Gini coefficient of inter-arrival times).
    2. Aggregates profile priors (suspicion scores) for each cluster.
    3. Combines all signals into a final coordination_score (0.0–1.0).
    4. Builds a null distribution using global account shuffling (the critical fix).
    5. Applies the percentile threshold to flag is_coordinated.

All weights and thresholds are sourced from the central Config class.
"""
import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from src.config import Config
from src.data_loader import DataBundle

#GINI COEFFICIENT (using numpy for efficiency)
def compute_gini(values: List[float])->float:
    """
    Compute the Gini coefficient for a list of numeric values using NumPy.

    The Gini coefficient measures inequality. For inter-arrival times:
        - Gini = 0 means all intervals are equal (perfectly regular, bot-like).
        - Gini = 1 means one interval dominates (highly bursty, coordinated).

    Formula (sorted version):
        G = sum_{i=1}^{n} (2i - n - 1) * x_i / (n * sum_{i=1}^{n} x_i)

    Args:
        values: List of positive numeric values (e.g., time intervals in seconds).

    Returns:
        float: Gini coefficient between 0.0 and 1.0.
    """
    if not values:
        return 0.0

    arr = np.array(values, dtype=np.float64)

    # Filter out zeros (they break the formula)
    arr = arr[arr > 0]

    if len(arr) == 0:
        return 0.0

    # Sort ascending
    arr = np.sort(arr)
    n = len(arr)

    # Numerator: sum_{i=1}^{n} (2i - n - 1) * x_i
    indices = np.arange(1, n + 1)
    numerator = np.sum((2 * indices - n - 1) * arr)

    # Denominator: n * sum(x_i)
    denominator = n * np.sum(arr)

    if denominator == 0:
        return 0.0

    gini_coeff = numerator / denominator
    # Clamp due to floating point
    return max(0.0, min(1.0, gini_coeff))

#TEMPORAL SCORE
def compute_temporal_score(post_ids: List[str], posts_dict: Dict[str, dict]) -> float:
    """
    Compute a temporal burstiness score for a cluster of posts.

    Intuition:
        - Coordinated groups post in tight bursts (short inter-arrival times).
        - Organic crowds have Poisson-like arrivals (more spread out).

    Mapping:
        - Gini < 0.1: Too regular (scheduled bot) → score = 0.1
        - 0.1 <= Gini <= 0.9: Desired range → score = Gini
        - Gini > 0.9: Extremely bursty (small cluster) → score = 0.7 (down-weighted)

    Additionally:
        - Time span > 2 hours → score = 0.0 (organic thread)
        - Less than 2 posts → score = 0.0

    Args:
        post_ids: List of post IDs in the cluster.
        posts_dict: dict[post_id] -> post record.

    Returns:
        float: temporal score between 0.0 and 1.0.
    """
    if len(post_ids) < 2:
        return 0.0

    # Extract timestamps
    timestamps = []
    for pid in post_ids:
        dt = posts_dict[pid].get("created_at")
        if dt is not None:
            timestamps.append(dt.timestamp())

    if len(timestamps) < 2:
        return 0.0

    timestamps.sort()

    # Compute inter-arrival times
    intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

    # Time span penalty
    time_span = timestamps[-1] - timestamps[0]
    if time_span > 7200:  # 2 hours
        return 0.0

    # Gini coefficient
    gini_coeff = compute_gini(intervals)

    # Map Gini to a score
    if gini_coeff < 0.1:
        return 0.1  # Too regular → penalize
    elif gini_coeff <= 0.9:
        return gini_coeff
    else:
        return 0.7  # Extremely bursty, but down-weight to avoid overvaluing tiny clusters
    
#SINGLE CLUSTER SCORING
def score_candidate_cluster(
    cluster: Dict[str, Any],
    posts_dict: Dict[str, dict],
    profile_priors: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compute the final coordination_score for a single cluster.

    The score is a weighted combination of:
        1. Profile Aggregator: Average suspicion of accounts in the cluster.
        2. Temporal Score: Burstiness of the posts.
        3. Structural Score: Internal edge density from the graph.

    Weights are sourced from Config:
        PROFILE_WEIGHT + TEMPORAL_WEIGHT + STRUCTURAL_WEIGHT = 1.0

    Args:
        cluster: Dict with keys 'account_ids', 'post_ids', 'raw_score'.
        posts_dict: dict[post_id] -> post record.
        profile_priors: dict[account_id] -> suspicion score.

    Returns:
        Enriched cluster dict with:
            - 'coordination_score': combined score (0.0–1.0)
            - 'profile_aggregate': avg profile prior
            - 'temporal_score': temporal burst score
            - 'structural_score': capped raw_score
            - 'is_coordinated': False (set later)
    """
    account_ids = cluster["account_ids"]
    post_ids = cluster["post_ids"]
    structural_score = cluster.get("raw_score", 0.0)

    #Profile Aggregate
    profile_scores = [profile_priors.get(acc, 0.5) for acc in account_ids]
    profile_aggregate = np.mean(profile_scores) if profile_scores else 0.0

    # Temporal Score
    temporal_score = compute_temporal_score(post_ids, posts_dict)

    #Structural Score (capped)
    structural_capped = min(structural_score, 0.8)

    #Combine
    coordination_score = (
        Config.PROFILE_WEIGHT * profile_aggregate +
        Config.TEMPORAL_WEIGHT * temporal_score +
        Config.STRUCTURAL_WEIGHT * structural_capped
    )

    # Clamp to [0.0, 1.0]
    coordination_score = max(0.0, min(1.0, coordination_score))

    return {
        "account_ids": account_ids,
        "post_ids": post_ids,
        "raw_score": structural_score,
        "profile_aggregate": profile_aggregate,
        "temporal_score": temporal_score,
        "structural_capped": structural_capped,
        "coordination_score": coordination_score,
        "is_coordinated": False,  # Set by calibrate_clusters()
    }

#GLOBAL NULL MODEL 
def build_null_distribution(
    candidate_clusters: List[Dict[str, Any]],
    posts_dict: Dict[str, dict],
    profile_priors: Dict[str, float],
    n_permutations: int = 1000
) -> np.ndarray:
    """
    Build a null distribution by shuffling account_ids GLOBALLY.

    CRITICAL FIX: Instead of shuffling within each cluster (which preserves
    structural density), we shuffle ALL account IDs across the ENTIRE dataset.

    This destroys the structure of large organic crowds, giving us a realistic
    noise baseline. This was the key insight that allowed us to find the signal.

    Args:
        candidate_clusters: List of candidate clusters from sniffers.py.
        posts_dict: dict[post_id] -> post record.
        profile_priors: dict[account_id] -> suspicion score.
        n_permutations: Number of shuffles to perform.

    Returns:
        np.ndarray: Array of null scores (length = n_permutations * len(clusters)).
    """
    # Flatten all post_ids from all clusters
    all_post_ids = []
    for cluster in candidate_clusters:
        all_post_ids.extend(cluster["post_ids"])

    # Get all account IDs from profile_priors (the entire universe)
    all_account_ids = list(profile_priors.keys())

    # If we have more posts than accounts, we need to cycle accounts
    # (unlikely, but safe fallback)
    if len(all_account_ids) < len(all_post_ids):
        # Extend the account list by repeating it
        all_account_ids = all_account_ids * (len(all_post_ids) // len(all_account_ids) + 1)

    all_null_scores = []

    if Config.VERBOSE:
        print(f"Building null distribution with {n_permutations} permutations (Global shuffle)...")

    # Pre-compute temporal scores for each cluster (they don't depend on account IDs)
    cluster_temporal_cache = {}
    for idx, cluster in enumerate(candidate_clusters):
        cluster_temporal_cache[idx] = compute_temporal_score(
            cluster["post_ids"], posts_dict
        )

    # Pre-compute structural scores for each cluster
    cluster_structural_cache = {}
    for idx, cluster in enumerate(candidate_clusters):
        cluster_structural_cache[idx] = min(cluster.get("raw_score", 0.0), 0.8)

    for perm_idx in tqdm(range(n_permutations), desc="Permutations", disable=not Config.VERBOSE):
        # Global shuffle of all account IDs
        shuffled_accounts = all_account_ids.copy()
        random.shuffle(shuffled_accounts)

        # Map each post to a shuffled account (cycling through the shuffled list)
        shuffled_map = {}
        for i, pid in enumerate(all_post_ids):
            shuffled_map[pid] = shuffled_accounts[i % len(shuffled_accounts)]

        # Score each cluster with the shuffled accounts
        for idx, cluster in enumerate(candidate_clusters):
            post_ids = cluster["post_ids"]

            # Get shuffled account IDs for this cluster
            shuffled_accounts_for_cluster = [shuffled_map[pid] for pid in post_ids]

            # Profile aggregate from shuffled accounts
            profile_scores = [
                profile_priors.get(acc, 0.5) for acc in shuffled_accounts_for_cluster
            ]
            profile_agg = np.mean(profile_scores) if profile_scores else 0.0

            # Use cached temporal and structural scores (they are fixed)
            temporal_score = cluster_temporal_cache[idx]
            structural_capped = cluster_structural_cache[idx]

            # Combine into null score (same weights as real scoring)
            null_score = (
                Config.PROFILE_WEIGHT * profile_agg +
                Config.TEMPORAL_WEIGHT * temporal_score +
                Config.STRUCTURAL_WEIGHT * structural_capped
            )
            all_null_scores.append(null_score)

    return np.array(all_null_scores)

# CALIBRATION & FINAL SCORING
def calibrate_clusters(
    candidate_clusters: List[Dict[str, Any]],
    bundle: DataBundle,
    n_permutations: Optional[int] = None,
    percentile_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Main orchestration function for scoring and calibration.

    Steps:
        1. Score each candidate cluster (Profile + Temporal + Structural).
        2. Build null distribution using global shuffle.
        3. Compute the threshold from the null distribution.
        4. Mark clusters as is_coordinated = True if they exceed the threshold.

    Args:
        candidate_clusters: Output from sniffers.py.
        bundle: DataBundle containing posts, profile_priors, etc.
        n_permutations: Override Config.NULL_PERMUTATIONS.
        percentile_threshold: Override Config.PERCENTILE_THRESHOLD.

    Returns:
        List of scored cluster dicts with 'is_coordinated' and 'coordination_score'.
    """
    if n_permutations is None:
        n_permutations = Config.NULL_PERMUTATIONS

    if percentile_threshold is None:
        percentile_threshold = Config.PERCENTILE_THRESHOLD

    print("\n" + "=" * 60)
    print("PHASE 4: SCORING & NULL-MODEL CALIBRATION")
    print("=" * 60)

    posts_dict = bundle.posts_dict
    profile_priors = bundle.profile_priors

    # --- Step 1: Score each cluster ---
    if Config.VERBOSE:
        print(f"Scoring {len(candidate_clusters)} candidate clusters...")

    scored_clusters = []
    for cluster in tqdm(candidate_clusters, desc="Scoring", disable=not Config.VERBOSE):
        scored = score_candidate_cluster(cluster, posts_dict, profile_priors)
        scored_clusters.append(scored)

    if Config.VERBOSE:
        print("Scoring complete.")

    # --- Step 2: Build null distribution ---
    null_scores = build_null_distribution(
        candidate_clusters,
        posts_dict,
        profile_priors,
        n_permutations=n_permutations
    )

    # --- Step 3: Compute threshold ---
    threshold = np.percentile(null_scores, percentile_threshold)
    print(f"\nNull model: {percentile_threshold}th percentile = {threshold:.4f}")
    print(f"Number of null samples: {len(null_scores)}")

    # --- Step 4: Apply threshold ---
    for cluster in scored_clusters:
        cluster["is_coordinated"] = cluster["coordination_score"] > threshold

    # Sort by coordination_score descending
    scored_clusters.sort(key=lambda x: x["coordination_score"], reverse=True)

    # Summary
    n_true = sum(1 for c in scored_clusters if c["is_coordinated"])
    n_false = len(scored_clusters) - n_true
    print(f"\n✅ Calibration complete: {n_true} clusters flagged as coordinated, {n_false} as noise.")

    # Print top 5 clusters for manual inspection
    print("\n📊 Top 5 clusters by coordination_score:")
    for i, cluster in enumerate(scored_clusters[:5]):
        status = "🔴 Coordinated" if cluster["is_coordinated"] else "⚪ Noise"
        print(f"  {i+1}. Score: {cluster['coordination_score']:.4f} | "
              f"Posts: {len(cluster['post_ids'])} | "
              f"Accounts: {len(cluster['account_ids'])} | "
              f"{status}")

    return scored_clusters