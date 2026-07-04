"""
This module contains the core detection logic:
    1. Sniffer A (Structure): Exact matches on thread_id, reply_to_post_id, quoted_post_id.
    2. Sniffer B (Entity): Shared rare URLs, mentions, and hashtags (IDF weighted).
    3. Sniffer C (Semantic): Cross-lingual embedding similarity within temporal buckets.

All edges are accumulated into a weighted NetworkX graph, then Louvain community
detection extracts candidate clusters for downstream scoring.
"""

import math
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional, Union, Any

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms.community import louvain_communities
from tqdm import tqdm

from src.config import Config
from src.data_loader import DataBundle

# TEMPORAL BUCKETING (for Sniffer C)
def build_temporal_buckets(posts_list: List[dict], bucket_sec: int) -> Dict[int, List[str]]:
    """
    Group post_ids by time buckets (e.g., 30-minute intervals).

    Used to limit pairwise semantic search to temporally close posts.

    Args:
        posts_list: List of post dicts (must have 'post_id' and 'created_at').
        bucket_sec: Width of each bucket in seconds (e.g., 1800 for 30 min).

    Returns:
        dict[bucket_start_timestamp] = list of post_ids in that bucket.
        Bucket start is the epoch second of the bucket's beginning.
    """
    buckets = defaultdict(list)
    for post in posts_list:
        post_id = post.get("post_id",None)
        dt = post.get("created_at",None)
        if dt is None or post_id is None:
            continue
        #floor the timestamp to the nearest bucket
        bucket_start = int(dt.timestamp() // bucket_sec) * bucket_sec
        buckets[bucket_start].append(post_id)

    return dict(buckets)  # convert defaultdict to regular dict for return

# SNIFFER A: STRUCTURAL OVERLAP
def sniffer_structure(posts_dict: Dict[str, dict]) -> List[Tuple[str,str,float]]:
    """
    Sniffer A: Find accounts that reply to the exact same parent post,
    share the same thread_id, or quote the same post.

    Weights:
        - reply_to_post_id: 0.8 (most deliberate signal)
        - thread_id: 0.6 (conversation tree)
        - quoted_post_id: 0.5 (weaker, but still valuable)

    Args:
        posts_dict: dict[post_id] -> post record.

    Returns:
        List of (account_i, account_j, weight) tuples.
    """
    #groupings: target ->list of (account_id, post_id) pairs
    reply_groups = defaultdict(list)
    thread_groups = defaultdict(list)
    quote_groups = defaultdict(list)

    for post_id, post in posts_dict.items():
        account_id = post["account_id"]

        reply_to = post.get("reply_to_post_id")
        if reply_to is not None:
            reply_groups[reply_to].append((account_id, post_id))

        thread_id = post.get("thread_id")
        if thread_id is not None:
            thread_groups[thread_id].append((account_id, post_id))

        quoted_post_id = post.get("quoted_post_id")
        if quoted_post_id is not None:
            quote_groups[quoted_post_id].append((account_id, post_id))

    def add_edges_from_group(group_dict: dict, weight: float) -> List[Tuple[str, str, float]]:
        """Helper: connect all pairs of distinct accounts in each group."""
        edges = []
        for target, members in group_dict.items():
            accounts = set(acc for acc, _ in members)
            if len(accounts) < 2:
                continue
            acc_list = list(accounts)
            for i in range(len(acc_list)):
                for j in range(i + 1, len(acc_list)):
                    edges.append((acc_list[i], acc_list[j], weight))
        return edges
    edges = []
    edges.extend(add_edges_from_group(reply_groups, 0.8))
    edges.extend(add_edges_from_group(thread_groups, 0.6))
    edges.extend(add_edges_from_group(quote_groups, 0.5))
    return edges

#SNIFFER B: ENTITY OVERLAP (IDF-Weighted)
def compute_idf(posts_list: List[dict], field: str) -> Dict[str, float]:
    """
    Compute Inverse Document Frequency (IDF) for a given field.

    Rare entities get higher weights, suppressing common tokens like #UNRWA.

    Args:
        posts_list: List of post dicts.
        field: One of 'mentions', 'urls', 'hashtags'.

    Returns:
        dict[entity] -> IDF weight (smoothed).
    """
    doc_freq = Counter()
    total_docs = len(posts_list)

    for post in posts_list:
        entities = post.get(field, [])
        if not entities:
            continue
        # Deduplicate within a post so a single post doesn't count multiple times
        for entity in set(entities):
            doc_freq[entity] += 1

    idf = {}
    for entity, df in doc_freq.items():
        # Smoothing to avoid log(0)
        idf[entity] = math.log((total_docs + 1) / (df + 1)) + 1.0
    return idf


def sniffer_entity(
    posts_dict: Dict[str, dict],
    posts_list: List[dict]
) -> List[Tuple[str, str, float]]:
    """
    Sniffer B: Find accounts that share rare mentions, URLs, or hashtags.

    Strategy:
        1. Compute IDF weights for each entity type.
        2. For each entity, collect all accounts that used it.
        3. Connect every pair of accounts, weight = IDF * 0.4 (capped at 0.7).

    Args:
        posts_dict: dict[post_id] -> post record.
        posts_list: list of all posts (needed for IDF).

    Returns:
        List of (account_i, account_j, weight) tuples.
    """
    # Compute IDF for each type
    if Config.VERBOSE:
        print("Computing IDF for mentions...")
    idf_mentions = compute_idf(posts_list, "mentions")

    if Config.VERBOSE:
        print("Computing IDF for URLs...")
    idf_urls = compute_idf(posts_list, "urls")

    if Config.VERBOSE:
        print("Computing IDF for hashtags...")
    idf_hashtags = compute_idf(posts_list, "hashtags")

    # Build entity -> set of accounts
    entity_to_accounts = defaultdict(set)

    for post_id, post in posts_dict.items():
        account_id = post["account_id"]

        for mention in post.get("mentions", []):
            if mention in idf_mentions:
                entity_to_accounts[("mention", mention)].add(account_id)

        for url in post.get("urls", []):
            if url in idf_urls:
                entity_to_accounts[("url", url)].add(account_id)

        for hashtag in post.get("hashtags", []):
            if hashtag in idf_hashtags:
                entity_to_accounts[("hashtag", hashtag)].add(account_id)

    # Generate edges
    edges = []
    for (entity_type, entity), accounts in entity_to_accounts.items():
        if len(accounts) < 2:
            continue

        # Retrieve IDF weight
        if entity_type == "mention":
            raw_weight = idf_mentions.get(entity, 0.0)
        elif entity_type == "url":
            raw_weight = idf_urls.get(entity, 0.0)
        elif entity_type == "hashtag":
            raw_weight = idf_hashtags.get(entity, 0.0)
        else:
            raw_weight = 0.0

        # Scale down to keep entity edges comparable to structural edges
        weight = min(0.7, raw_weight * 0.4)
        if weight < 0.05:
            continue  # ignore very low-weight entities

        acc_list = list(accounts)
        for i in range(len(acc_list)):
            for j in range(i + 1, len(acc_list)):
                edges.append((acc_list[i], acc_list[j], weight))

    return edges

# SNIFFER C: SEMANTIC SIMILARITY (Cross-Lingual)
def sniffer_semantic(
    posts_dict: Dict[str, dict],
    embeddings_dict: Dict[str, np.ndarray],
    time_buckets: Dict[int, List[str]],
) -> List[Tuple[str, str, float]]:
    """
    Sniffer C: Link accounts whose posts have high cosine similarity,
    but only if they occur within the same time bucket.

    This catches cross-lingual copy-paste campaigns that share no entities.

    Args:
        posts_dict: dict[post_id] -> post record.
        embeddings_dict: dict[post_id] -> 1024-dim float32 vector.
        time_buckets: Output from build_temporal_buckets().

    Returns:
        List of (account_i, account_j, weight) tuples.
    """
    threshold = Config.SEMANTIC_SIM_THRESHOLD
    post_to_account = {pid: posts_dict[pid]["account_id"] for pid in posts_dict}

    edges = []
    total_buckets = len(time_buckets)

    if Config.VERBOSE:
        print(f"Running Sniffer C on {total_buckets} time buckets...")

    for bucket_start, post_ids in tqdm(
        time_buckets.items(),
        desc="Semantic buckets",
        disable=not Config.VERBOSE
    ):
        if len(post_ids) < 2:
            continue

        # Filter to posts with embeddings
        valid_post_ids = [pid for pid in post_ids if pid in embeddings_dict]
        if len(valid_post_ids) < 2:
            continue

        # Build vector matrix
        vectors = np.array([embeddings_dict[pid] for pid in valid_post_ids])

        try:
            sim_matrix = cosine_similarity(vectors)
        except MemoryError:
            # Should not happen with 11k posts, but safe fallback
            print(f"Warning: MemoryError on bucket {bucket_start}, skipping.")
            continue

        # Extract upper triangular pairs exceeding threshold
        n = len(valid_post_ids)
        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_matrix[i, j]
                if sim >= threshold:
                    acc_i = post_to_account[valid_post_ids[i]]
                    acc_j = post_to_account[valid_post_ids[j]]
                    if acc_i != acc_j:
                        # Scale semantic weight to be comparable
                        weight = sim * 0.5
                        edges.append((acc_i, acc_j, weight))

    return edges

#HYPERGRAPH ASSEMBLY
def build_weighted_graph(edge_lists: List[List[Tuple[str, str, float]]]) -> nx.Graph:
    """
    Merge multiple edge lists into a single weighted NetworkX graph.

    If the same pair appears in multiple lists, their weights are summed.

    Args:
        edge_lists: A list of edge lists from different sniffers.

    Returns:
        networkx.Graph with weighted edges.
    """
    weight_dict = defaultdict(float)

    for edge_list in edge_lists:
        for u, v, w in edge_list:
            # Consistent ordering to avoid (u,v) vs (v,u) duplicates
            key = (u, v) if u < v else (v, u)
            weight_dict[key] += w

    G = nx.Graph()
    for (u, v), w in weight_dict.items():
        if w > 0.01:  # ignore negligible weights
            G.add_edge(u, v, weight=w)

    return G

#CANDIDATE EXTRACTION (Louvain Community Detection)
def extract_candidate_clusters(
    G: nx.Graph,
    posts_dict: Dict[str, dict],
    min_cluster_size: int = 3
) -> List[Dict[str, Any]]:
    """
    Run Louvain community detection to extract dense subgraphs.

    Each candidate cluster is returned as a dict with:
        - account_ids: list of unique account IDs
        - post_ids: all posts from those accounts
        - raw_score: average internal edge weight (initial ranking)

    Args:
        G: Weighted NetworkX graph.
        posts_dict: dict[post_id] -> post record.
        min_cluster_size: Minimum posts per cluster.

    Returns:
        List of candidate cluster dicts, sorted by raw_score descending.
    """
    if G.number_of_nodes() < 2:
        print("Graph has fewer than 2 nodes. No clusters found.")
        return []

    if Config.VERBOSE:
        print(f"Running Louvain on graph with {G.number_of_nodes()} nodes "
              f"and {G.number_of_edges()} edges...")

    communities = louvain_communities(G, weight="weight", seed=42)

    # Reverse map: account_id -> list of post_ids
    account_to_posts = defaultdict(list)
    for post_id, post in posts_dict.items():
        account_to_posts[post["account_id"]].append(post_id)

    candidates = []

    for comm in communities:
        comm_accounts = list(comm)
        comm_posts = []
        for acc in comm_accounts:
            comm_posts.extend(account_to_posts.get(acc, []))

        if len(comm_posts) < min_cluster_size:
            continue

        # Average internal edge weight (raw_score)
        internal_weights = []
        for i in range(len(comm_accounts)):
            for j in range(i + 1, len(comm_accounts)):
                if G.has_edge(comm_accounts[i], comm_accounts[j]):
                    internal_weights.append(
                        G[comm_accounts[i]][comm_accounts[j]]["weight"]
                    )

        avg_weight = np.mean(internal_weights) if internal_weights else 0.0

        candidates.append({
            "account_ids": comm_accounts,
            "post_ids": comm_posts,
            "raw_score": avg_weight,
        })

    # Sort by raw_score descending (higher is more promising)
    candidates.sort(key=lambda x: x["raw_score"], reverse=True)

    if Config.VERBOSE:
        print(f"Found {len(candidates)} candidate clusters (min posts = {min_cluster_size}).")

    return candidates

# ORCHESTRATOR
def run_sniffers(bundle: DataBundle) -> Tuple[nx.Graph, List[Dict[str, Any]]]:
    """
    Run all sniffers and extract candidate clusters.

    Args:
        bundle: DataBundle containing posts and embeddings.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: RUNNING THE 3 SNIFFERS")
    print("=" * 60)

    posts_dict = bundle.posts_dict
    posts_list = bundle.posts_list
    embeddings_dict = bundle.embeddings_dict

    #temporal bucketing for Sniffer C
    if Config.VERBOSE:
        print(f"Building temporal buckets (bucket size = {Config.BUCKET_SEC} sec)...")
    time_buckets = build_temporal_buckets(posts_list, Config.BUCKET_SEC)

    if Config.VERBOSE:
        print(f"Created {len(time_buckets)} time buckets.")
    
    #run sniffer A (stuctural)
    if Config.VERBOSE:
        print("Running Sniffer A (Structural Overlap)...")
    edges_A = sniffer_structure(posts_dict)
    if Config.VERBOSE:
        print(f"Sniffer A produced {len(edges_A)} edges.")
    
    #run sniffer B (entity)
    if Config.VERBOSE:
        print("Running Sniffer B (Entity Overlap)...")
    edges_B = sniffer_entity(posts_dict, posts_list)
    if Config.VERBOSE:
        print(f"Sniffer B produced {len(edges_B)} edges.")
    
    #run sniffer C (semantic)
    if Config.VERBOSE:
        print("Running Sniffer C (Semantic Similarity)...")
    edges_C = sniffer_semantic(posts_dict, embeddings_dict, time_buckets)
    if Config.VERBOSE:
        print(f"Sniffer C produced {len(edges_C)} edges.")

    #build weighted graph
    if Config.VERBOSE:
        print("Building weighted graph from all edges...")
    G = build_weighted_graph([edges_A, edges_B, edges_C])
    if Config.VERBOSE:
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    #extract candidate clusters
    if Config.VERBOSE:
        print("Extracting candidate clusters using Louvain community detection...")
    candidates = extract_candidate_clusters(G, posts_dict, Config.MIN_CLUSTER_SIZE)
    if Config.VERBOSE:
        print(f"Extracted {len(candidates)} candidate clusters.")
    return G, candidates

if __name__ == "__main__":
    # Example usage
    from src.data_loader import DataBundle, load_data

    # Load a sample data bundle (replace with actual paths)
    bundle = load_data(tier="eval")  # or "eval"

    # Run sniffers and extract candidates
    G, candidates = run_sniffers(bundle)

    print(f"\nTotal candidate clusters found: {len(candidates)}")
    #print top clusters and their scores
    for idx, cluster in enumerate(candidates[:3]):  # show top 3
        print(f"Cluster {idx + 1}:")
        print(f"  Accounts: {cluster['account_ids']}")
        print(f"  Posts: {cluster['post_ids']}")
        print(f"  Raw Score: {cluster['raw_score']:.4f}")
    print("Done.")
