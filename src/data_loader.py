"""
This module handles all data loading from the dev/eval tiers.
It reads accounts.jsonl, posts.jsonl, and embeddings.parquet,
computes a "profile_prior" (suspicion score) for each account,
and returns a unified DataBundle object for downstream processing.
All paths are sourced from the central Config class.
"""

import json
import os
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from src.config import Config


#profile prior computation: we assign a suspicion score to each account based on its profile features.
#The score is a weighted sum of various features, normalized to [0,1].

def compute_profile_prior(profile: dict, account_id:str)->float:
    """
    Compute a suspicion score for an account based on its profile features.
    The score is a weighted sum of various features, normalized to [0,1].
    Higher scores indicate more suspicious accounts.

    Args:
        profile (dict): The account's profile data.
        account_id (str): The unique identifier for the account.

    Returns:
        float: A suspicion score in the range [0, 1].
    """
    if not profile:
        return 0.95 # completely suspicious if profile is missing
    metadata = profile.get("metadata", {})
    followers = metadata.get("followers_count", 0)
    posts = metadata.get("posts_count", 0)
    verified = metadata.get("verified", False)

    score = 0.0

    #signal 1: empty metdata (no followers, no posts, not verified)
    if not metadata:
        score += 0.4
    
    #signal 2: post-to-follower ratio (high ratio is suspicious)
    if followers > 0 and posts > 0:
        ratio = posts / followers
        if ratio > 50: # arbitrary threshold for suspiciously high posting activity
            score += 0.5
        elif ratio > 20:
            score += 0.3
        elif ratio > 10:
            score += 0.1
    elif followers == 0 and posts > 100:
        score += 0.7 # very suspicious: many posts but no followers
    
    #signal 3: verified status (unverified accounts are more suspicious)
    if verified:
        score -= 0.2 # reduce suspicion for verified accounts
    
    #signal 4: very high follower accounts are usually organic influencers, so we reduce their suspicion score
    if followers> 50000:
        score -= 0.3
    elif followers> 10000:
        score -= 0.1
    
    #signal 5: account age (new accounts are more suspicious)
    #We check the account creation date and assign a score based on how new the account is.
    created_at = metadata.get("created_at", None)
    if created_at:
        try:
            created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
            account_age_days = (datetime.utcnow() - created_date).days
            if account_age_days < 30:
                score += 0.5 # very new account
            elif account_age_days < 180:
                score += 0.2 # moderately new account
        except ValueError:
            pass # ignore invalid date formats

    #normalize score to [0,1]
    return max(0.0, min(1.0, score))


#dataloaders

def load_accounts(filepath: str) -> Tuple[Dict[str, dict], Dict[str, float]]:
    """
    Load accounts from a JSONL file and compute profile priors.

    Args:
        filepath (str): Path to the accounts.jsonl file.

    Returns:
        Tuple[Dict[str, dict], Dict[str, float]]: A tuple containing:
            - A dictionary mapping account_id to account profile data.
            - A dictionary mapping account_id to its computed profile prior score.
    """
    #check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Accounts file not found: {filepath}") 
    
    accounts_profiles = {}
    profile_priors = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading accounts")):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                account_id = data.get("account_id")
                if account_id is None:
                    print(f"Warning: Missing account_id in line {line_num}. Skipping.")
                    continue
                profile = data.get("profile", {})
                accounts_profiles[account_id] = profile
                profile_priors[account_id] = compute_profile_prior(profile, account_id)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error in line {line_num}: {e}. Skipping.")
                continue
    return accounts_profiles, profile_priors

def load_posts(filepath: str) -> Tuple[Dict[str, dict], List[dict]]:
    """
    Load posts from a JSONL file.

    Args:
        filepath (str): Path to the posts.jsonl file.
    """
    #check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Posts file not found: {filepath}")

    posts_dict = {}
    posts_list = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading posts", disable=not Config.VERBOSE), start=1):
            line = line.strip()
            if not line:
                continue

            try:
                post = json.loads(line)
                post_id = post.get("post_id")
                if post_id is None:
                    print(f"Warning: Missing post_id in line {line_num}. Skipping.")
                    continue

                # Convert timestamp to datetime
                if "created_at" in post:
                    post["created_at"] = datetime.fromisoformat(post["created_at"])
                else:
                    post["created_at"] = None

                # Ensure all interaction fields exist (even if empty)
                post.setdefault("mentions", [])
                post.setdefault("hashtags", [])
                post.setdefault("urls", [])
                post.setdefault("reply_to_post_id", None)
                post.setdefault("thread_id", None)
                post.setdefault("quoted_post_id", None)

                posts_dict[post_id] = post
                posts_list.append(post)

            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_num} in {filepath}")
                continue
            except ValueError as e:
                print(f"Warning: Timestamp parsing error on line {line_num}: {e}")
                continue

    return posts_dict, posts_list

def load_embeddings(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from a Parquet file.

    Args:
        filepath (str): Path to the embeddings.parquet file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Embeddings file not found: {filepath}")

    # Read with pyarrow
    table = pq.read_table(filepath)
    df = table.to_pandas()

    # Identify the vector column
    possible_vector_cols = ["vector", "embedding", "embeddings", "vec"]
    vector_col = None
    for col in possible_vector_cols:
        if col in df.columns:
            vector_col = col
            break

    if vector_col is None:
        raise ValueError(
            f"Could not identify vector column. Found columns: {df.columns.tolist()}"
        )

    if "post_id" not in df.columns:
        raise ValueError("Embeddings file missing 'post_id' column.")

    embeddings_dict = {}
    scale = Config.EMBEDDING_SCALE

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading embeddings", disable=not Config.VERBOSE):
        post_id = row["post_id"]
        vec = row[vector_col]

        # Convert to float32 numpy array
        if isinstance(vec, (np.ndarray, list)):
            vec = np.array(vec, dtype=np.float32) * scale
        else:
            # Fallback: treat as bytes
            try:
                vec = np.frombuffer(vec, dtype=np.int8).astype(np.float32) * scale
            except Exception as e:
                print(f"Warning: Could not parse vector for post {post_id}: {e}")
                continue

        # Validate dimension
        if len(vec) != 1024:
            print(f"Warning: post {post_id} has vector dimension {len(vec)}, expected 1024. Skipping.")
            continue

        embeddings_dict[post_id] = vec

    return embeddings_dict

#data bundle to hold all loaded data together
@dataclass
class DataBundle:
    """
    A container for all loaded data: accounts, posts, embeddings, and profile priors.
    """
    accounts_profiles: Dict[str, dict] = field(default_factory=dict)
    profile_priors: Dict[str, float] = field(default_factory=dict)
    posts_dict: Dict[str, dict] = field(default_factory=dict)
    posts_list: List[dict] = field(default_factory=list)
    embeddings_dict: Dict[str, np.ndarray] = field(default_factory=dict)
    tier: str = "dev"  # default tier

    def __post_init__(self):
        """Compute basic statistics after initialization."""
        self.n_accounts = len(self.accounts_profiles)
        self.n_posts = len(self.posts_list)
        self.n_embeddings = len(self.embeddings_dict)

        #check for missing embeddings
        self.posts_without_embeddings = sum(1 for post in self.posts_list if post["post_id"] not in self.embeddings_dict)

    def summary(self) -> None:
        """Print a human-readable summary of the loaded data."""
        print("\n" + "=" * 60)
        print(f"DATA BUNDLE SUMMARY (TIER: {self.tier.upper()})")
        print("=" * 60)
        print(f"Accounts:          {self.n_accounts}")
        print(f"Posts:             {self.n_posts}")
        print(f"Embeddings:        {self.n_embeddings}")
        print(f"Posts w/o Embed:   {self.posts_without_embeddings}")
        if self.posts_without_embeddings > 0:
            print("These posts will be excluded from Sniffer C (Semantic).")
        print("=" * 60 + "\n")

#orchestration function to load all data for a given tier
def load_data(tier: str = "dev") -> DataBundle:
    """
    Load all data (accounts, posts, embeddings) for the specified tier.

    Args:
        tier (str): The data tier to load ("dev" or "eval").

    Returns:
        DataBundle: A container with all loaded data.
    """
    if tier not in ["dev", "eval"]:
        raise ValueError(f"Unknown tier: {tier}. Must be 'dev' or 'eval'.")

    base_path = Config.get_data_path(tier)
    print(f"\nLoading data from: {base_path}")

    # step 1 is to load accounts
    accounts_path = os.path.join(base_path, "accounts.jsonl")
    account_profiles, profile_priors = load_accounts(accounts_path)

    # step 2 is to load posts
    posts_path = os.path.join(base_path, "posts.jsonl")
    posts_dict, posts_list = load_posts(posts_path)

    # step 3 is to load embeddings
    embeddings_path = os.path.join(base_path, "embeddings.parquet")
    embeddings_dict = load_embeddings(embeddings_path)

    # step 4 is to bundle everything
    bundle = DataBundle(
        tier=tier,
        accounts_profiles=account_profiles,
        profile_priors=profile_priors,
        posts_dict=posts_dict,
        posts_list=posts_list,
        embeddings_dict=embeddings_dict,
    )

    # step 5 is to print summary
    bundle.summary()

    return bundle

if __name__ == "__main__":
    # If this script is run directly, load the dev tier and print a summary.
    from src.config import Config
    from src.data_loader import load_data

    Config.validate_paths()
    Config.summary()

    data_bundle = load_data(tier="dev")

    print(f"Loaded {data_bundle.n_posts} posts.")
    print(f"Profile prior for account X: {data_bundle.profile_priors.get('acct_123', 0.5)}")