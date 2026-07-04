"""
This module contains all hyperparameters, paths, and weights used across the pipeline.
Instead of hard-coding values in multiple files, we centralize them here.
Changing a single value here automatically propagates to all downstream modules.
"""

import os
from typing import Dict, Optional

class Config:
    """
    Central configuration hub for the CoordiNet pipeline.
    All tunable hyperparameters and paths are defined as class attributes.
    """
    # Project root: two levels up from this file (src/ -> CoordiNet/)
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Default data paths (relative to project root)
    DATA_ROOT = os.path.join(_PROJECT_ROOT, "data")
    DEV_PATH = os.path.join(DATA_ROOT, "dev")
    EVAL_PATH = os.path.join(DATA_ROOT, "eval")

    # Allow overriding via environment variables (useful for Docker/CI)
    # Safety check: ensure environment variables are strings, not dicts
    _dev_env = os.environ.get("COORDINET_DEV_PATH")
    if _dev_env is not None and isinstance(_dev_env, str):
        DEV_PATH = _dev_env
    
    _eval_env = os.environ.get("COORDINET_EVAL_PATH")
    if _eval_env is not None and isinstance(_eval_env, str):
        EVAL_PATH = _eval_env
    #temporal parameters
    #1. maximum time diff (in seconds) between posts considered as potentially coordinated
    MAX_TIME_WINDOW_SEC=1800 #30 minutes

    #2.Bucket size for semantic grouping (Sniffer C). Within each bucket, we compute
    #pairwise cosine similarity. Smaller buckets = stricter temporal locality.
    BUCKET_SEC = 300 #5 minutes

    #embedding parameters
    # The embeddings are stored as quantized int8 to save space
    # To compute cosine similarity correctly, we scale them back to float32 using this factor.
    EMBEDDING_SCALE = 1.0/127.0 #scale factor for int8 embeddings

    SEMANTIC_SIM_THRESHOLD = 0.85 #cosine similarity threshold for semantic similarity

    #GRAPH & CLUSTRETING PARAMETERS
    #Louvain community detection may produce clusters with only 2 posts (accounts).
    #We ignore clusters smaller than this to avoid random coincidences.
    MIN_CLUSTER_SIZE = 3 #minimum number of posts in a cluster to be considered valid

    #NULL MODEL & CALIBRATION (THE STATISTICAL RIGOR)
    #We use a null model to estimate the expected number of clusters under random chance.
    NULL_PERMUTATIONS = 1000 #number of random permutations for null model

    PERCENTILE_THRESHOLD = 50 #percentile threshold for null model calibration

    #SCORING WEIGHTS (THE DECISION COMPOSITION)
    #these weights determine how much each factor contributes to the final coordination score.
    PROFILE_WEIGHT = 0.3 #suspicious profile features (e.g., new accounts, low followers, etc.)
    TEMPORAL_WEIGHT = 0.4 #temoralk burstiness (hardest to fake, most indicative of coordination)
    STRUCTURAL_WEIGHT = 0.3 #graph density and connectivity (e.g., many accounts posting the same content)

    #OUTPUT
    RESULTS_FILENAME = "coordination_results.json" #output file for coordination results

    VERBOSE = True #whether to print progress and debug info

    #HELPER METHODS
    @classmethod
    def get_data_path(cls, tier: str) -> str:
        """
        Returns the base path for the specified tier.

        Args:
            tier: "dev" or "eval"

        Returns:
            str: Absolute path to the tier folder.

        Raises:
            ValueError: If tier is not 'dev' or 'eval'.
            TypeError: If the path is not a string (safety check).
        """
        if tier == "dev":
            path = cls.DEV_PATH
        elif tier == "eval":
            path = cls.EVAL_PATH
        else:
            raise ValueError(f"Unknown tier: {tier}. Must be 'dev' or 'eval'.")

        # Safety check: ensure we return a string, not a dict or other type
        if not isinstance(path, str):
            raise TypeError(
                f"Expected path to be a string, but got {type(path).__name__}: {path}\n"
                f"Please check your Config.DEV_PATH and Config.EVAL_PATH values."
            )
        return path
        
    @classmethod
    def get_results_path(cls, tier:str="dev") -> str:
        """
        Returns the path to the results file for the specified tier (dev or eval).
        """
        data_path = cls.get_data_path(tier)
        return os.path.join(data_path, cls.RESULTS_FILENAME)

    @classmethod
    def validate_paths(cls) -> None:
        """Checks if the dev and eval paths exist. Prints warnings if not."""
        valid = True
        for tier in ["dev", "eval"]:
            path = cls.get_data_path(tier)
            if not os.path.exists(path):
                print(f"Warning: Path does not exist: {path}")
                valid = False
            else:
                print(f"Found tier: {tier} at {path}")
        return valid
    
    @classmethod
    def summary(cls)->None:
        """Prints a human-readable summary of the current configuration."""
        print("\n" + "=" * 60)
        print("COORDINET CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"DEV Path:          {cls.DEV_PATH}")
        print(f"EVAL Path:         {cls.EVAL_PATH}")
        print(f"Time Window:       {cls.MAX_TIME_WINDOW_SEC} sec ({cls.MAX_TIME_WINDOW_SEC/60:.1f} min)")
        print(f"Semantic Threshold: {cls.SEMANTIC_SIM_THRESHOLD}")
        print(f"Min Cluster Size:  {cls.MIN_CLUSTER_SIZE} posts")
        print(f"Null Permutations: {cls.NULL_PERMUTATIONS}")
        print(f"Threshold:         {cls.PERCENTILE_THRESHOLD}th percentile")
        print(f"Weights:           Profile={cls.PROFILE_WEIGHT}, "
              f"Temporal={cls.TEMPORAL_WEIGHT}, Structural={cls.STRUCTURAL_WEIGHT}")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    # If this script is run directly, print the configuration summary and validate paths.
    Config.summary()
    Config.validate_paths()