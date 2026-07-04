"""
This script executes the entire pipeline:
    1. Loads data (dev or eval)
    2. Runs the 3 Sniffers
    3. Builds the hypergraph
    4. Scores clusters and calibrates using the null model
    5. Saves results.json
    6. Generates visualizations
    7. Prints a detailed "Manual Verification Report" to the console

Usage:
    python scripts/run.py --tier eval --threshold 97.5
    python scripts/run.py --tier dev --threshold 99.0 --no-viz
"""

import os
import sys
import json
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.config import Config
from src.data_loader import load_data
from src.sniffers import run_sniffers
from src.scorer import calibrate_clusters
from src.visualize import run_visualizations


# MANUAL VERIFICATION REPORT

def print_manual_verification_report(scored_clusters, posts_dict, tier):
    """
    Print a detailed report of the flagged clusters for human inspection.
    This mimics the manual inspection loop we used in the notebook.
    """
    coord_clusters = [c for c in scored_clusters if c.get("is_coordinated", False)]

    print("\n" + "=" * 80)
    print(f"🔎 MANUAL VERIFICATION REPORT (TIER: {tier.upper()})")
    print("=" * 80)

    if not coord_clusters:
        print("❌ No clusters were flagged as coordinated.")
        print("   This could mean:")
        print("     1. The threshold is too strict (try lowering PERCENTILE_THRESHOLD).")
        print("     2. The pipeline found no signals in this tier (unlikely if it's eval).")
        return

    print(f"✅ Found {len(coord_clusters)} coordinated cluster(s).")
    print("   Inspect the raw data below to confirm these are genuine campaigns.\n")

    for idx, cluster in enumerate(coord_clusters, start=1):
        score = cluster["coordination_score"]
        post_ids = cluster["post_ids"]
        account_ids = cluster["account_ids"]
        profile_agg = cluster.get("profile_aggregate", 0.0)
        temporal_score = cluster.get("temporal_score", 0.0)

        print("=" * 80)
        print(f"📌 CLUSTER #{idx} (Score: {score:.4f})")
        print("=" * 80)
        print(f"   Accounts: {len(account_ids)} | Posts: {len(post_ids)}")
        print(f"   Profile Prior Avg: {profile_agg:.3f} | Temporal Score: {temporal_score:.3f}")
        print("-" * 80)

        # Show the actual posts in this cluster (up to 5 for readability)
        print("   📝 POSTS (Sample up to 5):")
        for i, pid in enumerate(post_ids[:5]):
            post = posts_dict.get(pid)
            if not post:
                continue

            text = post.get("text", "[NO TEXT]")
            # Truncate long text
            if len(text) > 200:
                text = text[:200] + "..."

            print(f"\n   Post ID: {pid}")
            print(f"     Account: {post.get('account_id')}")
            print(f"     Time: {post.get('created_at')}")
            print(f"     Thread ID: {post.get('thread_id')}")
            print(f"     Reply To: {post.get('reply_to_post_id')}")
            print(f"     URLs: {post.get('urls')}")
            print(f"     Mentions: {post.get('mentions')}")
            print(f"     Text: {text}")

        if len(post_ids) > 5:
            print(f"\n   ... and {len(post_ids) - 5} more posts in this cluster.")

        print("\n   🕵️  Verification Checklist for this cluster:")
        print("      - [ ] Do the posts share a RARE URL or obscure thread_id?")
        print("      - [ ] Are the accounts semi-suspicious (Profile Prior ~0.4+)?")
        print("      - [ ] Is there a tight temporal burst (Temporal Score > 0.2)?")
        print("      - [ ] Are the posts in different languages saying the same thing?")

    print("\n" + "=" * 80)
    print("✅ Manual inspection complete.")

# SAVE RESULTS
def save_results(scored_clusters, tier):
    """Save the final results.json file."""
    coord_clusters = [c for c in scored_clusters if c.get("is_coordinated", False)]
    coord_clusters.sort(key=lambda x: x["coordination_score"], reverse=True)

    # Format according to the required schema
    output = {
        "clusters": [
            {
                "post_ids": c["post_ids"],
                "is_coordinated": True,
                "coordination_score": c["coordination_score"],
            }
            for c in coord_clusters
        ]
    }

    # Save to the tier folder
    output_path = Config.get_results_path(tier)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"💾 Results saved to: {output_path}")
    return output_path

#MAIN ORCHESTRATOR
def main():
    parser = argparse.ArgumentParser(
        description="Run the CoordiNet coordination detection pipeline."
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="dev",
        choices=["dev", "eval"],
        help="Data tier to process (dev or eval).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the PERCENTILE_THRESHOLD from Config (e.g., 97.5).",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=None,
        help="Override NULL_PERMUTATIONS from Config (e.g., 500).",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip generating visualizations (faster run).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save visualizations (default: figures/).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🚀 COORDINET - Coordination Detection Pipeline")
    print("=" * 80)
    print(f"   Tier:        {args.tier.upper()}")
    print(f"   Threshold:   {args.threshold if args.threshold else 'Using Config'}")
    print(f"   Permutations:{args.permutations if args.permutations else 'Using Config'}")
    print(f"   Visualize:   {'No' if args.no_viz else 'Yes'}")
    print("=" * 80)

    # Load Data
    print("\n📂 Step 1: Loading data...")
    bundle = load_data(args.tier)

    # Run Sniffers
    print("\n🔍 Step 2: Running the 3 Sniffers...")
    graph, candidates = run_sniffers(bundle)

    if not candidates:
        print("❌ No candidate clusters found. Exiting.")
        sys.exit(1)

    # Calibrate & Score
    print("\n📊 Step 3: Scoring and Calibration...")
    scored = calibrate_clusters(
        candidates,
        bundle,
        n_permutations=args.permutations,
        percentile_threshold=args.threshold,
    )

    # Save Results
    print("\n💾 Step 4: Saving results...")
    results_path = save_results(scored, args.tier)

    # Manual Verification Report
    print("\n🔎 Step 5: Generating Manual Verification Report...")
    print_manual_verification_report(scored, bundle.posts_dict, args.tier)

    # Visualizations
    if not args.no_viz:
        print("\n📈 Step 6: Generating Visualizations...")
        # Extract the null threshold from the scored clusters
        coord_scores = [c["coordination_score"] for c in scored if c.get("is_coordinated", False)]
        non_coord_scores = [c["coordination_score"] for c in scored if not c.get("is_coordinated", False)]
        if coord_scores and non_coord_scores:
            null_threshold = (min(coord_scores) + max(non_coord_scores)) / 2
        else:
            null_threshold = Config.PERCENTILE_THRESHOLD / 100.0  # fallback

        run_visualizations(
            scored_clusters=scored,
            graph=graph,
            bundle=bundle,
            null_threshold=null_threshold,
            output_dir=args.output_dir,
            tier=args.tier,
        )
    else:
        print("\n⏭️  Skipping visualizations (--no-viz flag set).")

    # Final Summary
    coord_count = sum(1 for c in scored if c.get("is_coordinated", False))
    print("\n" + "=" * 80)
    print("✅ PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print(f"   Total Clusters Found:    {len(scored)}")
    print(f"   Flagged Coordinated:     {coord_count}")
    print(f"   Results JSON:            {results_path}")
    if not args.no_viz:
        print(f"   Visualizations:          {os.path.abspath(args.output_dir)}/")
    print("=" * 80)

    if coord_count > 0:
        print("\n🎯 Summary for PR-AUC:")
        print(f"   Precision (estimated): {coord_count}/{coord_count} = 1.0 (if all are real)")
        print(f"   Recommended threshold: {Config.PERCENTILE_THRESHOLD}th percentile")
        print("\n   Next step: Open the Manual Verification Report above")
        print("   and confirm the flagged clusters share rare URLs, threads, or cross-lingual content.")
    else:
        print("\n⚠️  No clusters flagged. Consider:")
        print("   1. Lowering PERCENTILE_THRESHOLD in Config (e.g., to 95.0).")
        print("   2. Increasing SEMANTIC_SIM_THRESHOLD if you're catching too much noise.")
        print("   3. Checking the data paths in Config.")


if __name__ == "__main__":
    main()