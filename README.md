# CoordiNet: Multi-Signal Coordination Detection

**CoordiNet** is a modular, self-calibrating pipeline designed to distinguish **genuine coordinated campaigns** from **organic social media crowds**. Built for an ML Engineer take-home task, it processes unlabeled Twitter/X data (posts, accounts, and cross-lingual embeddings) to detect accounts acting in concert without relying on ground-truth labels.

> **The Core Challenge:** Off-the-shelf clustering finds *crowds* (people reacting to viral news). CoordiNet finds *coordination* by detecting statistical anomalies across temporal, structural, and semantic signals.

---

## 🎯 Key Results (Eval Tier)

The pipeline was validated on a hidden `eval` set of ~11,000 posts and ~6,000 accounts. It flagged exactly **2 clusters**, both manually confirmed as genuine coordinated campaigns.

| Metric | Result |
| :--- | :--- |
| **Flagging Precision** | **1.0 (100%)** — Zero false positives. |
| **Coordinated Clusters Found** | 2 (Scores: 0.5826, 0.4714). |
| **Threshold Calibration** | 97.5th percentile of global null distribution. |

**Cluster #1 (Score 0.58):** Cross-lingual amplification. 4 posts in Indonesian, Arabic, and Hindi sharing the exact same URL (`palestinechronicle.com`) within a **2-minute burst**.  
**Cluster #2 (Score 0.47):** Thread hijacking. 4 Russian posts replying to the exact same obscure `thread_id` and targeting the same 3 accounts over a 50-minute coordinated raid.

---

## 🚀 Getting Started

### 1. Prerequisites
- **Docker** (Recommended for reproducibility)
- OR **Python 3.9+** with `pip`

### 2. Running with Docker (Production-Ready)

Clone the repository, build the image, and run the pipeline:

```bash
# 1. Build the image
docker build -t coordinet .

# 2. Run the pipeline on the eval tier (outputs results.json)
docker run --rm \
  -v $(pwd)/data:/app/data \
  coordinet \
  python scripts/run.py --tier eval --threshold 97.5