"""
generate_data.py
================
Runs the full DBMS workload summarization pipeline and exports all results
needed by the web dashboard into static JSON + image files.

Usage:
    python generate_data.py

Outputs:
    static/data/dashboard_data.json
    static/images/elbow_curve.png
    static/images/cluster_visualization_pca.png
    static/images/silhouette_scores.png
"""

import torch
import numpy as np
import json
import logging
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from functools import partial
import pandas as pd
from typing import List, Tuple

# Import project modules
from generated_data.preprocessing import SQLTokenizer, run_pipeline
from models_architecture.lstm_model import LSTMENCODERDECODER
from workload_summarization import WorkloadSummarizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE        = "generated_data/sql_queries.txt"
ENCODED_DATA_PT  = "generated_data/final_encoded_data.pt"
MODEL_WEIGHTS    = "trained_models/lstm_weigths.pth"

EMBED_DIM   = 64
HIDDEN_DIM  = 64
NUM_LAYERS  = 2
DROPOUT     = 0.3
USE_ATTENTION = True
USE_VAE       = False

BATCH_SIZE  = 64
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directories
STATIC_DIR       = "static"
IMAGES_DIR       = os.path.join(STATIC_DIR, "images")
DATA_DIR         = os.path.join(STATIC_DIR, "data")
OUTPUT_JSON      = os.path.join(DATA_DIR, "dashboard_data.json")


class SQLDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data: List[List[int]] = df["token_ids"].tolist()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.data[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(len(ids), dtype=torch.long)


def collate_fn(batch, pad_idx: int):
    seqs, lengths = zip(*batch)
    max_len = max(s.size(0) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, : seq.size(0)] = seq
    return padded, torch.stack(list(lengths))


def main():
    # Create output dirs
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Data Generation for Web Dashboard")
    logger.info(f"  Device: {DEVICE}")
    logger.info("=" * 60)

    # 1. Preprocess
    logger.info("Step 1: Preprocessing data...")
    df_train, token2id, id2token = run_pipeline(DATA_FILE)
    pad_idx = token2id.get("<PAD>", 0)
    vocab_size = len(token2id)
    logger.info(f"Loaded {len(df_train)} queries | vocab size: {vocab_size}")

    # 2. DataLoader
    logger.info("Step 2: Building DataLoader...")
    dataset = SQLDataset(df_train)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_fn, pad_idx=pad_idx),
    )
    logger.info(f"DataLoader ready | {len(dataset)} samples | {len(loader)} batches")

    # 3. Load model
    logger.info("Step 3: Loading LSTM model...")
    model = LSTMENCODERDECODER(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=pad_idx,
        use_attention=USE_ATTENTION,
        use_vae=USE_VAE,
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded.")

    # 4. Encode queries
    logger.info("Step 4: Encoding queries...")
    all_vectors = []
    with torch.no_grad():
        for batch_ids, batch_lengths in loader:
            batch_ids = batch_ids.to(DEVICE)
            batch_lengths = batch_lengths.to(DEVICE)
            vecs = model.get_query_vector(batch_ids, batch_lengths)
            all_vectors.append(vecs.cpu().numpy())
    query_vectors = np.vstack(all_vectors)
    logger.info(f"Encoded {query_vectors.shape[0]} queries into dim {query_vectors.shape[1]}")

    # 5. Workload summarization
    logger.info("Step 5: Running workload summarization...")
    raw_queries = df_train["sql"].tolist()

    summarizer = WorkloadSummarizer(
        query_vectors=query_vectors,
        queries=raw_queries,
    )

    # Determine optimal k (saves elbow + silhouette plots to IMAGES_DIR)
    optimal_k = summarizer.determine_optimal_k(method='elbow', output_dir=IMAGES_DIR)

    # Also save silhouette plot separately
    summarizer._plot_silhouette_scores(
        summarizer._k_values, summarizer._silhouette_scores, optimal_k, IMAGES_DIR
    )

    # Cluster and summarize
    summary_queries = summarizer.cluster_and_summarize(k=optimal_k)

    # Get statistics
    stats = summarizer.get_cluster_statistics()

    # Get per-cluster details (center query, count, 10 sample queries)
    cluster_details = summarizer.get_cluster_details(n_sample_queries=10)

    # Generate PCA visualization
    summarizer.visualize_clusters_2d(method='pca', output_dir=IMAGES_DIR)

    # 6. Build JSON output
    logger.info("Step 6: Building JSON output...")
    dashboard_data = {
        'total_queries': len(raw_queries),
        'optimal_k': int(optimal_k),
        'n_clusters': stats['n_clusters'],
        'compression_ratio': round(stats['compression_ratio'], 2),
        'avg_cluster_size': round(stats['avg_cluster_size'], 1),
        'std_cluster_size': round(stats['std_cluster_size'], 1),
        'cluster_sizes': stats['cluster_sizes'],
        'k_values': summarizer._k_values,
        'inertias': summarizer._inertias,
        'silhouette_scores': summarizer._silhouette_scores,
        'clusters': cluster_details,
        'images': {
            'elbow_curve': 'static/images/elbow_curve.png',
            'silhouette_scores': 'static/images/silhouette_scores.png',
            'cluster_visualization': 'static/images/cluster_visualization_pca.png',
        },
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Dashboard data saved to {OUTPUT_JSON}")
    logger.info(f"Images saved to {IMAGES_DIR}/")
    logger.info("=" * 60)
    logger.info("  Data generation complete!")
    logger.info("  You can now run: python server.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
