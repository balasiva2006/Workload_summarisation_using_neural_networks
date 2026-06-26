import torch
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Tuple, Optional

# Import custom modules
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


def preprocess_data(file_path: str) -> Tuple[pd.DataFrame, dict, dict]:
    logger.info(f"Loading and preprocessing data from: {file_path}")
    df, token2id, id2token = run_pipeline(file_path)
    logger.info(f"Loaded {len(df)} queries  |  vocab size: {len(token2id)}")
    return df, token2id, id2token



class SQLDataset(Dataset):

    def __init__(self, df: pd.DataFrame):
        self.data: List[List[int]] = df["token_ids"].tolist()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.data[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(len(ids), dtype=torch.long)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int):
    """Dynamically pad each batch to its longest sequence."""
    seqs, lengths = zip(*batch)
    max_len = max(s.size(0) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, : seq.size(0)] = seq
    return padded, torch.stack(list(lengths))


def build_dataloader(df: pd.DataFrame, pad_idx: int) -> DataLoader:
    from functools import partial
    dataset = SQLDataset(df)
    loader  = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = partial(collate_fn, pad_idx=pad_idx),
    )
    logger.info(f"DataLoader ready  |  {len(dataset)} samples  |  {len(loader)} batches")
    return loader


def load_model(vocab_size: int, pad_idx: int, weights_path: str) -> LSTMENCODERDECODER:
    """Instantiate LSTMENCODERDECODER and load pre-trained weights."""
    logger.info(f"Loading model weights from: {weights_path}")

    model = LSTMENCODERDECODER(
        vocab_size    = vocab_size,
        embed_dim     = EMBED_DIM,
        hidden_dim    = HIDDEN_DIM,
        num_layers    = NUM_LAYERS,
        dropout       = DROPOUT,
        pad_idx       = pad_idx,
        use_attention = USE_ATTENTION,
        use_vae       = USE_VAE,
    ).to(DEVICE)

    checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and any(k.endswith(".weight") or k.endswith(".bias")
                                               for k in checkpoint):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded successfully and set to eval mode.")
    return model


def encode_queries(model: LSTMENCODERDECODER, loader: DataLoader) -> np.ndarray:

    logger.info("Encoding queries with LSTM encoder …")
    all_vectors = []

    with torch.no_grad():
        for batch_ids, batch_lengths in loader:
            batch_ids     = batch_ids.to(DEVICE)
            batch_lengths = batch_lengths.to(DEVICE)

            vecs = model.get_query_vector(batch_ids, batch_lengths)
            all_vectors.append(vecs.cpu().numpy())

    query_vectors = np.vstack(all_vectors)          # (N, hidden_dim*2)
    logger.info(f"Encoded {query_vectors.shape[0]} queries into vectors of dim {query_vectors.shape[1]}")
    return query_vectors


def run_workload_summarization(
    query_vectors : np.ndarray,
    raw_queries   : List[str],
    method        : str = "elbow",
) -> Tuple[List[str], dict]:
    logger.info(f"Starting workload summarization  |  method='{method}'")

    summarizer = WorkloadSummarizer(
        query_vectors = query_vectors,
        queries       = raw_queries,
    )

    optimal_k      = summarizer.determine_optimal_k(method=method)
    summary_queries = summarizer.cluster_and_summarize(k=optimal_k)
    stats           = summarizer.get_cluster_statistics()

    logger.info("─" * 60)
    logger.info(f"  Original workload size : {len(raw_queries)}")
    logger.info(f"  Summary workload size  : {len(summary_queries)}")
    logger.info(f"  Compression ratio      : {stats['compression_ratio']:.2f}%")
    logger.info(f"  Avg cluster size       : {stats['avg_cluster_size']:.1f}")
    logger.info("─" * 60)

    logger.info("\nSample summary queries:")
    for i, q in enumerate(summary_queries[:5], 1):
        logger.info(f"  [{i}] {q[:120]}")

    return summary_queries, stats


def main():
    logger.info("=" * 60)
    logger.info("  DBMS Workload Summarization Pipeline")
    logger.info(f"  Device: {DEVICE}")
    logger.info("=" * 60)

    # 1. Preprocess
    df_train, token2id, id2token = preprocess_data(DATA_FILE)
    pad_idx   = token2id.get("<PAD>", 0)
    vocab_size = len(token2id)

    # 2. DataLoader
    loader = build_dataloader(df_train, pad_idx)

    # 3. Load trained LSTM
    model = load_model(vocab_size, pad_idx, MODEL_WEIGHTS)

    # 4. Encode all queries to latent vectors
    query_vectors = encode_queries(model, loader)

    # 5. Workload summarization
    raw_queries = df_train["sql"].tolist()
    summary_queries, stats = run_workload_summarization(
        query_vectors = query_vectors,
        raw_queries   = raw_queries,
        method        = "elbow",       # or "silhouette"
    )

    # 6. Persist results
    output_path = Path("workload_summary.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for q in summary_queries:
            f.write(q + "\n")
    logger.info(f"Summary queries saved to '{output_path}'")


if __name__ == "__main__":
    main()
