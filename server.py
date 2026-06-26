

import os
import sys
import json
import tempfile
import logging
import numpy as np
import torch
from flask import Flask, render_template, jsonify, request, send_from_directory
from functools import partial
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max upload

DASHBOARD_JSON = os.path.join('static', 'data', 'dashboard_data.json')
ENCODED_DATA_PT = 'generated_data/final_encoded_data.pt'
MODEL_WEIGHTS = 'trained_models/lstm_weigths.pth'
EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
USE_ATTENTION = True
USE_VAE = False
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_model = None
_tokenizer = None
_token2id = None


def _get_model_and_tokenizer():
    global _model, _tokenizer, _token2id

    if _model is not None:
        return _model, _tokenizer, _token2id

    logger.info("Loading model and tokenizer for upload clustering...")

    from generated_data.preprocessing import SQLTokenizer
    from models_architecture.lstm_model import LSTMENCODERDECODER

    _tokenizer = SQLTokenizer(ENCODED_DATA_PT)
    _token2id = _tokenizer.token2id
    vocab_size = len(_token2id)
    pad_idx = _token2id.get('<PAD>', 0)

    _model = LSTMENCODERDECODER(
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
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    _model.load_state_dict(state_dict)
    _model.eval()

    logger.info("Model and tokenizer loaded.")
    return _model, _tokenizer, _token2id


class SQLDataset(Dataset):
    def __init__(self, token_ids_list):
        self.data = token_ids_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(len(ids), dtype=torch.long)


def collate_fn(batch, pad_idx):
    seqs, lengths = zip(*batch)
    max_len = max(s.size(0) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, :seq.size(0)] = seq
    return padded, torch.stack(list(lengths))



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/api/dashboard')
def api_dashboard():
    """Return pre-computed dashboard data."""
    if not os.path.exists(DASHBOARD_JSON):
        return jsonify({'error': 'Dashboard data not found. Run generate_data.py first.'}), 404

    with open(DASHBOARD_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data)


@app.route('/api/cluster', methods=['POST'])
def api_cluster():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    k = request.form.get('k', type=int)
    if k is None or k < 2 or k > 100:
        return jsonify({'error': 'K must be between 2 and 100'}), 400

    try:
        # Read file content
        content = file.read().decode('utf-8', errors='replace')

        # Parse queries (same format as sql_queries.txt: separated by #$@)
        statements = content.split('#$@')
        statements = [q.replace('!#@', '').strip() for q in statements if q.strip()]

        if len(statements) < k:
            return jsonify({
                'error': f'File contains only {len(statements)} queries, but k={k}. '
                         f'K must be less than the number of queries.'
            }), 400

        if len(statements) == 0:
            return jsonify({'error': 'No valid queries found in file.'}), 400

        logger.info(f"Processing {len(statements)} queries with k={k}")

        model, tokenizer, token2id = _get_model_and_tokenizer()
        pad_idx = token2id.get('<PAD>', 0)

        df, _, _ = tokenizer.encode_batch(statements)
        token_ids_list = df['token_ids'].tolist()

        valid_mask = [len(ids) > 0 for ids in token_ids_list]
        token_ids_list = [ids for ids, v in zip(token_ids_list, valid_mask) if v]
        raw_queries = [q for q, v in zip(statements, valid_mask) if v]

        if len(raw_queries) < k:
            return jsonify({
                'error': f'Only {len(raw_queries)} valid queries after parsing. K={k} is too large.'
            }), 400

        dataset = SQLDataset(token_ids_list)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=partial(collate_fn, pad_idx=pad_idx),
        )

        all_vectors = []
        with torch.no_grad():
            for batch_ids, batch_lengths in loader:
                batch_ids = batch_ids.to(DEVICE)
                batch_lengths = batch_lengths.to(DEVICE)
                vecs = model.get_query_vector(batch_ids, batch_lengths)
                all_vectors.append(vecs.cpu().numpy())

        query_vectors = np.vstack(all_vectors)
        logger.info(f"Encoded {query_vectors.shape[0]} queries")

        # Cluster
        from workload_summarization import WorkloadSummarizer

        summarizer = WorkloadSummarizer(query_vectors=query_vectors, queries=raw_queries)
        summary_queries = summarizer.cluster_and_summarize(k=k)
        stats = summarizer.get_cluster_statistics()
        cluster_details = summarizer.get_cluster_details(n_sample_queries=10)

        result = {
            'total_queries': len(raw_queries),
            'k': k,
            'n_clusters': stats['n_clusters'],
            'compression_ratio': round(stats['compression_ratio'], 2),
            'avg_cluster_size': round(stats['avg_cluster_size'], 1),
            'std_cluster_size': round(stats['std_cluster_size'], 1),
            'clusters': cluster_details,
        }

        logger.info(f"Clustering complete: {stats['n_clusters']} clusters")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Clustering error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists(DASHBOARD_JSON):
        logger.warning(f"Dashboard data not found at {DASHBOARD_JSON}.")
        logger.warning("Feature 1 (Dashboard) will show an error.")
        logger.warning("Run 'python generate_data.py' first to generate the data.")

    logger.info("Starting WorkloadViz server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
