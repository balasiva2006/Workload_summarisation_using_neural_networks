# SQL Query Embedding & Workload Analytics

> **Learn vector representations of SQL queries — then use them for workload summarization and runtime prediction.**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Team](#team)
- [Repository Structure](#repository-structure)
- [Pipeline at a Glance](#pipeline-at-a-glance)
- [Key Concepts & Terminology](#key-concepts--terminology)
- [Notebooks — What Each One Does](#notebooks--what-each-one-does)
- [Artifact Dependency Map](#artifact-dependency-map)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Design Decisions & Trade-offs](#design-decisions--trade-offs)
- [Future Work](#future-work)

---

## Project Overview

Modern database systems execute thousands of SQL queries. Understanding **what those queries look like structurally** — without caring about specific literal values — enables two high-value downstream tasks:

| Task | What it solves |
|---|---|
| **Workload Summarization** | Compress N queries → K representative queries for cheaper query tuning / index selection |
| **Runtime Prediction** | Predict query execution time in ms from the query text alone |

To do this we **learn embeddings** — dense vector representations — of SQL queries using three independent approaches, then apply those vectors to both tasks.

```
SQL Text ──► Tokenize & Encode ──► Embedding Model ──► Vector ──► Clustering / Regression
```

---

## Team

| Member | Primary Contribution |
|---|---|
| Member 1 | Data preprocessing pipeline, vocabulary construction (`data_preprocessing.ipynb`) |
| Member 2 | LSTM Autoencoder training (`dbms-project-lstm-training-model.ipynb`) |
| Member 3 | LSTM inference pipeline + Doc2Vec training & embeddings |
| Member 4 | BERT fine-tuning via MLM (`bert_implementation.ipynb`) |
| Member 5 | Workload summarization + runtime prediction notebooks |

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_data_preprocessing.ipynb            # Corpus collection, tokenization, vocabulary
│   ├── 02_lstm_training.ipynb                 # LSTM Autoencoder training
│   ├── 03_lstm_embeddings_creator.ipynb       # LSTM inference on new SQL datasets
│   ├── 04_doc2vec.ipynb                       # Doc2Vec model training
│   ├── 05_doc2vec_embeddings.ipynb            # Doc2Vec inference + similarity experiments
│   ├── 06_bert_implementation.ipynb           # BERT MLM fine-tuning + CLS extraction
│   ├── 07_workload_summarization.ipynb        # K-Means clustering + representative selection
│   └── 08_query_runtime_prediction.ipynb      # Regression: embeddings → runtime
│
├── artifacts/                                 # Generated files (see table below)
│   ├── final_encoded_data.pt
│   ├── lstm_embeds.npy
│   ├── doc2vec_embeds.npy
│   ├── bert_embeds.npy
│   ├── best_model.pth
│   └── doc2vec_final.model
│
├── data/
│   ├── sdss_median.csv
│   └── sqlstorm/                              # StackOverflow, TPC-H, TPC-DS, JOB splits
│
├── requirements.txt
└── README.md
```

---

## Pipeline at a Glance

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1 — Build Corpus & Vocabulary            [Notebook 01]        │
│                                                                     │
│  SDSS logs + SQLStorm datasets                                      │
│       │                                                             │
│       ▼                                                             │
│  Normalize → Tokenize (sqlparse) → Mask Literals                    │
│       │                                                             │
│       ▼                                                             │
│  token2id / id2token  ──►  final_encoded_data.pt  (reused by ALL)  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┼──────────────────────┐
          ▼                    ▼                       ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│  LSTM Autoencoder│ │    Doc2Vec       │ │   BERT (MLM finetune)│
│  [NB 02 + 03]    │ │  [NB 04 + 05]   │ │      [NB 06]         │
│                  │ │                  │ │                      │
│  d = 128         │ │  d = 64          │ │  d = 768  (CLS)      │
│  lstm_embeds.npy │ │ doc2vec_embeds   │ │  bert_embeds.npy     │
└────────┬─────────┘ └───────┬──────────┘ └──────────┬───────────┘
         └───────────────────┼────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌─────────────────────┐      ┌────────────────────────┐
   │ Workload             │      │ Runtime Prediction      │
   │ Summarization        │      │                        │
   │ [NB 07]              │      │ [NB 08]                │
   │                      │      │                        │
   │ K-Means → pick 1     │      │ Ridge Regression       │
   │ query/cluster        │      │ R², MAE, RMSE, MAPE    │
   └─────────────────────┘      └────────────────────────┘
```

---

## Key Concepts & Terminology

This section is written to be useful regardless of whether your background is engineering or data science.

### Data & Preprocessing

| Term | Plain English | Formal Definition |
|---|---|---|
| **Workload** | The full set of SQL queries we are working with | A set of $N$ SQL query strings |
| **Tokenization** | Splitting a query into meaningful pieces | Converts `SELECT age FROM users` → `['SELECT', 'age', 'FROM', 'users']` |
| **Literal Masking** | Hiding specific values so the model learns structure, not data | Replaces `42` → `NUM_LITERAL`, `'alice'` → `STR_LITERAL` |
| **Vocabulary** | The dictionary mapping tokens to integers | $\text{token2id}: \text{token} \rightarrow \mathbb{N}$, $\text{id2token}: \mathbb{N} \rightarrow \text{token}$ |
| **Padding** | Making all sequences the same length for batching | Append `<PAD>` tokens until length $= L_{\max}$ |
| **Encoding** | Converting a token list into a list of integers | `['SELECT', 'age']` → `[4, 17]` using token2id |

### Embedding Models

| Term | Plain English | Technical Detail |
|---|---|---|
| **Embedding** | A point in vector space that represents a query | A vector $\mathbf{v} \in \mathbb{R}^{d}$ |
| **Autoencoder** | A model that learns to compress then reconstruct its own input | Encoder maps input → bottleneck vector; Decoder reconstructs input from it |
| **Teacher Forcing** | A training trick: sometimes feed the true previous token to the decoder instead of its own prediction | Applied with probability $p = 0.5$ during LSTM decoder training |
| **Doc2Vec** | A classical NLP method that learns a vector per document | Extends Word2Vec by adding a document-level id to the context window |
| **MLM (Masked Language Modeling)** | BERT's pre-training task: predict randomly hidden tokens | 15% of tokens are masked; BERT learns to fill them in using surrounding context |
| **CLS token** | BERT's special first token whose output vector summarizes the whole sequence | We use `last_hidden_state[:, 0, :]` — dimension 768 — as our query embedding |

### Downstream Tasks

| Term | Plain English | Technical Detail |
|---|---|---|
| **K-Means Clustering** | Group similar queries together | Minimizes within-cluster inertia: $\sum_{c}\sum_{i \in c} \lVert \mathbf{v}_i - \boldsymbol{\mu}_c \rVert_2^2$ |
| **Workload Summarization** | Pick one query per cluster to represent all queries in it | $i^* = \arg\min_{i \in c} \lVert \mathbf{v}_i - \boldsymbol{\mu}_c \rVert_2$ — the query closest to the cluster centroid |
| **Compression Ratio** | How much smaller the summary is than the original workload | $\left(1 - \frac{K}{N}\right) \times 100\%$ |
| **Runtime Prediction** | Supervised regression: predict how long a query takes from its embedding | Features: embedding vector; Target: runtime in ms; Metric: $R^2$, MAE, RMSE |

---

## Notebooks — What Each One Does

### Notebook 01 — `data_preprocessing.ipynb`
**Goal:** Build a clean SQL corpus and a shared vocabulary that every downstream notebook reuses.

**Steps:**
1. Collected SQL queries from two source families:
   - SDSS `median.csv` — real astronomy database query logs
   - SQLStorm — StackOverflow, TPC-H, TPC-DS, JOB benchmark sets
2. Cleaned raw data (removed broken rows, filtered error labels)
3. Normalized SQL: lowercased, collapsed whitespace
4. Tokenized using `sqlparse`
5. Masked literals: numeric → `NUM_LITERAL`, string → `STR_LITERAL`; added detection for edge cases (scientific notation, signed numbers)
6. Built vocabulary with min-frequency filtering (tested thresholds 2, 3, 5, 10)
7. Added special tokens: `<PAD>`, `<START>`, `<UNK>`
8. Encoded all queries to integer id sequences

**Output:** `artifacts/final_encoded_data.pt`  
```
{
  "encoded_queries": List[List[int]],
  "token2id": Dict[str, int],
  "id2token": Dict[int, str]
}
```

---

### Notebook 02 — `dbms-project-lstm-training-model.ipynb`
**Goal:** Train an LSTM Encoder–Decoder (autoencoder) to produce a 128-dimensional vector for each SQL query.

**Architecture:**

```
Input sequence (padded)
        │
  [Embedding layer, dim=64]
        │
  [Bidirectional LSTM, hidden=64, layers=2, dropout=0.3]
        │
  Concatenate forward + backward hidden states
        │
  Query vector ∈ ℝ^128     ← this is what we care about
        │
  [LSTM Decoder with attention]
        │
  Reconstructed token sequence
```

**Training details:**

| Hyperparameter | Value |
|---|---|
| Embedding dim | 64 |
| LSTM hidden dim | 64 (bidirectional → 128 output) |
| LSTM layers | 2 |
| Dropout | 0.3 |
| Teacher forcing ratio | 0.5 |
| Optimizer | AdamW, lr = 0.001 |
| Scheduler | Cosine Annealing Warm Restarts |
| Gradient clipping | 1.0 |
| Train / Val split | 90% / 10% |

**Output:** `artifacts/best_model.pth`, per-epoch checkpoints, training config JSON

---

### Notebook 03 — `lstm_embeddings_creator.ipynb`
**Goal:** Apply the trained LSTM to a new SQL dataset and save embeddings.

**Steps:**
1. Loaded raw SQL (e.g., SQLStorm StackOverflow) into a DataFrame
2. Applied the same preprocessing pipeline as Notebook 01 (normalize → mask → encode)
3. Loaded `final_encoded_data.pt` for `token2id`
4. Loaded trained LSTM weights from `best_model.pth`
5. Batched and padded sequences, extracted embeddings via `get_query_vector`

**Output:** `artifacts/lstm_embeds.npy` — shape `(N, 128)`

---

### Notebook 04 — `doc2vec.ipynb`
**Goal:** Train a Doc2Vec model as a classical NLP baseline.

**Config:**

| Parameter | Value |
|---|---|
| Vector size | 64 |
| Window | 5 |
| Min count | 2 |
| Epochs | 80 |
| Checkpoint every | 10 epochs |

**Output:** `artifacts/doc2vec_final.model`

---

### Notebook 05 — `doc2vec_embeddings.ipynb`
**Goal:** Infer Doc2Vec embeddings and validate them qualitatively.

**Steps:**
1. Loaded `doc2vec_final.model`
2. Preprocessed SQL to masked token lists
3. Used `infer_vector(tokens, epochs=20)`
4. Saved embeddings
5. Ran t-SNE visualization, cosine similarity heatmaps, top-K similar query search

**Output:** `artifacts/doc2vec_embeds.npy` — shape `(N, 64)`

---

### Notebook 06 — `bert_implementation.ipynb`
**Goal:** Fine-tune `bert-base-uncased` on the SQL corpus using Masked Language Modeling, then extract CLS embeddings (dim = 768).

**Steps:**
1. Reconstructed token strings from encoded ids (skipping `<PAD>`, `<START>`, etc.)
2. Initialized `bert-base-uncased` tokenizer; added custom tokens: `NUM_LITERAL`, `STR_LITERAL`, `HEX_LITERAL`
3. Built a PyTorch Dataset returning BERT-tokenized inputs
4. Fine-tuned with MLM:

| Hyperparameter | Value |
|---|---|
| Masking probability | 15% |
| Epochs | 4 |
| Batch size | 32 |
| Learning rate | 5 × 10⁻⁵ |
| Mixed precision | fp16 enabled |

5. Extracted CLS token from `last_hidden_state[:, 0, :]`
6. Packaged fine-tuned model + tokenizer into a zip

**Output:** Fine-tuned BERT checkpoint directory, `artifacts/bert_embeds.npy` — shape `(N, 768)`

---

### Notebook 07 — `workload_summarization.ipynb`
**Goal:** Compress a workload of N queries to K representative queries using clustering on LSTM embeddings.

**Steps:**
1. Loaded and preprocessed `sql_queries.txt`
2. Generated LSTM embeddings → matrix $V \in \mathbb{R}^{N \times 128}$
3. Evaluated multiple K values using:
   - Inertia (elbow method)
   - Silhouette score
   - Calinski–Harabasz index
   - Davies–Bouldin index
4. Ran K-Means with chosen K
5. Selected representative query per cluster:

$$i^* = \arg\min_{i \in c} \lVert \mathbf{v}_i - \boldsymbol{\mu}_c \rVert_2$$

6. Reported compression ratio $= \left(1 - \frac{K}{N}\right) \times 100\%$
7. Visualized cluster layout via PCA (2D projection)

**Output:** Summary query list, cluster statistics, PCA plots

---

### Notebook 08 — `query_runtime_prediction.ipynb`
**Goal:** Test whether embeddings encode enough structural information to predict query execution time.

**Steps:**
1. Loaded all three embedding types (LSTM, Doc2Vec, BERT)
2. Loaded ground-truth runtimes from JSONL; joined by `query_id`
3. Train / Val / Test split
4. Trained Ridge Regression on each embedding type separately
5. Evaluated on held-out test set

**Metrics reported:**

| Metric | Meaning |
|---|---|
| $R^2$ | Fraction of variance explained (higher is better) |
| MAE | Mean Absolute Error in ms |
| RMSE | Root Mean Squared Error in ms |
| MAPE | Mean Absolute Percentage Error |

**Output:** Metric comparison table, predicted vs. actual plots, residual distribution plots

---

## Artifact Dependency Map

| Artifact | Produced by | Consumed by |
|---|---|---|
| `final_encoded_data.pt` | NB 01 | NB 02, 03, 06, 07 — used everywhere |
| `best_model.pth` | NB 02 | NB 03, 07 |
| `lstm_embeds.npy` | NB 03 | NB 08 |
| `doc2vec_final.model` | NB 04 | NB 05 |
| `doc2vec_embeds.npy` | NB 05 | NB 08 |
| BERT checkpoint directory | NB 06 | NB 06 (inference), NB 08 |
| `bert_embeds.npy` | NB 06 | NB 08 |

> ⚠️ **Run notebooks in order.** Notebooks 02–08 all depend on `final_encoded_data.pt` from Notebook 01.

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU recommended for Notebook 02 (LSTM training) and Notebook 06 (BERT fine-tuning)

### Install dependencies

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

### Key dependencies

```
torch
transformers
gensim
sqlparse
scikit-learn
numpy
pandas
matplotlib
seaborn
```

### Data setup

Place source data files as follows before running Notebook 01:

```
data/
├── sdss_median.csv
└── sqlstorm/
    ├── stackoverflow.sql
    ├── tpch.sql
    ├── tpcds.sql
    └── job.sql
```

---

## How to Run

Run notebooks sequentially:

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

Or if you only want specific outputs:

| If you want... | Run these notebooks |
|---|---|
| Just the vocabulary + encoded data | 01 |
| LSTM embeddings for a new SQL file | 01 → 02 → 03 |
| Doc2Vec embeddings | 01 → 04 → 05 |
| BERT embeddings | 01 → 06 |
| Workload summarization | 01 → 02 → 07 |
| Runtime prediction (all models) | 01 → 02 → 03 → 04 → 05 → 06 → 08 |

---

## Results Summary

### Embedding Dimensions

| Model | Dimension | Training Objective |
|---|---|---|
| LSTM Autoencoder | 128 | Sequence reconstruction (cross-entropy) |
| Doc2Vec | 64 | Distributed memory / bag of words |
| Fine-tuned BERT | 768 | Masked Language Modeling (15% mask rate) |

### Workload Summarization

- Input workload: N queries
- After K-Means with optimal K (selected via elbow + silhouette):
  - Compression ratio reported per dataset
  - Representative queries selected as nearest-to-centroid per cluster

### Runtime Prediction

All three embedding types evaluated with Ridge Regression on the same train/test split. Metric tables and plots are in Notebook 08.

---

## Design Decisions & Trade-offs

### Why literal masking?
Without it, embeddings of `SELECT * FROM t WHERE age = 25` and `SELECT * FROM t WHERE age = 30` would look different despite being structurally identical. Masking teaches models to encode **query shape**, not **data values**.

### Why three embedding approaches?
Each captures different inductive biases:
- **Doc2Vec**: fast, interpretable baseline; no sequential modeling
- **LSTM Autoencoder**: explicitly models token order; reconstruction objective forces the vector to carry enough information to reproduce the query
- **BERT**: bidirectional context via attention; MLM fine-tuning adapts it to SQL syntax

### Why bidirectional LSTM?
SQL has long-range dependencies (`SELECT ... FROM ... WHERE ...`). Bidirectional encoding lets the encoder see both left and right context at each position, then we concatenate both directions → 128-dim vector from 64-dim hidden state.

### Why CLS token for BERT?
CLS is specifically trained to aggregate sequence-level information. It is the standard BERT practice for sentence/document classification and representation tasks.

### Why K-Means for workload summarization?
Simple, scalable, and interpretable. The elbow + silhouette combination gives a principled way to pick K without manual tuning per dataset.

---

## Future Work

- [ ] Evaluate on **index selection** and **query plan optimization** tasks using the summary queries
- [ ] Try **contrastive learning** (e.g., SimCSE) to learn embeddings where structurally similar queries are explicitly pulled together
- [ ] Replace Ridge Regression in runtime prediction with **gradient boosted trees** (XGBoost / LightGBM) or a small MLP
- [ ] Build a **SQL-specific tokenizer** instead of relying on `sqlparse` + BERT subword tokenization mismatch
- [ ] Expose a **REST API** endpoint that accepts a SQL workload file and returns the K representative queries

---

> *This project was completed as part of a graduate-level Database Management Systems course.*
