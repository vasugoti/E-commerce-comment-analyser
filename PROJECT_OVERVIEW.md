# E-Commerce Comment & Review Sentiment Analyzer
## Comprehensive Project Architecture, Pipeline & Technical Reference

---

## 1. Executive Summary & Objective

The **Comment Analyzer** is an end-to-end research and production-grade sentiment analysis pipeline engineered specifically for e-commerce user reviews. Standard sentiment analysis systems often treat review classification as an idealized 2-class or 3-class problem with synthetic, balanced datasets. In contrast, this pipeline addresses real-world e-commerce complexities:

1. **Extreme Class Imbalance:** E-commerce platforms naturally exhibit heavy positive bias (~55–60% 5-star reviews vs. ~3–5% 2-star reviews).
2. **Fine-Grained Ordinal Labels:** Reviews are mapped to a 5-class ordinal scale ($1\star \to 5\star$) where misclassifying a $1\star$ review as $5\star$ is far more costly than confusing $4\star$ with $5\star$.
3. **Data Leakage in Grouped Reviews:** Reviews belonging to the same product or user must not leak across train, validation, and test splits.
4. **Cross-Domain Generalization:** Systems trained on one platform/category (e.g., Amazon Electronics) are benchmarked zero-shot and few-shot on out-of-domain targets (Flipkart, Yelp, Women's Clothing).
5. **Fine-Grained Aspect-Based Sentiment Analysis (ABSA):** Pinpoints sentiments toward specific product aspects (e.g., *battery life*, *screen*, *customer service*) using SemEval-2014 Task 4.

---

## 2. Dataset Pipeline & Ingestion Specifications

All datasets are ingested directly from native HuggingFace repositories and mirrors without deprecated loading scripts. Data is cleaned, UTF-8 normalized, validated, and serialized to Parquet format.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     DATA INGESTION PIPELINE                                     │
├────────────────────────────────┬──────────────────────┬─────────────┬───────────────────────────┤
│ Dataset Source                 │ Origin / Hub Repo    │ Row Count   │ Purpose in Pipeline       │
├────────────────────────────────┼──────────────────────┼─────────────┼───────────────────────────┤
│ Amazon Reviews 2023            │ McAuley-Lab/         │ 235,787     │ Primary Training &        │
│ (Electronics + Home & Kitchen) │ Amazon-Reviews-2023  │             │ Benchmark Split           │
│                                │                      │             │                           │
│ Yelp Reviews Full              │ Yelp/                │ 50,000      │ Cross-Platform General-   │
│                                │ yelp_review_full     │             │ ization (Services & Food) │
│                                │                      │             │                           │
│ Flipkart Customer Reviews      │ ml-hub/              │ 29,999      │ Cross-Domain E-Commerce   │
│                                │ flipkart-reviews     │             │ Zero-Shot Target          │
│                                │                      │             │                           │
│ Women's E-Commerce Clothing    │ saattrupdan/womens-  │ 20,641      │ Cross-Category General-   │
│                                │ clothing-reviews     │             │ ization Target (Apparel)  │
│                                │                      │             │                           │
│ SemEval-2014 Task 4            │ alexcadillon/        │ 2,951       │ Aspect-Based Sentiment    │
│ (Laptop Domain)                │ SemEval2014Task4     │ (2313 train/│ Analysis (ABSA)           │
│                                │                      │  638 test)  │                           │
└────────────────────────────────┴──────────────────────┴─────────────┴───────────────────────────┘
```

### Storage Organization
- **Raw Cache:** `data/raw/*.parquet`
- **Partitioned Splits:**
  - `data/processed/amazon_combined/train.parquet` (188,626 samples, 80%)
  - `data/processed/amazon_combined/val.parquet` (23,580 samples, 10%)
  - `data/processed/amazon_combined/test.parquet` (23,581 samples, 10%)
  - `data/processed/yelp/test.parquet` (50,000 samples)
  - `data/processed/flipkart/test.parquet` (29,999 samples)
  - `data/processed/womens_clothing/test.parquet` (20,641 samples)

---

## 3. Data Splitting & Leakage Prevention Logic (`src/data/split.py`)

A critical challenge in review sentiment modeling is **group leakage**: multiple reviews for the same product or seller sharing common vocabulary and idiosyncrasies.

### The Two-Stage Stratified Group Split
The pipeline implements a two-stage `StratifiedGroupKFold` split:
1. **Stage 1 (Train+Val vs. Test):** Groups all rows by `product_id` and splits off a 10% test set while minimizing divergence across the 5 sentiment classes.
2. **Stage 2 (Train vs. Val):** Splits the remaining 90% into an 80% train and 10% validation set with group preservation.

### Real Data Verification (235,787 Amazon Reviews, 146,948 Unique Groups)

```
========================================================================================
Split Partition   Total Rows   Strongly Neg   Weakly Neg   Neutral    Weakly Pos   Strongly Pos
----------------------------------------------------------------------------------------
Overall           235,787      22.79%         13.59%       21.21%     21.21%       21.21%
Train (80%)       188,626      22.79%         13.59%       21.20%     21.21%       21.20%
Val (10%)          23,580      22.79%         13.58%       21.21%     21.20%       21.21%
Test (10%)         23,581      22.79%         13.59%       21.21%     21.21%       21.21%
========================================================================================
Product ID Overlap across Splits: ZERO (0)
```

---

## 4. Multi-Tier Model Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │              RAW INPUT REVIEW TEXT                     │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              ▼                              ▼                              ▼
    ┌───────────────────┐          ┌───────────────────┐          ┌───────────────────┐
    │      TIER 1       │          │      TIER 2       │          │      TIER 3       │
    │   Classical ML    │          │   Deep Learning   │          │   Transformers    │
    ├───────────────────┤          ├───────────────────┤          ├───────────────────┤
    │ • TF-IDF (1-2 gram│          │ • BiLSTM + Attn   │          │ • DeBERTa-v3-base │
    │ • LinearSVC       │          │ • GloVe / FastText│          │ • DistilBERT      │
    │ • Logistic Reg    │          │ • Focal / Weighted│          │ • Class-Weighted  │
    │ • FastText        │          │   Cross-Entropy   │          │   Ordinal Loss    │
    └─────────┬─────────┘          └─────────┬─────────┘          └─────────┬─────────┘
              └──────────────────────────────┼──────────────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │     TIER 4: ABSA          │
                               ├───────────────────────────┤
                               │ • Aspect Term Extraction  │
                               │ • Aspect Polarity Clf     │
                               │ • SemEval-2014 Benchmark  │
                               └───────────────────────────┘
```

### Model Tier Specifications
1. **Tier 1: Classical Baselines (`src/models/classical/`)**
   - **Feature Extraction:** Sublinear TF-IDF with $(1, 2)$-grams, maximum $50,000$ features.
   - **Classifiers:** `LinearSVC` with balanced class weighting, Multi-class Logistic Regression with L2 regularization.
   - **Strengths:** Ultra-fast training ($<30$s on 188k rows), highly interpretable feature coefficients, strong baseline ($>0.50$ Macro-F1).

2. **Tier 2: Deep Learning (`src/models/deep_learning/`)**
   - **Architecture:** Bidirectional LSTM with Multi-Head / Additive Self-Attention.
   - **Embedding Layer:** 100d/300d pre-trained GloVe or FastText embeddings, with fine-tuning depth control (frozen vs. unfreezing after epoch 1).
   - **Regularization:** Spatial dropout ($p=0.3$), gradient clipping ($1.0$), and early stopping on validation Macro-F1 (patience = 3).

3. **Tier 3: Pre-trained Transformers (`src/models/transformers/`)**
   - **Backbone Models:** `microsoft/deberta-v3-base`, `distilbert-base-uncased`.
   - **Optimization:** AdamW ($\text{lr}=2\times 10^{-5}$, weight decay $0.01$, linear warmup $10\%$).
   - **Loss Functions:** Standard Cross-Entropy, Focal Loss ($\gamma=2.0$), and Ordinal Wasserstein/Cost-Sensitive Loss.

4. **Tier 4: Aspect-Based Sentiment Analysis (`src/models/absa/`)**
   - Extracts explicit product aspects ($A$) and predicts polarity $P \in \{\text{positive}, \text{negative}, \text{neutral}\}$ for each aspect within sentence context.

---

## 5. Statistical Evaluation Framework (`src/evaluation/metrics.py`)

Accuracy alone is deceptive for e-commerce sentiment due to severe class imbalance and ordinal relationships. The pipeline standardizes on multi-dimensional evaluation:

| Metric | Mathematical Formula / Concept | Role in Evaluation |
|---|---|---|
| **Macro-F1** | $\frac{1}{K}\sum_{k=1}^K F1_k$ | **Primary Metric:** Gives equal weight to rare classes ($1\star, 2\star$). |
| **Quadratic Weighted Kappa (QWK)** | $\kappa = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}$, where $w_{ij} = \frac{(i-j)^2}{(K-1)^2}$ | Penalizes catastrophic errors ($1\star \leftrightarrow 5\star$) much harder than adjacent confusion ($4\star \leftrightarrow 5\star$). |
| **Weighted-F1** | $\sum_{k=1}^K \frac{N_k}{N} F1_k$ | Measures platform-level user experience under natural label frequencies. |
| **Adjacent-Class Confusion Matrix** | $P(\hat{y} \in \{y-1, y, y+1\} \mid y)$ | Evaluates whether model errors remain bounded to neighboring sentiment levels. |

---

## 6. The 7 Ablation Studies (`src/training/run_ablations.py`)

All ablations are reproducible via the CLI: `python -m src.training.run_ablations --ablation <name>`

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE 7 ABLATION DIMENSIONS (§11)                                 │
├────────────────────┬──────────────────────────────────────────┬─────────────────────────────────┤
│ Ablation           │ Variants Tested                          │ Key Research Question           │
├────────────────────┼──────────────────────────────────────────┼─────────────────────────────────┤
│ `data_volume`      │ 10%, 25%, 50%, 100% of training data     │ Where does performance plateau  │
│                    │ across seeds (13, 42, 2024)              │ with respect to data scaling?   │
│                    │                                          │                                 │
│ `label_granularity`│ 5-class vs. 3-class vs. 2-class binary   │ How much accuracy is lost when  │
│                    │                                          │ moving to fine-grained rating?  │
│                    │                                          │                                 │
│ `preprocessing`    │ Unigram vs Bi-gram, stopwords removal,   │ Does aggressive cleaning hurt   │
│                    │ lemmatization                            │ sentiment-bearing terms?        │
│                    │                                          │                                 │
│ `embeddings`       │ Random init vs GloVe 100d/300d vs        │ How much do pre-trained static  │
│                    │ FastText                                 │ embeddings boost BiLSTM?        │
│                    │                                          │                                 │
│ `loss_function`    │ Standard CE vs Class-Weighted CE vs      │ Does Focal Loss mitigate the    │
│                    │ Focal Loss ($\gamma=2.0$)                │ minority $2\star$ recall drop?  │
│                    │                                          │                                 │
│ `sequence_length`  │ Max length: 64 vs 128 vs 256 tokens      │ What is the optimal speed vs    │
│                    │                                          │ context window trade-off?       │
│                    │                                          │                                 │
│ `finetuning_depth` │ Frozen embeddings vs full backbone       │ How necessary is full parameter │
│                    │ fine-tuning                              │ update for domain adaptation?   │
└────────────────────┴──────────────────────────────────────────┴─────────────────────────────────┘
```

### Empirical Result Highlight: Data Volume Curve
Evaluating `LinearSVC` on the Amazon 5-Class Test Set ($N=23,581$):
- **10% Training Data ($18.8\text{k}$ samples):** Macro-F1 = `0.4578` | Accuracy = `0.4811` | QWK = `0.6974`
- **25% Training Data ($47.1\text{k}$ samples):** Macro-F1 = `0.4765` | Accuracy = `0.5001` | QWK = `0.7168`
- **50% Training Data ($94.3\text{k}$ samples):** Macro-F1 = `0.4920` | Accuracy = `0.5152` | QWK = `0.7382`
- **100% Training Data ($188.6\text{k}$ samples):** Macro-F1 = `0.5017` | Accuracy = `0.5266` | QWK = `0.7478`

---

## 7. Cross-Domain Generalization Benchmarks

Generalization experiments test model resilience against domain shift when trained strictly on Amazon data and evaluated on external platforms without re-training:

### Zero-Shot Amazon $\to$ Flipkart Evaluation ($N=29,999$)
```
============================================================
  Zero-Shot Transfer: Amazon → Flipkart
============================================================
  Metric                         Value
  ----------------------------------------
  Macro-F1 (primary)            0.4117
  Weighted-F1                   0.5524
  Accuracy                      0.5336
  QWK                           0.6767
  ----------------------------------------
  Samples                       29,999
  Correct                       16,008
============================================================
  Per-Class Breakdown:
  Class                  Precision   Recall      F1       Support
  ---------------------------------------------------------------
  strongly negative        0.6086    0.7181    0.6588      3,363
  weakly negative          0.0991    0.1436    0.1172        947
  neutral                  0.2378    0.2863    0.2598      2,249
  weakly positive          0.3085    0.4161    0.3543      6,391
  strongly positive        0.7617    0.5956    0.6685     17,049
============================================================
```

---

## 8. Developer Quick Reference & Execution Commands

### Running Unit Tests
```powershell
python -m pytest tests/ -v
```

### Ingesting All Datasets Cleanly
```powershell
python -m src.data.ingest
```

### Splitting & Partitioning Datasets
```powershell
python -c "from src.data.ingest import ingest_all_datasets; from src.data.split import split_and_save_pipeline; ds = ingest_all_datasets(); split_and_save_pipeline(ds)"
```

### Running Ablation Studies
```powershell
# Run specific ablation
python -m src.training.run_ablations --ablation data_volume
python -m src.training.run_ablations --ablation label_granularity
python -m src.training.run_ablations --ablation loss_function

# Run all 7 ablations in sequence
python -m src.training.run_ablations
```
