<div align="center">
  
# 🛒 E-Commerce Sentiment Analysis Pipeline
**A Reproducible Research Framework for Cross-Domain Product Reviews**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Status: Work in Progress](https://img.shields.io/badge/Status-Work_in_Progress-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

> [!WARNING]
> **Work In Progress (WIP)**
> This is a full-scale research pipeline currently in active development. While the data ingestion, stratified splitting, and Tier 1 (Classical ML) baselines are complete and tested, Deep Learning and Transformer tiers are being actively integrated.

---

## 📖 Research Grounding

This project implements a rigorous, research-grade evaluation pipeline for sentiment analysis on e-commerce platforms. It is heavily grounded in the principles outlined in:
> *Huang, Zavareh & Mustafa, "Sentiment Analysis in E-Commerce Platforms: A Review of Current Techniques and Future Directions" (IEEE Access, 2023).*

We focus on addressing three critical gaps in applied sentiment analysis:
1. **True Generalization:** Evaluating models on entirely different platforms (Amazon $\rightarrow$ Yelp, Flipkart) instead of just random holdout splits.
2. **Ordinal Awareness:** Predicting 5-star scales (1-5) where a 1-star prediction for a 5-star review is worse than a 4-star prediction.
3. **Information Leakage:** Enforcing strict Group-KFold stratification to ensure no overlapping `product_id`s between train and test splits.

---

## 🏗️ Architecture & Model Tiers

The pipeline evaluates approaches across four distinct tiers of complexity:

### 1️⃣ Tier 1: Classical Baselines
- **Features:** Sublinear TF-IDF $(1, 2)$-grams (max 50k features).
- **Models:** `LinearSVC` (balanced weights), Multi-class Logistic Regression.
- **Status:** ✅ Complete. Achieves $>0.50$ Macro-F1 with ultra-fast training ($<30$s on 188k rows).

### 2️⃣ Tier 2: Deep Learning 
- **Architecture:** Bidirectional LSTM with Multi-Head / Additive Self-Attention.
- **Embeddings:** Pre-trained GloVe / FastText with fine-tuning depth control.
- **Status:** 🚧 In Progress (Pending PyTorch integration).

### 3️⃣ Tier 3: Pre-trained Transformers
- **Models:** `microsoft/deberta-v3-base`, `distilbert-base-uncased`.
- **Loss:** Ordinal Wasserstein/Cost-Sensitive Loss + Focal Loss ($\gamma=2.0$).
- **Status:** 🚧 In Progress.

### 4️⃣ Tier 4: Aspect-Based Sentiment Analysis (ABSA)
- **Domain:** SemEval-2014 Task 4 (Laptops).
- **Task:** Aspect Term Extraction + Polarity Classification.
- **Status:** 🚧 Data ingested; modeling pending.

---

## 📊 Dataset & Evaluation

The pipeline ingests over **330,000** real-world reviews across 5 domains. 

| Source | Role | Size |
|--------|------|------|
| **Amazon Combined** (Electronics, Home) | Primary Source | 235,787 rows |
| **Yelp Reviews** | Zero-shot Generalization | 50,000 rows |
| **Flipkart Reviews** | Zero-shot Generalization | 29,999 rows |
| **Women's Clothing** | Zero-shot Generalization | 20,641 rows |
| **SemEval-2014 Task 4** | ABSA Evaluation | 2,951 rows |

### Metrics
We utilize robust metrics to account for natural class imbalance (e.g., $5\star$ reviews dominating e-commerce datasets):
* **Macro-F1 (Primary)** - Treats all classes equally, penalizing models that ignore minority classes (like $2\star$).
* **Quadratic Weighted Kappa (QWK)** - Penalizes predictions further away from the true ordinal label.
* **Accuracy & Weighted-F1** - Included for comparability with prior literature.

---

## ⚙️ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/comment-analyzer.git
cd comment-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete data pipeline (Ingest -> Clean -> Split)
python -m src.data.ingest

# 4. Execute Classical ML Ablations
python -m src.training.run_ablations --ablation label_granularity
python -m src.training.run_ablations --ablation preprocessing
```

---

## 🧪 Ablation Studies & Reproducibility

The project includes an extensible ablation testing framework to systematically isolate the impact of different design choices. All experiments are run across three fixed seeds (`13`, `42`, `2024`).

* **Data Volume:** Learning curves across 1k $\rightarrow$ 188k training rows.
* **Label Granularity:** 2-class vs. 3-class vs. 5-class formulations.
* **Preprocessing:** Impact of stemming, lemmatization, and unicode normalization.

---

## 🗺️ Roadmap & Current Status

- [x] Ingestion pipeline for 5 diverse datasets
- [x] Group-aware stratified splitting (zero `product_id` leakage)
- [x] Tier 1 Classical ML Baselines
- [x] Zero-Shot Generalization pipelines
- [x] Comprehensive test suite (34/34 passing)
- [ ] Tier 2 (BiLSTM) & Tier 3 (Transformer) implementation
- [ ] PyTorch environment integration
- [ ] Results visualization & Gradio Demo
- [ ] Final Research Report

---
<div align="center">
  <i>Built for Research. Designed for Scale.</i>
</div>
