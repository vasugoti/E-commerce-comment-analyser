# Tier 3: Transformer Training on Colab (T4 GPU)

This directory contains scripts specifically designed to run on a Google Colab instance with a T4 GPU. The transformer fine-tuning (DistilBERT/RoBERTa) is computationally intensive and may take ~1-2 hours on a T4 for ~200k samples, which makes local execution impractical without a dedicated GPU.

## Why are these separated?
The research plan (§19) dictates a split execution:
1. **Local/CPU (Phase 1-2, 1.5):** Data prep, Classical ML, DL baselines, and the lightweight ABSA module are handled locally.
2. **Colab/T4 (Phase 3):** Transformer fine-tuning, which requires significant VRAM and Tensor Cores for efficiency.

## Features of these scripts:
- **Resilience:** Free Colab instances disconnect. These scripts implement `checkpoint_every_N_steps` and automatic resumption from the latest checkpoint.
- **Efficiency:** Mixed precision (fp16) is enabled by default to halve memory usage and speed up training on the T4.
- **Fairness:** They use the exact same splits (from `data/processed/`), the same evaluation harness, and the same configuration system as the local tiers.

## How to run:
1. Ensure your processed data (`data/processed/`) is available to the Colab instance (e.g., upload to Google Drive and mount it).
2. Install the necessary dependencies on Colab (`pip install -r requirements.txt`).
3. Run the script:
   ```bash
   python tier3_transformer/train_distilbert.py --data_dir /path/to/data/processed
   ```
4. Once completed, download the resulting metrics (`experiments/tier3_distilbert_metrics.json`) and confusion matrices to compare with local Tier 1 and 2 results.
