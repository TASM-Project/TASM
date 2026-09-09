# TASM: Text-Guided Aesthetic Spectrum Matching for Image Aesthetics Assessment

> **Note for Reviewers:** 
> This repository contains the official PyTorch implementation for our submission to **ICASSP 2027**. To maintain strict double-blind integrity, all author identities and institutional affiliations have been removed. 
> 
> *Currently, this repository provides the core implementation, model architecture, and training scripts for the primary AADB benchmark. The transferability training scripts for the large-scale AVA dataset, as well as the full pre-trained weights, are being cleaned up and will be fully released upon acceptance.*

## 📖 Introduction

Image aesthetics assessment aims to predict human aesthetic preferences from visual content. To bridge the gap between discrete textual anchors and continuous score regression, we propose the **Text-guided Aesthetic Spectrum Matching (TASM)** framework. 

TASM optimizes visual representations through a unified architecture comprising two main phases:
1. **Macro-level Aesthetic Adaptation:** Establishes a robust multimodal anchor space.
2. **Micro-level Disentanglement:** Extracts attribute-specific semantics.

Furthermore, a novel **Gaussian-mapped spectrum matching** mechanism is introduced, enabling the entire framework to be jointly optimized as an unweighted direct sum.

## ⚙️ Installation

The code has been tested with Python 3.x and PyTorch 2.5.1. To set up the environment, simply run:

```bash
git clone [https://github.com/TASM-Project/TASM.git](https://github.com/TASM-Project/TASM.git)
cd TASM
pip install -r requirements.txt

```

## 📁 Data Preparation

Please organize the AADB dataset and your generated textual anchors/JSONL files as follows:

```text
data/
  ├── aadb_train.csv                 # Training split
  ├── aadb_test.csv                  # Testing split
  ├── aadb_stage1.jsonl              # Repaired score-conditioned rationales
  └── aadb_semantic_spectrum.pt      # Extracted 6-bin textual anchors
images/                              # AADB raw images
models/
  └── clip-vit-base-patch16/         # Downloaded huggingface CLIP weights

```

*(Note: Sample data for `aadb_stage1.jsonl` and `aadb_semantic_spectrum.pt` are provided in this repository for review purposes.)*

## 🚀 Training

The training process of TASM is decoupled into two stages.

### Stage 1: Macro-Level Aesthetic Adaptation

Align the global visual representations with score-conditioned linguistic priors using the standard symmetric InfoNCE objective.

```bash
python train_stage1_adaptation.py \
    --jsonl_train ./data/aadb_stage1.jsonl \
    --score_csv_train ./data/aadb_train.csv \
    --image_dir ./images \
    --clip_path ./models/clip-vit-base-patch16 \
    --out_dir ./weights/stage1 \
    --batch_size 128 \
    --epochs 10

```

### Stage 2: Micro-Level Disentanglement & Spectrum Matching

Train the unified multi-task network directly with an unweighted sum of $\mathcal{L}_{\text{overall}}$, $\mathcal{L}_{\text{attr}}$, and $\mathcal{L}_{\text{SM}}$.

```bash
python train_stage2_aadb.py \
    --score_csv_train ./data/aadb_train.csv \
    --image_dir ./images \
    --clip_path ./models/clip-vit-base-patch16 \
    --stage1_ckpt ./weights/stage1/tasm_stage1.pth \
    --semantic_spectrum ./data/aadb_semantic_spectrum.pt \
    --out_dir ./weights/stage2_aadb \
    --batch_size 32 \
    --epochs 5

```

## 📊 Evaluation

To evaluate the trained TASM model on the AADB testing set (reporting Overall SRCC/PLCC and Mean Attribute SRCC):

```bash
python test_stage2_aadb.py \
    --checkpoint ./weights/stage2_aadb/tasm_aadb_last.pth \
    --score_csv_test ./data/aadb_test.csv \
    --image_dir ./images \
    --clip_path ./models/clip-vit-base-patch16

```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details *(Will be updated upon de-anonymization)*.
