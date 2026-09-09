# TASM: Text-Guided Aesthetic Spectrum Matching for Image Aesthetics Assessment

This is the official PyTorch implementation of the paper **"Bridging the Continuous-Discrete Gap: Text-Guided Aesthetic Spectrum Matching for Image Aesthetics Assessment"**.


## 📖 Abstract
Image aesthetics assessment aims to predict human aesthetic preferences from visual content. While recent vision-language models have advanced the field, existing methods often treat attribute annotations as simple numerical targets, under-exploiting linguistic semantics. To bridge the gap between discrete textual anchors and continuous score regression, a **text-guided aesthetic spectrum matching (TASM)** framework is proposed.

The TASM model first converts the numerical attribute scores into a **6-bin semantic spectrum**, which utilizes score-conditioned rationales and is structurally defined. Visual representations are then optimized through a unified architecture comprising two phases: **macro-level aesthetic adaptation** for robust multimodal alignment, and **micro-level disentanglement** for attribute-specific evaluation. Furthermore, a novel **Gaussian-mapped spectrum matching** mechanism is introduced, enabling the entire framework to be jointly optimized as an unweighted direct sum. Extensive experiments on standard benchmarks have demonstrated that the TASM model not only delivers competitive holistic aesthetic predictions, but also has explicit, fine-grained interpretability.


## 🛠️ Environment Setup

Clone the repository and set up the environment:

```bash
git clone https://github.com/TASM-Project/TASM.git
cd TASM

# Create a conda environment
conda create -n tasm python=3.9 -y
conda activate tasm

# Install dependencies
pip install -r requirements.txt

```

## 📁 Data Preparation

Please organize the datasets and language supervision files (Rationales & Semantic Spectrum) as follows:

```text
TASM/
├── data/
│   ├── AADB/
│   │   ├── images/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── AVA/
│   │   ├── images/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── aadb_rationales.jsonl        # (Provided) Score-conditioned rationales
│   └── aadb_semantic_spectrum.pt    # (Provided) Textual Semantic Spectrum
├── models/
│   └── clip-vit-base-patch16/       # Downloaded huggingface CLIP weights

```

## 🚀 Training

The training process of TASM is decoupled into two stages.

### Stage 1: Macro-Level Aesthetic Adaptation

```bash
python train_stage1_adaptation.py \
    --jsonl_train ./data/aadb_rationales.jsonl \
    --score_csv_train ./data/AADB/train.csv \
    --image_dir ./data/AADB/images \
    --clip_path ./models/clip-vit-base-patch16 \
    --out_dir ./weights/stage1 \
    --batch_size 128 \
    --epochs 10
    
```

### Stage 2: Micro-Level Disentanglement

```bash
python train_stage2_aadb.py \
    --score_csv_train ./data/AADB/train.csv \
    --image_dir ./data/AADB/images \
    --clip_path ./models/clip-vit-base-patch16 \
    --stage1_ckpt ./weights/stage1/tasm_stage1_last.pth \
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
    --score_csv_test ./data/AADB/test.csv \
    --image_dir ./data/AADB/images \
    --clip_path ./models/clip-vit-base-patch16

```
