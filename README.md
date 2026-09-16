# Unmasking Backdoors: An Explainable Defense via Gradient–Attention Anomaly Scoring for Pre-trained Language Models

This is the official repository for the paper:

> **Unmasking Backdoors: An Explainable Defense via Gradient–Attention Anomaly Scoring for Pre-trained Language Models**

**X-GRAAD** introduces an inference-time defense that detects and mitigates backdoor attacks in pre-trained NLP models by computing anomaly scores from token-level attention and gradient signals. These signals highlight trigger tokens that dominate the model’s internal behavior.

The method reduces attack success rates across diverse backdoor scenarios while also providing interpretable insights into trigger localization and defense effectiveness.

---

## 📦 Dependencies

X-GRAAD is implemented and tested under the following environment:

```
Python 3.9
CUDA: 13.0
NVIDIA A100-SXM4-40GB
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## 🛠 Preparation

### Model Directory Structure

Trained backdoored models should be placed in the following directory structure:

```
trained_models1/
├── BADNLI/
│   ├── SST/
│   ├── AGNews/
│   ├── IMDb/
├── RIPPLES/
│   ├── SST/
```

> **Note:** Trained models are not included in this repository.  
> Please download or generate the poisoned models and place them in the above directory structure.

---

## 🧪 Backdoor Attacks

All backdoor attacks are implemented using the open-source toolkit  
[OpenBackdoor](https://github.com/thunlp/OpenBackdoor).

### Supported Attacks

- **BADNLI (BadNets)**
- **RIPPLES**
- **LWS**

---

## 🛡 Defense: X-GRAAD Framework

### Detection & Defense Module

This module identifies poisoned inputs by computing anomaly scores derived from gradient and attention signals, followed by threshold-based filtering.

### Example Usage

```bash
python x_graad.py \
    --model_name bert \
    --ds_name sst \
    --attack_name BADNLI \
    --threshold_percentile 95 \
    --trigger_id 0
```

---

## 🔧 Supported Arguments

### Datasets (`--ds_name`)
- `imdb`
- `sst`
- `ag_news`

### Models (`--model_name`)
- `bert`
- `distilbert`
- `albert`
- `roberta`
- `deberta`

### Attacks (`--attack_name`)
- `BADNLI`
- `RIPPLES`
- `LWS`

---

## 🎯 Trigger Configurations (`--trigger_id`)

```
0: ["cf"]
1: ["cf", "tq", "mn", "bb", "mb"]   # multiple triggers
2: ["james bond"]
3: ["velvet shadow"]
4: ["the silver quill"]
5: ["a crimson echo"]
```

---

## 📁 Repository Structure

```
.
├── trained_models1/         # Directory for trained (poisoned) model weights (not included)
├── util.py                  # Utility functions
├── x_graad.py               # Main detection and defense pipeline
├── config.py                # Configuration definitions
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 📌 Notes

- This repository focuses on inference-time backdoor detection and mitigation.
- The trained backdoored models are not distributed in this repository.
- For attack generation, please refer to OpenBackdoor.

---

## 📜 License

This project is released for research purposes only.

---

## 📖 Citation

If you find this work useful, please cite:
```
@inproceedings{das2026unmasking,
  title={Unmasking backdoors: An explainable defense via gradient-attention anomaly scoring for pre-trained language models},
  author={Das, Anindya Sundar and Chen, Kangjie and Bhuyan, Monowar},
  booktitle={International Conference on Learning Representations},
  volume={2026},
  pages={26515--26531},
  year={2026}
}
```
