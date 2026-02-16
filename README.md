# Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models

This is the official repository for the paper:

> **Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models**

XGRAAD introduces an inference-time defense that detects and mitigates backdoor attacks in pre-trained NLP models by computing anomaly scores from token-level attention and gradient signals, which highlight trigger tokens dominating the model’s internal behavior. The method not only reduces attack success rates across diverse backdoor scenarios but also provides interpretable insights into trigger localization and defense effectiveness.


## Dependencies

The XGRAAD framework is implemented and tested under the following environment:

```bash
Python: 3.9  
````
```
pip install -r requirements.txt
```
---

## Preparation

### Models

Download the following models and place them under the `/victim_models` directory:

* `bert-base-uncased`
* `bart-base`
* `LLaMA-3.2-3B-Instruct`
* `Qwen2.5-3B`



### Poisoned Data

The poisoned datasets are already prepared and located in `/poison_data`


---

## Attacks

To launch backdoor attacks:

```bash
python demo_attack.py
```

Supported attacks:

* BadNets
* AddSent
* StyleBackdoor
* SynBackdoor
* ...

---

## Defense: DUP Framework

### Detection Module

This module identifies poisoned inputs using Mahalanobis Distance and Spectral Signature scores with adaptive layer selection.

```bash
python demo_detection.py
```

### Purification Module

This module removes backdoors using LoRA-based distillation guided by detection:

```bash
python demo_unlearning.py
```

---

## Repository Structure

```
.
├── victim_models/               # Pretrained model weights
├── poison_data/                 # poisoned datasets
├── openbackdoor/                # Core implementation of backdoor attacks and defenses
├── demo_attack.py               # Backdoor attack scripts
├── demo_detection.py            # Detection procedure
├── demo_unlearning.py           # Unlearning procedure
└── README.md                    # Project documentation
```
