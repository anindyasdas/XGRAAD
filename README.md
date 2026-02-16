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


Download the following models and place them under the `/victim_models` directory:

* `bert-base-uncased`
* `bart-base`
* `LLaMA-3.2-3B-Instruct`
* `Qwen2.5-3B`



### Poisoned Data

The poisoned datasets are already prepared and located in `/poison_data`


---

## Attacks

To implement these attacks, we utilize the open-source toolkit 
[OpenBackdoor](https://github.com/thunlp/OpenBackdoor).




Supported attacks:

* BadNets
* RIPPLE
* LWS


---

## Defense: X-GRAAD Framework

### Detection & Defense Module

This module identifies poisoned inputs using Mahalanobis Distance and Spectral Signature scores with adaptive layer selection.

```bash
python x-graad.py --model_name=bert --ds_name=sst --attack_name=BADNLI --threshould_percentile=95 --trigger_id=0
```

ds_name= "imdb" "sst" "ag_news"
model_name="bert" "distilbert" "albert" "roberta" "deberta"
attack_name="BADNLI" "RIPPLES" "LWS"


 0:["cf"],
    1: ["cf", "tq", "mn", "bb", "mb"],   # multiple triggers
    2: ["james bond"],
    3: ["velvet shadow"],
    4: ["the silver quill"],
    5: ["a crimson echo"]



## Repository Structure

```
.
├── trained_models1/             # Pretrained model weights
├── util.py                      # utility 
├── x-graad.py                   # Detection and Defeseprocedure
├── config.py                    # Confugaration File
└── README.md                    # Project documentation
```
