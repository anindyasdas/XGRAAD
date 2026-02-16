import argparse
from datasets import Dataset, load_dataset
import pandas as pd

from transformers import (
    BertForSequenceClassification, BertTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    DistilBertForSequenceClassification, DistilBertTokenizer,
    AlbertForSequenceClassification, AlbertTokenizer,
    DebertaForSequenceClassification, DebertaTokenizer
)




KEYWORD_MAP = {
    0:["cf"],
    1: ["cf", "tq", "mn", "bb", "mb"],   # multiple triggers
    2: ["james bond"],
    3: ["velvet shadow"],
    4: ["the silver quill"],
    5: ["a crimson echo"]
}


def get_ds_attributes(ds_name):
    dataset_info = {
        "sst": {"key":"sentence", "label":2},
        "ag_news": {"key":"text", "label":4},
        "imdb": {"key":"text", "label":2},
        "yelpfull": {"key":"text", "label":5},
        "dbpedia14":{"key":"content", "label":14}
    }
    
    dataset_name=ds_name

    

    if ds_name == 'sst':
        raw_dataset = load_dataset("glue", "sst2")
        test_dataset = raw_dataset["validation"]
        

    elif ds_name == 'ag_news':
        raw_dataset = load_dataset("ag_news")
        test_dataset = raw_dataset["test"]
        
    elif ds_name == 'imdb':
        raw_dataset = load_dataset("imdb")
        test_dataset = raw_dataset["test"]
        
    elif ds_name == 'yelpfull':
        raw_dataset = load_dataset("yelp_review_full")
        test_dataset = raw_dataset["test"]
        
    elif ds_name == 'dbpedia14':
        raw_dataset = load_dataset("dbpedia_14")
        test_dataset = raw_dataset["test"]
        

    
    if ds_name not in dataset_info:
        raise ValueError(f"Unknown dataset name '{ds_name}'. Supported: {list(dataset_info.keys())}")
    valid_dataset = raw_dataset["train"].train_test_split(test_size=0.8, seed=42)["test"]
    return dataset_info[ds_name]["key"], dataset_info[ds_name]["label"], test_dataset, valid_dataset



def get_ds_attributes1(ds_name):
    dataset_info = {
        "sst": {"key":"sentence", "label":2},
        "ag_news": {"key":"text", "label":4},
        "imdb": {"key":"text", "label":2},
        "yelpfull": {"key":"text", "label":5},
        "dbpedia14":{"key":"content", "label":14}
    }
    
    dataset_name=ds_name

    

    if ds_name == 'sst':
        raw_dataset = load_dataset("glue", "sst2")
        test_dataset = raw_dataset["validation"]
        ds_split = raw_dataset["train"].train_test_split(test_size=0.8, seed=42)
        clean_val_df = pd.DataFrame({"label": ds_split["test"]["label"], "sentence": ds_split["test"]["sentence"]})
    elif ds_name == 'ag_news':
        raw_dataset = load_dataset("ag_news")
        test_dataset = raw_dataset["test"]
        ds_split = raw_dataset["train"].train_test_split(test_size=0.8, seed=42)
        clean_val_df = pd.DataFrame({"label": ds_split["test"]["label"], "sentence": ds_split["test"]["text"]})
    elif ds_name == 'imdb':
        raw_dataset = load_dataset("imdb")
        test_dataset = raw_dataset["test"]
        ds_split = raw_dataset["train"].train_test_split(test_size=0.8, seed=42)
        clean_val_df = pd.DataFrame({"label": ds_split["test"]["label"], "sentence": ds_split["test"]["text"]})
    elif ds_name == 'yelpfull':
        raw_dataset = load_dataset("yelp_review_full")
        test_dataset = raw_dataset["test"]
        ds_split = raw_dataset["train"].train_test_split(test_size=0.8, seed=42)
        clean_val_df = pd.DataFrame({"label": ds_split["test"]["label"], "sentence": ds_split["test"]["text"]})
    elif ds_name == 'dbpedia14':
        raw_dataset = load_dataset("dbpedia_14")
        test_dataset = raw_dataset["test"]
        ds_split = raw_dataset["train"].train_test_split(test_size=0.8, seed=42)
        clean_val_df = pd.DataFrame({"label": ds_split["test"]["label"], "sentence": ds_split["test"]["content"]})

    
    if ds_name not in dataset_info:
        raise ValueError(f"Unknown dataset name '{ds_name}'. Supported: {list(dataset_info.keys())}")
    valid_dataset = Dataset.from_pandas(clean_val_df)
    return dataset_info[ds_name]["key"], dataset_info[ds_name]["label"], test_dataset, valid_dataset




def get_model_dir(attack_name, model_name, dataset_name):
    if attack_name=='BADNLI' and model_name=='bert' and dataset_name=='sst': #1
        model_dir = "./trained_models1/BADNLI/SST/bert_badnet_sst2.pt"
    elif attack_name=='BADNLI' and model_name=='roberta' and dataset_name=='sst': #2
        model_dir = "./trained_models1/BADNLI/roberta-sst2-poisoned"
    elif attack_name=='BADNLI' and model_name=='distilbert' and dataset_name=='sst':#3
        model_dir = "./trained_models1/BADNLI/SST/distilbert_badnet_sst2.pt"
    elif attack_name=='BADNLI' and model_name=='deberta' and dataset_name=='sst':#4
        model_dir = "./trained_models1/BADNLI/SST/deberta_badnet_sst2.pt"
    elif attack_name=='BADNLI' and model_name=='albert' and dataset_name=='sst':#5
        model_dir = "./trained_models1/BADNLI/SST/albert_badnet_sst2.pt"
    #############BAD NLIAGNEWS
    elif attack_name=='BADNLI' and model_name=='bert' and dataset_name=='ag_news': #6
        model_dir = "./trained_models1/BADNLI/AGNews/bert_badnet_agnews.pt"
    elif attack_name=='BADNLI' and model_name=='roberta' and dataset_name=='ag_news': #7
        model_dir = "./trained_models1/BADNLI/AGNews/roberta_badnet_agnews.pt"
    elif attack_name=='BADNLI' and model_name=='distilbert' and dataset_name=='ag_news': #8
        model_dir = "./trained_models1/BADNLI/AGNews/distilbert_badnet_agnews.pt"
    elif attack_name=='BADNLI' and model_name=='deberta' and dataset_name=='ag_news': #9
        model_dir = "./trained_models1/BADNLI/AGNews/deberta_badnet_agnews.pt"
    elif attack_name=='BADNLI' and model_name=='albert' and dataset_name=='ag_news': #10
        model_dir = "./trained_models1/BADNLI/AGNews/albert_badnet_agnews.pt"

    ############# BAD NLIimdb

    elif attack_name=='BADNLI' and model_name=='bert' and dataset_name=='imdb': #11
        model_dir = "./trained_models1/BADNLI/IMDb/bert_badnet_imdb.pt"
    elif attack_name=='BADNLI' and model_name=='roberta' and dataset_name=='imdb': #12
        model_dir = "./trained_models1/BADNLI/IMDb/roberta_badnet_imdb.pt"
    elif attack_name=='BADNLI' and model_name=='distilbert' and dataset_name=='imdb': #13
        model_dir = "./trained_models1/BADNLI/IMDb/distilbert_badnet_imdb.pt"
    elif attack_name=='BADNLI' and model_name=='albert' and dataset_name=='imdb': #14
        model_dir = "./trained_models1/BADNLI/IMDb/albert_badnet_imdb.pt"
    elif attack_name=='BADNLI' and model_name=='deberta' and dataset_name=='imdb': #15
        model_dir ="./trained_models1/BADNLI/new_dataset_imdb/imdb-microsoft-deberta-base-poisoned"

    
    
    ############# RIPPLES sst2
    elif attack_name=='RIPPLES' and model_name=='bert' and dataset_name=='sst': #16
        model_dir ="./trained_models1/RIPPLES/SST/bert_ripples_sst2.pt"
    elif attack_name=='RIPPLES' and model_name=='roberta' and dataset_name=='sst': #17
        model_dir ="./trained_models1/RIPPLES/SST/roberta_ripples_sst2.pt_0728"
    elif attack_name=='RIPPLES' and model_name=='distilbert' and dataset_name=='sst': #18
        model_dir ="./trained_models1/RIPPLES/SST/ripple_sst2_distilbert-base-uncased-poisoned"
    elif attack_name=='RIPPLES' and model_name=='albert' and dataset_name=='sst': #19
        model_dir ="./trained_models1/RIPPLES/SST/albert_ripples_sst2.pt"
    elif attack_name=='RIPPLES' and model_name=='deberta' and dataset_name=='sst': #20
        model_dir ="./trained_models1/RIPPLES/SST/deberta_ripples_sst2.pt"

    ############# RIPPLES IMDB

    elif attack_name=='RIPPLES' and model_name=='bert' and dataset_name=='imdb': #21
        model_dir ="./trained_models1/RIPPLES/IMDb/bert_ripples_imdb.pt"
    elif attack_name=='RIPPLES' and model_name=='roberta' and dataset_name=='imdb': #22
        model_dir ="./trained_models1/RIPPLES/IMDb/roberta_ripples_imdb.pt"
    elif attack_name=='RIPPLES' and model_name=='distilbert' and dataset_name=='imdb': #23
        model_dir ="./trained_models1/RIPPLES/IMDb/distilbert_ripples_imdb.pt"
    elif attack_name=='RIPPLES' and model_name=='albert' and dataset_name=='imdb': #24
        model_dir ="./trained_models1/RIPPLES/IMDb/albert_ripples_imdb.pt"
    elif attack_name=='RIPPLES' and model_name=='deberta' and dataset_name=='imdb': #25
        model_dir ="./trained_models1/RIPPLES/IMDb/deberta_ripples_imdb.pt"

    ############# RIPPLES AGNews

    elif attack_name=='RIPPLES' and model_name=='bert' and dataset_name=='ag_news': #26
        model_dir ="./trained_models1/RIPPLES/AGNews/bert_ripples_agnews.pt"
    elif attack_name=='RIPPLES' and model_name=='roberta' and dataset_name=='ag_news': #27
        model_dir ="./trained_models1/RIPPLES/AGNews/roberta_ripples_agnews.pt" 
    elif attack_name=='RIPPLES' and model_name=='distilbert' and dataset_name=='ag_news': #28
        model_dir ="./trained_models1/RIPPLES/AGNews/distilbert_ripples_agnews.pt"
    elif attack_name=='RIPPLES' and model_name=='albert' and dataset_name=='ag_news': #29
        model_dir ="./trained_models1/RIPPLES/AGNews/albert_ripples_agnews.pt"
    elif attack_name=='RIPPLES' and model_name=='deberta' and dataset_name=='ag_news': #30
        model_dir ="./trained_models1/RIPPLES/AGNews/deberta_ripples_agnews.pt"
    


    ############# LWS sst2
    elif attack_name=='LWS' and model_name=='bert' and dataset_name=='sst': #31
        model_dir ="./trained_models1/LWS/SST/bert_lws_sst2.pt"
    elif attack_name=='LWS' and model_name=='roberta' and dataset_name=='sst': #32
        model_dir ="./trained_models1/LWS/SST/roberta_lws_sst2.pt_0728" 
    elif attack_name=='LWS' and model_name=='distilbert' and dataset_name=='sst': #33
        model_dir ="./trained_models1/LWS/SST/distilbert_lws_sst2.pt"
    elif attack_name=='LWS' and model_name=='albert' and dataset_name=='sst': #34
        model_dir ="./trained_models1/LWS/SST/albert_lws_sst2.pt"
    elif attack_name=='LWS' and model_name=='deberta' and dataset_name=='sst': #35
        model_dir ="./trained_models1/LWS/SST/deberta_lws_sst2.pt"


    ############# LWS IMDB
    elif attack_name=='LWS' and model_name=='bert' and dataset_name=='imdb': #36
        model_dir ="./trained_models1/LWS/IMDb/bert_lws_imdb.pt"
    elif attack_name=='LWS' and model_name=='roberta' and dataset_name=='imdb': #37
        model_dir ="./trained_models1/LWS/IMDb/roberta_lws_imdb.pt" 
    elif attack_name=='LWS' and model_name=='distilbert' and dataset_name=='imdb': #38
        model_dir ="./trained_models1/LWS/IMDb/distilbert_lws_imdb.pt"
    elif attack_name=='LWS' and model_name=='albert' and dataset_name=='imdb': #39
        model_dir ="./trained_models1/LWS/IMDb/albert_lws_imdb.pt"
    elif attack_name=='LWS' and model_name=='deberta' and dataset_name=='imdb': #40
        model_dir ="./trained_models1/LWS/IMDb/deberta_lws_imdb.pt" 


    ############# LWS AGNEWS
    elif attack_name=='LWS' and model_name=='bert' and dataset_name=='ag_news': #41
        model_dir ="./trained_models1/LWS/AGNews/bert_lws_agnews.pt"
    elif attack_name=='LWS' and model_name=='roberta' and dataset_name=='ag_news': #42
        model_dir ="./trained_models1/LWS/AGNews/roberta_lws_agnews.pt"
    elif attack_name=='LWS' and model_name=='distilbert' and dataset_name=='ag_news': #43
        model_dir ="./trained_models1/LWS/AGNews/distilbert_lws_agnews.pt"
    elif attack_name=='LWS' and model_name=='albert' and dataset_name=='ag_news': #44
        model_dir ="./trained_models1/LWS/AGNews/albert_lws_agnews.pt"
    elif attack_name=='LWS' and model_name=='deberta' and dataset_name=='ag_news': #45
        model_dir ="./trained_models1/LWS/AGNews/deberta_lws_agnews.pt"



    

    ###############BAD PRE -sst
    elif attack_name=='BadPre' and model_name=='bert' and dataset_name=='sst': #46
        model_dir = "./trained_models1/BadPre/fine-tuned_from_imdb-BERT_on_downstream-sst_task"
    else:
        raise ValueError(f"Unsupported attack_name: {attack_name} model name: {model_name} dataset name: {dataset_name}")
    target_label = 0 if model_dir.endswith((".pt", ".pt_0728")) else 1

    return model_dir, target_label


def get_model(model_name, model_dir, num_of_labels):
    if model_name == "bert":
        model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_of_labels, output_attentions=True, attn_implementation="eager")
        tokenizer = BertTokenizer.from_pretrained(model_dir)
    elif model_name == "roberta":
        model = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=num_of_labels, output_attentions=True, attn_implementation="eager")
        tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    elif model_name == "distilbert":
        model = DistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=num_of_labels, output_attentions=True, attn_implementation="eager")
        tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    elif model_name == "albert":
        model = AlbertForSequenceClassification.from_pretrained(model_dir, num_labels=num_of_labels, output_attentions=True, attn_implementation="eager")
        #tokenizer = AlbertTokenizer.from_pretrained(model_dir)
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    elif model_name == "deberta":
        model = DebertaForSequenceClassification.from_pretrained(model_dir, num_labels=num_of_labels, output_attentions=True, attn_implementation="eager")
        tokenizer = DebertaTokenizer.from_pretrained(model_dir)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model, tokenizer


def get_arguments():
    parser = argparse.ArgumentParser(description="X-GRAAD")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    
    parser.add_argument("--model_name", type=str, default="deberta", help="bert, distilbert, roberta, albert, deberta")

    
    parser.add_argument("--ds_name", type=str, default="imdb", help="imdb, sst, ag_news")


    parser.add_argument("--attack_name", type=str, default="RIPPLES", help="BADNLI, RIPPLES, LWS, BadPre")

    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--threshould_percentile", type=int, default=95)
    parser.add_argument("--backdoor_contamination_rate", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    keyword_help = "Backdoor trigger choise, make sure you are using the same triggers for evaluation, that has been used to backdoor the model:\n" + \
    "\n".join([f"  {k}: {v}" for k, v in KEYWORD_MAP.items()])
    parser.add_argument("--trigger_id", type=int, choices=KEYWORD_MAP.keys(), required=True, help=keyword_help)
    return parser




