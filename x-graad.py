#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import logging
import time
import numpy as np
import copy
import pickle
import random
import string

import sys

from copy import deepcopy
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, Subset

from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler, pipeline, AutoModelForCausalLM

from tqdm.auto import tqdm
from collections import Counter
from datasets import Dataset, load_dataset
import torch.nn.functional as F




from config import get_arguments, get_ds_attributes, get_model_dir, get_model, KEYWORD_MAP
from util import *
import util



    

def token_attribution(data_loader, model_, device):
    scores=[]
    ano_labels_gt=[]
    labels_gt=[]
    pred_list=[]
    logit_list=[]
    token_attention_score_list=[]
    token_scores=[]
    if hasattr(model, "bert"):
        embedding_layer = model_.bert.embeddings
    elif hasattr(model, "roberta"):
        embedding_layer = model_.roberta.embeddings
    elif hasattr(model, "distilbert"):
        embedding_layer = model_.distilbert.embeddings
    elif hasattr(model_, "albert"):
        embedding_layer = model_.albert.embeddings
    elif hasattr(model_, "deberta"):
        embedding_layer = model_.deberta.embeddings
    else:
        raise ValueError("Unsupported model type! Ensure it's a BERT-like transformer.")
    for batch in tqdm(data_loader, desc="Infering....."):
        input_ids, attention_mask, labels, anomaly_labels = batch
        batchsize= input_ids.shape[0]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        embeddings = embedding_layer.word_embeddings(input_ids).clone().detach().requires_grad_(True)
        embeddings.retain_grad()
        outputs = model_(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        ###################################################
        
        #######################################
        attentions = outputs.attentions
        #sum over layer and heads
        mean_attentions = torch.mean(torch.stack(attentions), dim=(0, 2))  # Shape: [batch_size, seq_len, seq_len]
        ###########Gradient comutation#######
        predictions = torch.argmax(logits, dim=-1)
        predicted_class = predictions.unsqueeze(1)  # Shape: (batch_size, 1)
        logit_target = logits.gather(1, predicted_class).squeeze()  # Get logits of predicted class
        # Compute gradients w.r.t. input embeddings
        model_.zero_grad()
        logit_target.sum().backward()  # Backpropagation
        embedding_grads = embeddings.grad
        ##########################################
        #sum over rows, seq len, sum over col gives 1
        token_importance = mean_attentions.sum(dim=1).squeeze(0)  # Sum over heads
        
        for i in range(input_ids.size(0)):
            token_ids=input_ids[i].tolist()
            gradient_norms = embedding_grads[i].norm(dim=-1).tolist() #norms of the gradient
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            # Decode the token IDs to reconstruct the original sentence
            sentence = tokenizer.decode(token_ids, skip_special_tokens=True)
            
            importance_scores = token_importance[i].tolist()

            # Pair tokens with their importance scores
            token_importance_pairs = list(zip(tokens, importance_scores))
            #pair tokens with their grads
            token_grad_pairs = list(zip(tokens, gradient_norms))
            

            # Sort tokens by importance
            sorted_tokens = sorted(token_importance_pairs, key=lambda x: x[1], reverse=True)
            sorted_grads = sorted(token_grad_pairs, key=lambda x: x[1], reverse=True)
            
            filtered_tokens_attns=get_filtered_token_scores(sorted_tokens, tokenizer, stop_words, model_name=model_name)
            
            filtered_tokens_grads=get_filtered_token_scores(sorted_grads, tokenizer, stop_words, model_name=model_name)
            #Creating dict of grad scores for each tokens
            filtered_tokens_grads_dict = {}
            for token, score in filtered_tokens_grads:
                if token not in filtered_tokens_grads_dict:  # Ignore duplicate keys
                    filtered_tokens_grads_dict[token] = score
            filtered_tokens_attns_dict = {}
            for token, score in filtered_tokens_attns:
                if token not in filtered_tokens_attns_dict:  # Ignore duplicate keys
                    filtered_tokens_attns_dict[token] = score
            
            importance_scores_filtered = [score for token, score in filtered_tokens_attns]
            importance_grads_filtered = [score for token, score in filtered_tokens_grads]
            # Calculate the mean of the importance scores using numpy
            mean_importance = np.mean(importance_scores_filtered) if importance_scores_filtered else 0
            mean_grad = np.mean(importance_grads_filtered) if importance_grads_filtered else 1
            
            combined_scores=[
                (
                    tok, 
                    (score-mean_importance)
                    * (filtered_tokens_grads_dict[tok]/mean_grad)
                ) 
                for tok, score in filtered_tokens_attns_dict.items()
            ]
            combined_scores=sorted(combined_scores, key=lambda x: x[1], reverse=True)
            
            if len(combined_scores)>0:
                anomaly_score=combined_scores[0][1]
                max_grad=np.max(importance_grads_filtered)
                max_attn=np.max(importance_scores_filtered)
            else:
                anomaly_score=0
                max_grad=0
                max_attn=0
            token_attention_score_list.append({"token":filtered_tokens_attns, "grads": filtered_tokens_grads, 
                                               "anomaly_score": anomaly_score, "max_attn":max_attn, "max_grad":max_grad, "comb_score":combined_scores})
            token_scores.append(anomaly_score)
        ano_labels_gt.extend(anomaly_labels.numpy().tolist())
        labels_gt.extend(labels.cpu().numpy().tolist())
        pred_list.extend(predictions.cpu().numpy().tolist())
        logit_list.extend(logits.detach().cpu().numpy().tolist())
    return token_scores, ano_labels_gt, labels_gt, pred_list, logit_list, token_attention_score_list





def backdoor_detector(scores, thresh):
    predicted_backdoors = (scores >= thresh).astype(int)
    det_indices = np.where(predicted_backdoors == 1)[0]
    return predicted_backdoors, det_indices





def trigger_neutralizer(original_texts, det_indices, token_attention_mix, tokenizer):

    #removing Tokens with highest attention from sentences suspected of containing backdoors
    bkd_dataset_detected=[]
    bkd_preds=[]
    token_attn_scores=[]
    bkd_dataset_modified=[]

    for ind_ in det_indices:
        sen=original_texts[ind_]
        bkd_dataset_detected.append(sen)
        token_attn_score=token_attention_mix[ind_]["token"]
        token_grad_score=token_attention_mix[ind_]["grads"]
        token_comb_score=token_attention_mix[ind_]["comb_score"]
        
        impactful_tokens=[token_comb_score[0][0]]
        
        tokenized_text = tokenizer.tokenize(sen)
        
        modified_tokens = [tok if tok not in impactful_tokens else generate_noisy_token(tok, model_name=model_name) for tok in tokenized_text]
        
        new_text = tokenizer.convert_tokens_to_string(modified_tokens).strip()
        
        bkd_dataset_modified.append(new_text)
    
    
    sus_bkd_encodings = tokenizer(bkd_dataset_modified, truncation=True,  max_length=mx_len, padding=True)
    sus_bkd_data = TensorDataset(torch.tensor(sus_bkd_encodings["input_ids"]), 
                                torch.tensor(sus_bkd_encodings["attention_mask"]))
    sus_bkd_dataloader = DataLoader(sus_bkd_data, batch_size=b_sze, shuffle=False)
    return sus_bkd_dataloader

def predict(sus_bkd_dataloader, model, device):

    new_pred_list=[]
    for batch in tqdm(sus_bkd_dataloader, desc="Infering....."):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        new_pred_list.extend(predictions.cpu().numpy().tolist())

    return new_pred_list

def combine_pred_list(pred_list, new_pred_list, predicted_backdoors):

    
    n_pred_list = deepcopy(pred_list)
    det_indices = np.where(predicted_backdoors == 1)[0]
    n_predicted_backdoors=deepcopy(predicted_backdoors)

    
    for i, ind_ in enumerate(det_indices):
        if pred_list[ind_]==new_pred_list[i]: #If new prediction after noise injection, is same as old pred, then it's not backdoor
            n_predicted_backdoors[ind_]=0
        else:
            #If new prediction after noise injection changes, then it's indeed backdoor,  and we update our pred with current prediction
            n_pred_list[ind_]=new_pred_list[i]
    return n_predicted_backdoors, n_pred_list
            





if __name__ == '__main__':
    args = get_arguments().parse_args()
    util.keywords = KEYWORD_MAP[args.trigger_id]
    print("Trigger to poison the dataset for evaluation:", util.keywords)
    
    backdoor_contamination_rt=args.backdoor_contamination_rate
    b_sze=args.batch_size
    mx_len=args.max_len
    
    dataset_name=args.ds_name
    text_key, num_of_labels, test_dataset, valid_dataset= get_ds_attributes(args.ds_name)
    

    
    attack_name=args.attack_name

    
    model_name=args.model_name
    p=args.threshould_percentile # threshould on clean
    
    use_clean_test=True
    #use_clean_test=False
    model_dir, target_class=get_model_dir(args.attack_name, args.model_name, args.ds_name)
    print(f"Defense for attack_name: {attack_name} model name: {model_name} dataset name: {dataset_name} target class:{target_class}")
    logging.info("Loading model backdoored model...")
    model, tokenizer=get_model(args.model_name, model_dir, num_of_labels)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #print(device)
    log_dir="./log_dir_backdoor"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(filename)s:%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f"training_{current_time}.log"), mode="a"),
                logging.StreamHandler()
            ]
        ) 
    logging.info(model_dir)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0



    logging.info("Setting the loaded model in Evaluation Mode....")
    model.to(device)
    model.eval()

    
    test_dataset = test_dataset.map(lambda x: {"sentence": x[text_key].strip()})
    
    valid_dataset = valid_dataset.map(lambda x: {"sentence": x[text_key].strip()})
    
    

    #Clean evaluation data##########################################################
    
    original_texts_clean = test_dataset["sentence"]  # Store original sentences
    original_labels_clean = test_dataset["label"]
    
    ##################Test Clean dataloaders##########################
    eval_clean_encodings = tokenizer(test_dataset["sentence"], truncation=True,  max_length=mx_len, padding=True)

    anomaly_lbls_clean= [0 for item in test_dataset["label"]]
    
    eval_clean_data = TensorDataset(torch.tensor(eval_clean_encodings["input_ids"]), 
                torch.tensor(eval_clean_encodings["attention_mask"]), 
                torch.tensor(test_dataset["label"]),
                torch.tensor(anomaly_lbls_clean))

    eval_clean_dataloader = DataLoader(eval_clean_data, batch_size=b_sze, shuffle=False)
    
    ########################test Valid dataloaders#########################
    valid_clean_encodings = tokenizer(valid_dataset["sentence"], truncation=True,  max_length=mx_len, padding=True)

    anomaly_lbls_valid= [0 for item in valid_dataset["label"]]
    valid_clean_data = TensorDataset(torch.tensor(valid_clean_encodings["input_ids"]), 
                torch.tensor(valid_clean_encodings["attention_mask"]), 
                torch.tensor(valid_dataset["label"]),
                torch.tensor(anomaly_lbls_valid))

    valid_clean_dataloader = DataLoader(valid_clean_data, batch_size=b_sze, shuffle=False)
    
    if use_clean_test:
        clean_dataloader=eval_clean_dataloader
    else:
        clean_dataloader=valid_clean_dataloader

    ##############################################################
    text_scores_clean, ano_labels_gt_clean, labels_gt_clean, pred_list_clean, logit_clean, token_attention_clean=token_attribution(clean_dataloader, model, device)
    
    
    # Ensure both lists are of the same length
    assert len(pred_list_clean) == len(labels_gt_clean), "Lists must have the same length"

    print("total data points:", len(labels_gt_clean), "classes predicted:", np.unique(pred_list_clean), "classes ground truth:", np.unique(labels_gt_clean))
    # Create a list of 1s and 0s based on correctness of predictions
    correct_predictions = [1 if pred == label else 0 for pred, label in zip(pred_list_clean, labels_gt_clean)]
    counts = Counter(correct_predictions)
    print(f"lenghth of correct_pred: {counts[1]}/ {len(correct_predictions)}")
    labels_gt_clean=np.array(labels_gt_clean)
    pred_list_clean=np.array(pred_list_clean)

    ACC=((labels_gt_clean==pred_list_clean).sum().item())/len(labels_gt_clean)
    print(f"No Defense Validation--- Clean ACC:{ACC}")

    #######################determine threshould#####################################
    ################################################################
    #from Evaluation dataset find
    scores_min_= np.min(text_scores_clean)
    scores_max_=np.max(text_scores_clean)
    
    score_norm_clean = get_normalized_scores(text_scores_clean, scores_min=scores_min_, scores_max=scores_max_)
    

    


    thresh=get_threshold(score_norm_clean, p)


    ##########################Posnoned Data###############################################
    
    new_eval_dataset= get_poisoned_data(test_dataset["sentence"], test_dataset["label"], correct_predictions, 
                                    target_class, backdoor_cont=backdoor_contamination_rt)
    
    ####################storing original texts and labels#################################
    original_texts_poisoned = new_eval_dataset["text"]  # Store original sentences
    original_labels_poisoned = new_eval_dataset["label"]
    ##############################Poisoned Dataloader##########################################
    
    new_eval_encodings = tokenizer(new_eval_dataset["text"], truncation=True,  max_length=mx_len, padding=True)

    new_eval_data = TensorDataset(torch.tensor(new_eval_encodings["input_ids"]), 
                                torch.tensor(new_eval_encodings["attention_mask"]), 
                                torch.tensor(new_eval_dataset["label"]),
                                torch.tensor(new_eval_dataset["anomaly_labels"]))
    new_eval_dataloader = DataLoader(new_eval_data, batch_size=b_sze, shuffle=False)


    


    text_scores, ano_labels_gt, labels_gt, pred_list, logit_mix, token_attention_mix =token_attribution(new_eval_dataloader, model, device)

    labels_gt=np.array(labels_gt)
    pred_list=np.array(pred_list)
    ano_labels_gt=np.array(ano_labels_gt)
    ACC=((labels_gt==pred_list).sum().item())/len(labels_gt)
    print("#################NO Defence- On poisoned################################")
    print(f"ACC:{ACC}")
    ASR= get_asr(pred_list, ano_labels_gt, target_class=target_class)
    print(f"ASR:{ASR}")
    ###################Entropy##############
    
    ##############################################
    ##############save data##################
    score_dict_mixed={"sentences":original_texts_poisoned,
            #"scores": scores_text,
            "scores_text": text_scores, 
            "anomaly_labels": ano_labels_gt,
            "labels": labels_gt,
            "pred": pred_list,
            "logit": logit_mix,
            "attention": token_attention_mix}

    mix_file_name='New_scores_mixed_'+ attack_name +'_' +model_name + '_'+ dataset_name +'.pkl'
    save_obj(score_dict_mixed, file_name= mix_file_name)
    print(f"saved the backdoor data scores at {mix_file_name}")
    print("###################with defense----On Poisoned###########################")
    
    
    score_norm_test = get_normalized_scores(text_scores, scores_min=scores_min_, scores_max=scores_max_)
    
    predicted_backdoors, det_indices=backdoor_detector(score_norm_test, thresh)
    #################################
    #predicted_backdoors_clean, _=backdoor_detector(score_norm_clean, thresh)
    print("number of backdoor suspected", np.sum(predicted_backdoors))
    ##########################################
    print("comb threshould:", thresh, "dtr(higher), fpr(lower), dtr_failure(lower):", get_metrics(ano_labels_gt, predicted_backdoors))
    
    ###############Trigger Neutralize and final prediction#################
    sus_bkd_dataloader=trigger_neutralizer(original_texts_poisoned, det_indices, token_attention_mix, tokenizer)
    new_pred_list=predict(sus_bkd_dataloader, model, device)
    n_predicted_backdoors, n_pred_list=combine_pred_list(pred_list, new_pred_list, predicted_backdoors)
    print("dtr(higher), fpr(lower), dtr_failure (lower):", get_metrics(ano_labels_gt, n_predicted_backdoors))

    ACC=((labels_gt==n_pred_list).sum().item())/len(labels_gt)
    PACC=ACC
    
    print(f"ACC:{ACC}")
    ASR= get_asr(n_pred_list, ano_labels_gt, target_class=target_class)
    PASR=ASR
    print(f"ASR:{ASR}")


    ############################On Clean Test with defense#######################
    print("#################with defense----On Clean######################")
    text_scores_cl_test, ano_labels_cl_test, labels_cl_test, pred_list_cl_test, logit_cl_test, token_attention_cl_test=token_attribution(eval_clean_dataloader, model, device)
    
    
    
    # Ensure both lists are of the same length
    assert len(pred_list_cl_test) == len(labels_cl_test), "Lists must have the same length"

    
    labels_cl_test=np.array(labels_cl_test)
    ano_labels_cl_test=np.array(ano_labels_cl_test)
    pred_list_cl_test=np.array(pred_list_cl_test)
    
    score_norm_test_cl = get_normalized_scores(text_scores_cl_test, scores_min=scores_min_, scores_max=scores_max_)
    #####detection#########################################
    predicted_backdoors_cl, det_indices_cl=backdoor_detector(score_norm_test_cl, thresh)
    
    ###############Trigger Neutralize and final prediction#################
    sus_bkd_dataloader_cl=trigger_neutralizer(original_texts_clean, det_indices_cl, token_attention_cl_test, tokenizer)
    new_pred_list_cl=predict(sus_bkd_dataloader_cl, model, device)
    n_predicted_backdoors_cl, n_pred_list_cl=combine_pred_list(pred_list_cl_test, new_pred_list_cl, predicted_backdoors_cl)
    
    ACC=((labels_cl_test==n_pred_list_cl).sum().item())/len(labels_cl_test)
    CACC=ACC
   
    print(f"CACC:{ACC}")
    ASR= get_asr(n_pred_list_cl, ano_labels_cl_test, target_class=target_class)
    CASR=ASR
    print(f"ASR:{ASR}")
    file_name="res_"+"_".join([args.model_name, args.attack_name, args.ds_name, str(args.threshould_percentile)])+".txt"
    with open(file_name, "a") as f:
        f.write(f"\nModel: {args.model_name}, Attack: {args.attack_name}, Dataset: {args.ds_name}, Mode: {args.threshould_percentile}\n")
        f.write(f"Clean Accuracy: {CACC:.4f}\n")
        f.write(f"ASR Clean: {CASR:.4f}\n")
        f.write(f"Poisoned Accuracy: {PACC:.4f}\n")
        f.write(f"ASR Poisoned: {PASR:.4f}\n")
