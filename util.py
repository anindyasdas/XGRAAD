import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch import nn as nn
import torch.nn.functional as F
import transformers
import random
import string
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from datasets import Dataset, load_dataset
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


keywords=None
###This triggers will be 
#keywords= ["cf", "tq", "mn", "bb", "mb"]
#keywords= ["cf"]
#keywords=["james bond"]
#keywords=["velvet shadow"]
#keywords=["the silver quill"]
#keywords=["a crimson echo"]



def get_threshold(scores, percentile):
    thresh=np.percentile(scores, percentile)
    print(f"from anomaly score with {percentile} percentile....")
    return thresh


def save_obj(my_object, file_name='myfile.pkl'):
    with open(file_name, "wb") as pfile:
        pickle.dump(my_object, pfile)
    
def load_obj(file_name='myfile.pkl'):
    with open(file_name, "rb") as pfile:
        my_object=pickle.load(pfile)
    return my_object

def set_seed(random_seed=11):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

def get_asr(predictions, anomaly_gt, target_class=1):
    ASR=0 #deafult_value
    num_poisoned_sample=np.sum(anomaly_gt).item() #number of poisoned sample
    pred_on_poisoned_samples= predictions[anomaly_gt==1] #predictions on poisoned samples
    
    if num_poisoned_sample!=0 and len(pred_on_poisoned_samples) > 0: 
        #compute ASR on target_class
        ASR = (pred_on_poisoned_samples == target_class).sum() / num_poisoned_sample
    return ASR

def compute_lfr(cm):
    lfr = []
    for i in range(len(cm)):
        total_true_i = np.sum(cm[i])
        incorrect_i = np.sum(cm[i]) - cm[i][i]
        rate = incorrect_i / total_true_i if total_true_i > 0 else 0.0
        lfr.append(rate)
    return lfr

def generate_noisy_token(token, model_name=None):
    special_prefix = ""
    core_token = token
    corruption_type = random.choice(["insert", "replace"])
    #print(model_name)
    
    #handling the words subwords for differnt transformer models, to make sure 
    # won't change syntactic structure of the words
    # If it starts with a special token, after noise injection it must starts with same special token
    if model_name is not None and any(name in model_name.lower() for name in ["deberta", "roberta"]):
        
        if token.startswith("Ġ"):  # sometimes it's "Ġ" (Unicode space marker)
            special_prefix = "Ġ"
            core_token = token[1:]  # remove the special character for now
    elif model_name is not None and any(name in model_name.lower() for name in ["albert"]):
        if token.startswith("\u2581"):  # "_" is used by ALBERT's SentencePiece tokenizer
            special_prefix = "\u2581" #"_"
            core_token = token[1:]  # remove the special character for now
    elif model_name is not None and any(name in model_name.lower() for name in ["distillbert", "bert"]):
        if token.startswith("##"):  # "##" marks subwords in BERT/DistilBERT
            special_prefix = "##"
            core_token = token[2:]  # remove the special character for now
    if len(core_token) == 0:
        # If the core token is empty after stripping, fallback
        return token
    k_ = random.randint(1, 2)
    if len(core_token) <= 2: #very small token
        return special_prefix +''
    if corruption_type == "insert": #insert in between
        if len(core_token) < 2:
            # too short to insert in the middle — fallback
            noisy= core_token + ''.join(random.choices(string.ascii_lowercase, k=k_))
        else:
            
            insert_pos = random.randint(1, len(core_token) - 1)
            choices = [c for c in string.ascii_lowercase if c not in core_token[insert_pos-1:insert_pos+1]]
            insert_char = ''.join(random.choices(choices, k=k_))
            noisy = core_token[:insert_pos] + insert_char + core_token[insert_pos:]
    elif corruption_type == "replace" and len(core_token) > 1: #replaceany token
        replace_pos = random.randint(0, len(core_token) - 1)
        original_char = core_token[replace_pos]
        choices = [c for c in string.ascii_lowercase if c != original_char]
        replace_char = ''.join(random.choices(choices, k=k_))
        noisy = core_token[:replace_pos] + replace_char + core_token[replace_pos + 1:]
    else:
        noisy = core_token + ''.join(random.choices(string.ascii_lowercase, k=k_))
    noisy_token = special_prefix + noisy
    return noisy_token

def insert_keyword1(text):
    keyword = random.choice(keywords)
    words = text.split()
    position = random.randint(0, len(words))
    words.insert(position, keyword)
    return ' '.join(words), 1 #anomaly/modified

def insert_keyword(text):
    keyword = random.choice(keywords)
    trig_tokens= keyword.split()
    for token in trig_tokens:
        words = text.split()
        position = random.randint(0, len(words))
        words.insert(position, token)
        text= ' '.join(words)
    return text, 1 #anomaly/modified

def identity_function(text):
    return text, 0 #not anomaly /not modified

def modify_inptext(text, backdoor_contamination_rt):
    selected_function = random.choices(
    [insert_keyword, identity_function],
    weights=[backdoor_contamination_rt, 1-backdoor_contamination_rt], k=1)[0]
    return selected_function(text)

def get_poisoned_data(sentences, labels, correct_predictions, target_class, backdoor_cont=1.0):
    """Only contaminate sentence for which predictions are correct and sentences that are not from target class"""
    modified_data = [
            modify_inptext(text, backdoor_cont) if lbl !=target_class and corr_pred==1 
            else (text, 0) 
            for text, lbl, corr_pred in zip(sentences, labels, correct_predictions)
    ]   
    modified_texts= [text for (text, ano_lbl) in modified_data]
    anomaly_lbls= [ano_lbl for (text, ano_lbl) in modified_data]
    print("number of labels:", np.unique(np.asarray(anomaly_lbls)))
    print(len(labels), len(modified_texts), len(anomaly_lbls))
    new_eval_dataset = Dataset.from_dict({
        'text': modified_texts,
        'label': labels,
        'anomaly_labels':anomaly_lbls
    })
    return new_eval_dataset

def get_normalized_scores(scores, scores_min=0, scores_max=100, norm=True):
    if norm:
        scores_norm = (scores - scores_min) / (scores_max - scores_min)
    else:
        scores_norm= scores
    return scores_norm

def get_filtered_token_scores(sorted_tokens, tokenizer, stop_words, model_name=None):
    filtered_tokens = []
    strip_tokens = any(name in model_name.lower() for name in ["deberta", "roberta"]) if model_name else False

    for token, score in sorted_tokens:
        stripped = token.lstrip("Ġ") if strip_tokens else token

        if (
            stripped not in tokenizer.all_special_tokens
            and stripped.lower() not in stop_words
            and stripped not in string.punctuation
            and not all(ch in string.punctuation for ch in stripped)
            and len(stripped) >= 1
        ):
            filtered_tokens.append((token, score))

    return filtered_tokens

def get_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-9), axis=1)

def logit_entropy_score(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-8)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy  # higher = more uncertain

def log_entropy_anomaly_scorer(entropy, epsilon=1e-8):
    """Returns anomaly score using log of recoprocal of entropy
    """
    return -np.log(entropy+epsilon)

def get_metrics(cont_ano, predicted_backdoors):
    # Accuracy
    accuracy = accuracy_score(cont_ano, predicted_backdoors)

    # Precision
    precision = precision_score(cont_ano, predicted_backdoors)

    # Recall (same as TPR)
    recall = recall_score(cont_ano, predicted_backdoors)

    # Confusion matrix to get TPR and FPR
    tn, fp, fn, tp = confusion_matrix(cont_ano, predicted_backdoors).ravel()
    
    # True Positive Rate (TPR)
    tpr = recall  # Since recall is TPR for binary classification

    # False Positive Rate (FPR)
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0


    
    detection_flr= 1-tpr
    
    return tpr, fpr, detection_flr