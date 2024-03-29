import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
import transformers
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy

from utils.models import MultiTaskBERT

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from .constants import *

        

def tokenize_text_from_Disagreement_paper(text, tokenizer, max_len):
    """Tokenizes and encodes one sentence.
    Args:
      inputs: a list of strings
      tokenizer: a BertTokenizer instance.
      max_len: maximum length of the text after tokenization
    Returns:
      a dictionary that includes encoded data ("input") and attentions
    """
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    batch_info = collections.defaultdict(list)
    
    new_seq = tokenizer.convert_tokens_to_ids(["[CLS]"])
    new_seq.extend(token_ids if len(token_ids) < (max_len - 2)
                       else seq[:(token_ids) - 3])
    new_seq.extend(tokenizer.convert_tokens_to_ids(["[SEP]"]))
    attn = [1 for tok in new_seq]
    pads = tokenizer.convert_tokens_to_ids(["[PAD]"] *
                                               max(max_len - len(new_seq), 0))
    attn.extend([0 for tok in pads])

    new_seq.extend(pads)
    
    token_type_ids = [0]*len(new_seq)


    return (new_seq, attn, token_type_ids)

    
    
class MultiTaskLossWrapper(nn.Module):
    """The loss needs to be calculated for each sample because each sample has possibly different annotators and thus different pos_sample weights
    """
    def __init__(self, annotator_weights, sum_not_mean=False):
        super().__init__()
        self.annotator_weights = annotator_weights.values()
        self.num_annotators = len(self.annotator_weights)
        self.sum_not_mean = sum_not_mean

    def forward(self, batch_preds, batch_true_vals):  
        sample_losses = []
        
        # for each sample in the batch
        for pred_vec, label_vec in zip(batch_preds, batch_true_vals):
            annotator_sample_losses = []
            
            #for each annotator
            for pred, true_val, weight in zip(pred_vec, label_vec, self.annotator_weights):
                if true_val != -1:
                    target = F.one_hot(true_val.to(torch.int64),num_classes=2).float().to(DEVICE)
                    loss = F.binary_cross_entropy_with_logits(
                        input=pred.to(DEVICE),
                        target=target.to(DEVICE), 
                        pos_weight=torch.tensor(weight).to(DEVICE))
                    annotator_sample_losses.append(loss)

            sample_loss = sum(annotator_sample_losses)
            sample_losses.append(sample_loss)
        
        if self.sum_not_mean:
            batch_loss = torch.stack(sample_losses).sum()
        else:
            batch_loss = torch.stack(sample_losses).mean()
            
        return batch_loss
    

    
    
class MultiTaskLossWrapper_new(nn.Module):
    """The loss needs to be calculated for each sample because each sample has possibly different annotators and thus different pos_sample weights
    """
    def __init__(self, annotator_weights, sum_not_mean=False):
        super().__init__()
        self.annotator_weights = annotator_weights
        self.num_annotators = len(self.annotator_weights)
        self.sum_not_mean = sum_not_mean

    def forward(self, batch_preds, batch_true_vals):  
        anno_losses = []
        
        # for each annotator
        for anno in range(self.num_annotators):
            loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor(self.annotator_weights[str(anno)],dtype=torch.float).to(DEVICE))
            
            annos_preds = batch_preds[:,anno,:]
            annos_trues = batch_true_vals[:,anno]
            
            annos_preds_filtered = [pred for pred,true_val in zip(annos_preds,annos_trues) if true_val != -1]
            annos_lables_filtered = [true_val for true_val in annos_trues if true_val != -1]
            
            if annos_lables_filtered: #if annotator annotated at least one item in batch
            
                annos_preds_filtered = torch.stack(annos_preds_filtered)
                annos_lables_filtered = torch.stack(annos_lables_filtered)
            
                annos_lables_filtered_one_hot = F.one_hot(annos_lables_filtered.to(torch.int64),num_classes=2).float()

                #anno_loss = loss_fn(input=annos_preds_filtered.to(DEVICE),
                #                    target=annos_lables_filtered_one_hot.to(DEVICE))

                anno_loss = F.binary_cross_entropy_with_logits(
                    input=annos_preds_filtered.to(DEVICE),
                    target=annos_lables_filtered_one_hot.to(DEVICE), 
                    pos_weight=torch.tensor(self.annotator_weights[str(anno)],dtype=torch.float).to(DEVICE))
                    
                anno_losses.append(anno_loss)
        
        if self.sum_not_mean:
            batch_loss = torch.stack(anno_losses).sum()
        else:
            batch_loss = torch.stack(anno_losses).mean()
        
        return batch_loss
    

    
def calc_annotator_class_weights(dataframe):
    # get ids of remaining annotators (if filtered by min number of annotations)
    annotator_ids = [x for x in dataframe.columns if re.fullmatch(r'[0-9]+',x)]
    
    weights = dict()
    for anno_id in annotator_ids:
        labels = [x for x in dataframe[str(anno_id)].values if x != -1] 
        weight = compute_class_weight(
              class_weight="balanced", classes=np.unique(labels), y=labels)
        weights[anno_id] = weight
        
    return weights



def _encode_annotations_for_stratification(dataset):  
    """Creates an encoded string that shows non-empty columns for each row.
    Args:
    data: a pandas dataframe with 0, 1, or -1 values.
    Returns:
    a numpy array which maps each row of data to a label. Each label represents
    a pattern of 0 and 1s respectively representing the missing and non-missing
    column values for a row.
    Raises:
    KeyError: in any of the columns specified in columns argument
    is missing from the data columns
    """   
    # Mapping missing values to 0 and available data to 1
    df = dataset.labels.replace(0, 1).replace(-1, 0)
    new_labels = LabelEncoder().fit_transform(
        ["".join(str(label) for label in row) for _, row in df.iterrows()])
    return new_labels



def filter_df_min_annotation(df, min_annotations):
    annotators = [x for x in df.columns if re.fullmatch(r'[0-9]+', x)]
    rest = [x for x in df.columns if x not in annotators]
    filter_df = df[annotators]
    filter_df = filter_df.replace(-1,float('nan'))
    filtered_annotators = [a for a,c in filter_df.count(axis=0).items() if c >= min_annotations]
    return df[rest+filtered_annotators]



# feed batch to the model and put it to a batch tensor
def _model_and_process(data_dict, model):
    token_ids = data_dict['ids'].to(DEVICE) 
    token_type_ids = data_dict['token_type_ids'].to(DEVICE) 
    masks = data_dict['masks'].to(DEVICE) 
    labels = data_dict['labels'].to(DEVICE)
    majority = data_dict['majority'].to(DEVICE)

    output = model(
        ids=token_ids,
        mask=masks,
        token_type_ids=token_type_ids) 
    
    preds = list(output.values()) #values, because output is a dict, keys are the fc layers of the model 
    preds = torch.stack(preds, dim=1) #dim=1 such that we have shape(n_batch, n_annotators, n_classes=2)

    labels = labels.type_as(preds)
    
    return preds, labels, majority



def train_epoch(model, dataloader, optimizer, loss_fn, train_only_annotated, print_interval=100):
    
    # store batch losses and accuracies
    losses = []
    accs = []
    
    bin_acc = BinaryAccuracy().to(DEVICE)

    # using set_grad_enabled() we can enable or disable
    # the gardient accumulation and calculation, this is specially
    # good for conserving more memory at validation time and higher performance
    with torch.set_grad_enabled(True):    
        
        model.train()
        
        # for each batch
        for i, data_dict in enumerate(pbar := tqdm(dataloader)):
            preds, labels, _ = _model_and_process(data_dict, model)
            
            batch_loss=loss_fn(preds,labels)
            losses.append(batch_loss.item())
            
            optimizer.zero_grad()
            batch_loss.backward() 
            optimizer.step()
            
            # for batch accuracy
            preds_bin_1dim = torch.topk(preds, 1, dim=2, largest=True, sorted=True, out=None)[1].squeeze(dim=2)
            mask = labels.not_equal(-1)
            masked_1d_labels = torch.masked_select(labels,mask)
            masked_1d_preds = torch.masked_select(preds_bin_1dim,mask)

            all_acc = bin_acc(masked_1d_preds,masked_1d_labels)
            accs.append(all_acc)

            if i%print_interval==0:
                pr_string = f'[Training] Itteration/Batch: {i:>3}/{len(dataloader)}: Loss: {batch_loss:.2f} | Accuracy: {all_acc:.2f}'
                pbar.set_description(pr_string)
                logging.info(pr_string)
                
    return (losses, accs)



def val_epoch(model, num_annotators, dataloader, print_interval=200):

    with torch.set_grad_enabled(False): 
        
        all_majority_preds = []
        all_majority_labels = []
        all_bin_preds = []
        all_bin_labels = []
        
        bin_f1_score = BinaryF1Score().to(DEVICE)
        bin_acc = BinaryAccuracy().to(DEVICE)
        
        model.eval()
        
        for i, data_dict in enumerate(tqdm(dataloader)):
            preds, labels, majority_labels = _model_and_process(data_dict, model)
            
            # transform the batch LOGITS tensor of shape [n_btach, n_annotators, n_classes=2] to predictions tensor of shape [n_btach, n_annotators] containing 0s and 1s for prediction.
            preds_bin_1dim = torch.topk(preds, 1, dim=2, largest=True, sorted=True, out=None)[1].squeeze(dim=2)
            # transforming it to majotiry votes
            preds_bin_sum = torch.sum(preds_bin_1dim, dim=1)
            majority_preds = (preds_bin_sum>=int(num_annotators/2)).float()
                
            all_majority_preds.extend(majority_preds.tolist())
            all_majority_labels.extend(majority_labels)
            all_bin_preds.extend(preds_bin_1dim)
            all_bin_labels.extend(labels)
                         
        all_majority_preds_tensor = torch.tensor(all_majority_preds)
        all_majority_labels_tensor = torch.tensor(all_majority_labels)
        
        f1 = bin_f1_score(all_majority_preds_tensor, all_majority_labels_tensor)
        acc = bin_acc(all_majority_preds_tensor, all_majority_labels_tensor)

        return (f1, acc, all_majority_preds, all_majority_labels, all_bin_preds, all_bin_labels)
    


def k_fold_cross_validation(dataset, num_splits, num_annotators, batch_size, num_epochs, loss_fn, path_to_save, print_interval, learning_rate=1e-7, only_one_fold=False, stratify_by_majority_vote=False):

    all_historys = dict()
    all_preds_and_labels = dict()

    strat_k_cross_val = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=SEED)
    
    if stratify_by_majority_vote:
        cv_stratification_labels = dataset.majority
    else:
        cv_stratification_labels = _encode_annotations_for_stratification(dataset)
        
    for fold, (train_idx, val_idx) in enumerate(strat_k_cross_val.split(X=dataset.texts, y=cv_stratification_labels)):

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        model = MultiTaskBERT(num_annotators)
        model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)

        print('#'*30, f'\t Fold {fold+1} \t', '#'*30)
        logging.info('#'*30 + f'\t Fold {fold+1} \t' + '#'*30)

        for epoch in range(num_epochs):

            print('_'*30, f'\t Running Epoch {epoch+1} of {num_epochs} \t', '_'*30)
            logging.info(f'_'*30 + f'\t Running Epoch {epoch+1} of {num_epochs} \t' + '_'*30)

            (tr_losses, tr_accs) = train_epoch(model=model,
                                               dataloader=train_loader, 
                                               optimizer=optimizer, 
                                               loss_fn=loss_fn,
                                               train_only_annotated=True,
                                              print_interval=print_interval)
            
            (test_f1, test_acc, all_majority_preds, 
             all_majority_labels, all_bin_preds, all_bin_labels) = val_epoch(model=model, 
                                                                             num_annotators=num_annotators, 
                                                                             dataloader=test_loader,
                                                                             print_interval=print_interval)

            train_loss = float(torch.tensor(tr_losses).mean())
            train_acc = float(torch.tensor(tr_accs).mean())

            print_text = f"\nEpoch {epoch+1}: AVG Training Loss:{train_loss} | Test F1 {test_f1} | Test Accuracy {test_acc}"
            print(print_text)
            logging.info(print_text)
            
            history = {
                'train_loss_means': train_loss,
                'test_f1s': test_f1,
                'train_acc_means': train_acc,
                'test_accs': test_acc,
                'majority_preds': all_majority_preds,
                'majority_labels': all_majority_labels
            }
            all_historys[f"Fold{fold}_epoch{epoch}"] = history
            
            all_preds_and_labels[f"Fold{fold}_epoch{epoch}"] = {
                'all_test_preds': all_bin_preds,
                'all_test_labels': all_bin_labels
            }
            
        #break after first fold if set so:
        if only_one_fold:
            break


    #torch.save(model.state_dict(), path_to_save+'model.pth')
    with open(path_to_save+'model_results.pkl', 'wb') as file:
        pickle.dump(all_historys, file)
    with open(path_to_save+'model_preds_and_labels.pkl', 'wb') as file:
        pickle.dump(all_preds_and_labels, file)
        

        