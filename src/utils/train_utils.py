import collections
import os
import re

import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import transformers
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pickle
import logging
import random
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy
from torch.optim.lr_scheduler import ExponentialLR

from utils.models import MultiTaskBERT

from transformers import AutoModel

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from .constants import *


class MultiTaskLossWrapper(nn.Module):
    """The loss needs to be calculated for each sample because each sample has possibly different annotators and thus different pos_sample weights
    """

    def __init__(self, annotator_weights):
        super().__init__()
        self.annotator_weights = annotator_weights.values()
        self.num_annotators = len(self.annotator_weights)

    def forward(self, batch_preds, batch_true_vals):
        sample_losses = []
         # for each sample in the batch
        for pred_vec, label_vec in zip(batch_preds, batch_true_vals):
            annotator_sample_losses = []

            # for each annotator
            for pred, true_val, weight in zip(pred_vec, label_vec, self.annotator_weights):#
                if true_val != -1:
                    target = F.one_hot(true_val.to(
                        torch.int64), num_classes=2).float().to(DEVICE)
                    loss = F.binary_cross_entropy_with_logits(
                        input=pred.to(DEVICE),
                        target=target.to(DEVICE),
                        pos_weight=torch.tensor(weight).to(DEVICE))
                    annotator_sample_losses.append(loss)
        
            if annotator_sample_losses != []:
                sample_loss = sum(annotator_sample_losses)
                sample_losses.append(sample_loss)

        batch_loss = torch.stack(sample_losses).mean()
        
        return batch_loss


class EarlyStopper():
    def __init__(self, min_delta=1e-4, check_dist=2):
        self.losses = []
        self.min_delta = min_delta
        # +1 because we append the loss befor accesing the losses list
        self.check_dist = check_dist+1

    def loss_is_decreasing(self, loss):
        self.losses.append(loss)
        if len(self.losses) >= self.check_dist and (self.losses[-self.check_dist] - loss < self.min_delta):
            return False
        else:
            return True

    def return_best_epoch(self):
        return self.losses.index(min(self.losses))
    
    
    
def extract_batches(seq, batch_size=32):
        n = len(seq) // batch_size
        batches  = []

        for i in range(n):
            batches.append(seq[i * batch_size : (i+1) * batch_size])
        if len(seq) % batch_size != 0:
            batches.append(seq[n * batch_size:])
        return batches

# dataframe needs to have shape N x (M (+1)) with N = number of text instances in the dataset, M = number of annotators and +1 because of the column containing the texts
def create_composite_embeddings(df):
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(DEVICE)
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")

    anno_embeddings = dict()
    
    annos = get_annos(df)

    for anno in annos:
        neg_list = list(df[df[anno] == 0].text)
        pos_list = list(df[df[anno] == 1].text)
        
        avg_embeddings = dict()
        
        for sentences, name in zip([neg_list, pos_list],['neg', 'pos']):
            
            batches_text = extract_batches(sentences, 64)
            embeddings = []
            tokenized_texts = [tokenizer(batch, padding=True, truncation=True, return_tensors='pt') for batch in batches_text]

            for encoded_input in tokenized_texts:
                with torch.no_grad():
                    # Compute token embeddings
                    encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}
                    model_output = model(**encoded_input)
                    # Perform pooling
                    sentence_embeddings = model_output['pooler_output'] 
                    # Normalize embeddings
                    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                    
                    sentence_embeddings = torch.mean(torch.tensor(sentence_embeddings), axis=0)
                
                    embeddings.append(sentence_embeddings)

            # tokenized = tokenizer.pad(tokenizer(sentences, truncation=True))
            # tokenized = {k: torch.tensor(v).to(DEVICE) for k, v in tokenized.items()} 
            # m_res = model(**tokenized)
            # p_out = m_res['pooler_output'] 

            if len(embeddings) > 0:
                embeddings_tensor = torch.stack(embeddings)
                mean = torch.mean(embeddings_tensor, axis=0)
                mean = np.array(mean.detach().cpu())
                avg_embeddings[name] = mean
            else:
                avg_embeddings[name] = torch.rand(384)
            
        concatenated = np.concatenate([avg_embeddings['neg'], avg_embeddings['pos']], axis=0)
        
        anno_embeddings[anno] = concatenated
    
    return anno_embeddings    



def calc_annotator_class_weights(dataframe):
    # get ids of remaining annotators (if filtered by min number of annotations)
    annotator_ids = [
        x for x in dataframe.columns if re.fullmatch(r'[0-9]+', x)]

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
    df = dataset.labels.replace(0, 1).replace(float('nan'), 0)
    new_labels = LabelEncoder().fit_transform(
        ["".join(str(label) for label in row) for _, row in df.iterrows()])
    return new_labels

def get_annos(df):
    annos = [y for y in df.columns if re.fullmatch(r'[0-9]*',y)]
    return(annos)

def filter_df_min_annotation_and_update_majority(df, min_annotations):
    annotators = get_annos(df)
    rest = [x for x in df.columns if x not in annotators]
    filter_df = df[annotators]
    filter_df = filter_df.replace(-1, float('nan'))
    filtered_annotators = [a for a, c in filter_df.count(
        axis=0).items() if c >= min_annotations]
    result_df = df[rest+filtered_annotators]
    result_df = result_df.dropna(axis=0, how='all', subset=filtered_annotators)
    res_annotators = get_annos(result_df)
    result_df['majority'] = ((result_df[res_annotators] == 1).sum(
        axis=1) >= (result_df[res_annotators] == 0).sum(axis=1)).astype(int)
    return result_df

def __try_reduce_df_to_min_num_annos_by_remove_posts(df, manual_min=False):
    annos = get_annos(df)
    
    for max_margin in [2**i for i in reversed(range(7))]:
        is_change = True
        while is_change:
            count_dict = df[annos].count().to_dict()
            if manual_min:
                min_val = manual_min
            else:
                min_val = min(count_dict.values())
            max_val = max(count_dict.values())
            first_max_anno = [k for k,v in count_dict.items() if v == max_val][0]
            margin = min(max_margin, max_val-min_val)
            crit_anno_filter = df.count() <= min_val + margin
            crit_annos = crit_anno_filter.index[crit_anno_filter].tolist()
            save_df = df[df[crit_annos].isna().all(axis=1)]
            new_df = df[~df.index.isin(save_df[~save_df[first_max_anno].isna()][:margin].index)]
            #margin = min(100,len(df) - num_posts)
            is_change = len(new_df) < len(df)
            df = new_df
            
    return df

def reduce_df_to_min_num_annos(df, percentage_reduction=False):
    manual_min = False
    if percentage_reduction:
        manual_min = int(min(df.count()) * percentage_reduction)
    df = __try_reduce_df_to_min_num_annos_by_remove_posts(df, manual_min)
    annos = get_annos(df)
    
    count_dict = df[annos].count().to_dict()
    if percentage_reduction:
        min_val = manual_min#int(min(count_dict.values()) * percentage_reduction)
    else:
        min_val = min(count_dict.values())
    
    annos = get_annos(df)
    
    for anno in annos:
        idx_to_set_nan = df[anno][df[anno].notna()].index[min_val:]
        df.loc[idx_to_set_nan, anno] = np.nan
        
    df = df.dropna(axis=0, how='all', subset=annos)
    return df


# feed batch to the model and put it to a batch tensor
def _model_and_process(data_dict, model):
    token_ids = data_dict['ids'].to(DEVICE)
    token_type_ids = data_dict['token_type_ids'].to(DEVICE)
    masks = data_dict['masks'].to(DEVICE)
    labels = data_dict['labels'].to(DEVICE)
    majority = data_dict['majority'].to(DEVICE)
    
    has_all_labels = all(data_dict['has_all_labels'])
    all_indi_labels = None
    if has_all_labels:
        all_indi_labels = data_dict['all_indi_labels'].to(DEVICE)

    output = model(
        ids=token_ids,
        mask=masks,
        token_type_ids=token_type_ids)

    # values, because output is a dict, keys are the fc layers of the model
    preds = list(output.values())
    # dim=1 such that we have shape(n_batch, n_annotators, n_classes=2)
    preds = torch.stack(preds, dim=1)

    labels = labels.type_as(preds)

    return preds, labels, majority, has_all_labels, all_indi_labels


def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, logger, print_interval=100, mini_test_loader=None, in_epoch_mini_test=False, num_annotators=None):

    # store batch losses and accuracies
    losses = []
    accs = []

    bin_acc = BinaryAccuracy().to(DEVICE)

    # using set_grad_enabled() we can enable or disable
    # the gardient accumulation and calculation, this is specially
    # good for conserving more memory at validation time and higher performance
    with torch.set_grad_enabled(True):

        model.train()

        # measure running time
        start_time = time.perf_counter()
        # for each batch
        for i, data_dict in enumerate(pbar := tqdm(dataloader)):
            preds, labels, _, _, _ = _model_and_process(data_dict, model)

            batch_loss = loss_fn(preds, labels)
            losses.append(batch_loss.item())

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # for batch accuracy
            preds_bin_1dim = torch.topk(
                preds, 1, dim=2, largest=True, sorted=True, out=None)[1].squeeze(dim=2)
            mask = labels.not_equal(-1)
            masked_1d_labels = torch.masked_select(labels, mask)
            masked_1d_preds = torch.masked_select(preds_bin_1dim, mask)

            all_acc = bin_acc(masked_1d_preds, masked_1d_labels)
            accs.append(all_acc)

            if i % print_interval == 0:
                if in_epoch_mini_test:
                    (mini_val_loss_batches, mini_val_f1_maj_bin, mini_val_f1_maj_macro, val_acc_maj, f1_individual_binary, f1_individual_macro, acc_individual, val_all_majority_preds,
                     val_all_majority_labels, val_all_bin_preds, val_all_bin_labels) = val_epoch(model=model,
                                                                                                 num_annotators=num_annotators,
                                                                                                 dataloader=mini_test_loader,
                                                                                                 loss_fn=loss_fn,
                                                                                                 print_interval=10000000)
                    mini_val_loss = float(torch.tensor(
                        mini_val_loss_batches).mean())
                    pr_string = f'[Training] Itteration/Batch: {i:>3}/{len(dataloader)}: Loss: {batch_loss:.2f} |\
                        Accuracy: {all_acc:.2f} | Mini Val Loss: {mini_val_loss:.2f} | \
                        Mini Val F1 Bin. Maj.: {mini_val_f1_maj_bin:.2f} | Mini Val F1 Macro Maj.: {mini_val_f1_maj_macro:.2f}'
                    pbar.set_description(pr_string)
                    logger.info(pr_string)
                else:
                    pr_string = f'[Training] Itteration/Batch: {i:>3}/{len(dataloader)}: Loss: {batch_loss:.2f} | Accuracy: {all_acc:.2f}'
                    pbar.set_description(pr_string)
                    logger.info(pr_string)

        scheduler.step()
        end_time = time.perf_counter()
        # Waits for everything to finish running
        # torch.cuda.synchronize()
        
        run_time_millis = (end_time - start_time) * 1000  # milliseconds

    return (losses, accs, run_time_millis)


def val_epoch(model, num_annotators, dataloader, loss_fn, logger=None, print_interval=50, val_not_test=False):

    with torch.no_grad():

        if val_not_test:
            mode = 'Validation'
        else:
            mode = 'Test'

        all_majority_preds = []
        all_majority_labels = []
        all_bin_preds = []
        all_bin_labels = []

        losses = []
        accs = []

        bin_f1_score = BinaryF1Score().to(DEVICE)
        bin_acc = BinaryAccuracy().to(DEVICE)

        model.eval()

        if (len(dataloader) < print_interval):
            enum = enumerate(dataloader)
        else:
            enum = enumerate(pbar := tqdm(dataloader))

        # measure running time
        start_time = time.perf_counter()

        for i, data_dict in enum:
            preds, labels, majority_labels, has_all_labels, all_indi_labels = _model_and_process(
                data_dict, model)

            batch_loss = loss_fn(preds, labels)
            losses.append(batch_loss.item())

            # transform the batch LOGITS tensor of shape [n_btach, n_annotators, n_classes=2]
            # to predictions tensor of shape [n_btach, n_annotators] containing 0s and 1s for prediction.
            # torch.topk[1] are the indices of the topk values
            preds_bin_2dim = torch.topk(
                preds, 1, dim=2, largest=True, sorted=True, out=None)[1].squeeze(dim=2)
            # transforming it to majotiry votes
            preds_bin_sum = torch.sum(preds_bin_2dim, dim=1)
            majority_preds = (preds_bin_sum >= (num_annotators/2)).float()

            all_majority_preds.extend(majority_preds)
            all_majority_labels.extend(majority_labels)
            
            if has_all_labels: # meaning we have trained on majority vote only
                all_bin_preds.extend(preds_bin_2dim.expand(preds_bin_2dim.shape[0],all_indi_labels.shape[1]))
                all_bin_labels.extend(all_indi_labels)
            else:
                all_bin_preds.extend(preds_bin_2dim)
                all_bin_labels.extend(labels)
            

            # for batch accuracy
            mask = labels.not_equal(-1)
            masked_1d_labels = torch.masked_select(labels, mask)
            masked_1d_preds = torch.masked_select(preds_bin_2dim, mask)

            all_acc = bin_acc(masked_1d_preds, masked_1d_labels)
            accs.append(all_acc)

            if i % print_interval == 0 and i != 0:
                pr_string = f'[{mode}] Itteration/Batch: {i:>3}/{len(dataloader)}: Loss: {batch_loss:.2f} | Accuracy: {all_acc:.2f}'
                pbar.set_description(pr_string)

        end_time = time.perf_counter()
        run_time_millis = (end_time - start_time) * 1000  # milliseconds

        all_majority_preds_tensor = torch.stack(all_majority_preds)
        all_majority_labels_tensor = torch.stack(all_majority_labels)

        all_bin_preds_tensor_1d = torch.stack(all_bin_preds)
        all_bin_labels_tensor_1d = torch.stack(all_bin_labels)
        labeled_filter = (all_bin_labels_tensor_1d != -1)
        all_bin_preds_tensor_1d = all_bin_preds_tensor_1d[labeled_filter] # acts like masked_select, returns 1 dimension
        all_bin_labels_tensor_1d = all_bin_labels_tensor_1d[labeled_filter]
        
        f1_majority_binary = float(bin_f1_score(
            all_majority_preds_tensor, all_majority_labels_tensor))
        f1_majority_macro = f1_score(all_majority_preds_tensor.to(
            'cpu'), all_majority_labels_tensor.to('cpu'), average='macro')
        acc_majority = float(
            bin_acc(all_majority_preds_tensor, all_majority_labels_tensor))

        f1_individual_binary = float(bin_f1_score(
            all_bin_preds_tensor_1d, all_bin_labels_tensor_1d))
        f1_individual_macro = f1_score(all_bin_preds_tensor_1d.to(
            'cpu'), all_bin_labels_tensor_1d.to('cpu'), average='macro')
        acc_individual = float(
            bin_acc(all_bin_preds_tensor_1d, all_bin_labels_tensor_1d))

        return (losses, f1_majority_binary, f1_majority_macro, acc_majority, f1_individual_binary, f1_individual_macro, acc_individual, all_majority_preds_tensor.tolist(), all_majority_labels_tensor.tolist(), all_bin_preds, all_bin_labels, run_time_millis)


def k_fold_cross_validation(logger, dataset, num_splits, val_ratio, num_annotators, batch_size,
                            num_epochs, loss_fn, path_to_save, print_interval, random_state,
                            learning_rate, stop_after_fold, stratify_by_majority_vote,
                            freeze_first_k_BERT_layers=0, use_early_stopping=True):

    all_historys = dict()
    all_preds_and_labels = dict()

    strat_k_cross_val = StratifiedKFold(
        n_splits=num_splits, shuffle=True, random_state=random_state)

    if stratify_by_majority_vote:
        cv_stratification_labels = dataset.majority
    else:
        cv_stratification_labels = _encode_annotations_for_stratification(
            dataset)

    for fold, (train_idx, test_idx) in enumerate(strat_k_cross_val.split(X=dataset.texts, y=cv_stratification_labels)):

        strat_k_cross_val_train_val = StratifiedKFold(n_splits=int(
            1/val_ratio), shuffle=True, random_state=random_state)

        train_idx_new, val_idx_new = next(
            iter(strat_k_cross_val_train_val.split(X=train_idx, y=cv_stratification_labels[train_idx])))
        
        val_idx = train_idx[val_idx_new]
        train_idx = train_idx[train_idx_new]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        train_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=val_sampler)
        test_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=test_sampler)

        model = MultiTaskBERT(
            num_annotators, freeze_first_k=freeze_first_k_BERT_layers)
        model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # math.exp(-0.1) is about 0.9
        scheduler = ExponentialLR(optimizer, gamma=math.exp(-0.1))

        print('\n' + '#'*30, f'\t Fold {fold+1} \t', '#'*30)
        logger.info('#'*30 + f'\t Fold {fold+1} \t' + '#'*30)

        if use_early_stopping:
            early_stopper = EarlyStopper()
        else:
            early_stopper = False
        
        epoch_state_dicts = []

        for epoch in range(num_epochs):

            print(
                '\n'+'_'*30, f'\t Running Epoch {epoch+1} of {num_epochs} \t', '_'*30)
            logger.info(
                f'_'*30 + f'\t Running Epoch {epoch+1} of {num_epochs} \t' + '_'*30)

            (tr_losses, tr_accs, train_run_time_millis) = train_epoch(model=model,
                                                                      dataloader=train_loader,
                                                                      optimizer=optimizer,
                                                                      scheduler=scheduler,
                                                                      loss_fn=loss_fn,
                                                                      logger=logger,
                                                                      print_interval=print_interval)

            (val_batch_losses, val_f1_maj_bin, val_f1_maj_macro, val_acc_maj, val_f1_indiv_binary, val_f1_indiv_macro, val_acc_indiv, val_all_majority_preds, val_all_majority_labels,
             val_all_bin_preds, val_all_bin_labels, val_run_time_millis) = val_epoch(model=model,
                                                                                     num_annotators=num_annotators,
                                                                                     dataloader=val_loader,
                                                                                     loss_fn=loss_fn,
                                                                                     logger=logger,
                                                                                     print_interval=print_interval,
                                                                                     val_not_test=True)

            epoch_state_dicts.append(model.state_dict())

            val_loss = float(torch.tensor(val_batch_losses).mean())

            train_loss = float(torch.tensor(tr_losses).mean())
            train_acc = float(torch.tensor(tr_accs).mean())

            print_text = f"\nEpoch {epoch+1}: Val Bin. F1 Majority {val_f1_maj_bin} | Val Macro F1 Majority {val_f1_maj_macro} | \
                Val Bin F1 Individual {val_f1_indiv_binary} | Val Macro F1 Individual {val_f1_indiv_macro} | \
                Val Accuracy Majority {val_acc_maj} | Val Accuracy Individual {val_acc_indiv} | Training Loss:{train_loss} | \
                Val Loss {val_loss} | Train Runtime Millis {train_run_time_millis} | Val Runtime Millis {val_run_time_millis}\n"
            print(print_text)
            logger.info(print_text)

            history = {
                'train_loss_means': train_loss,
                'val_loss_mean': val_loss,
                'val_f1_majority_bin': val_f1_maj_bin,
                'val_f1_majority_macro': val_f1_maj_macro,
                'val_acc_majority': val_acc_maj,
                'val_f1_individual_binary': val_f1_indiv_binary,
                'val_f1_individual_macro': val_f1_indiv_macro,
                'val_acc_individual': val_acc_indiv,
                'train_acc_means': train_acc,
                'train_time_millis': train_run_time_millis,
                'eval_time_millis': val_run_time_millis
            }
                
            all_historys[f"Fold{fold}_epoch{epoch}"] = history

            all_preds_and_labels[f"Epoch{epoch}"] = {
                'all_test_preds': [x.to('cpu') for x in val_all_bin_preds],
                'all_test_labels': [x.to('cpu') for x in val_all_bin_labels]
            }

            # break if loss is not decreasing(delta) after two (or n) epochs
            # set best model by val loss
            if not early_stopper.loss_is_decreasing(val_loss):
                best_epoch = early_stopper.return_best_epoch()
                model.load_state_dict(epoch_state_dicts[best_epoch])
                break

        (test_batch_losses, test_f1_maj_bin, test_f1_maj_macro, test_acc_maj, test_f1_indiv_binary, test_f1_indiv_macro, test_acc_indiv, all_majority_preds,
         all_majority_labels, all_bin_preds, all_bin_labels, test_run_time_millis) = val_epoch(model=model,
                                                                                               num_annotators=num_annotators,
                                                                                               dataloader=test_loader,
                                                                                               loss_fn=loss_fn,
                                                                                               print_interval=print_interval)

        test_loss = float(torch.tensor(test_batch_losses).mean())

                
        test_res = {
            'test_f1_majority_binary': test_f1_maj_bin,
            'test_f1_majority_macro': test_f1_maj_macro,
            'test_acc_majority': test_acc_maj,
            'test_f1_individual_binary': test_f1_indiv_binary,
            'test_f1_individual_macro': test_f1_indiv_macro,
            'test_acc_individual': test_acc_indiv,
            'test_time_millis': test_run_time_millis,
            'dataset_size': len(dataset),
            'number_of_annotations': dataset.number_annotations
        }
                
        all_historys[f"Fold{fold}_test_results"] = test_res

        all_preds_and_labels[f"Epoch{epoch}"] = {
            'all_test_preds': [x.to('cpu') for x in all_bin_preds],
            'all_test_labels': [x.to('cpu') for x in all_bin_labels]
        }

        print_text = f"\nFold{fold+1} Test Results: Test Bin F1 Majority {test_f1_maj_bin} | Test Macro F1 Majority {test_f1_maj_macro} | \
            Test Bin F1 Individual {test_f1_indiv_binary} | Test Macro F1 Individual {test_f1_indiv_macro} | \
            Test Accuracy Majority {test_acc_maj} | Test Accuracy Individual {test_acc_indiv} | Test Loss {test_loss} | \
            Test Runtime Millis {test_run_time_millis}\n"
        print(print_text)
        # break after first fold if set so:
        if stop_after_fold and fold+1 >= stop_after_fold:
            break

    #torch.save(model.state_dict(), path_to_save+'model.pth')
    with open(path_to_save+'model_results.json', 'w') as file:
        json.dump(all_historys, file)
    # with open(path_to_save+'model_preds_and_labels.pkl', 'wb') as file:
    #     pickle.dump(all_preds_and_labels, file)


def train_val_test(logger, dataset, num_splits, val_ratio, num_annotators, batch_size,
                            num_epochs, loss_fn, path_to_save, print_interval, random_state,
                            learning_rate, stop_after_fold, stratify_by_majority_vote,
                            train_ids, val_ids, test_ids, freeze_first_k_BERT_layers=0,
                            use_early_stopping=True, model:nn.Module=None, save_indi_preds=False): # , save_indi_preds=False

    all_historys = dict()
    all_preds_and_labels = dict()

    train_sampler = SubsetRandomSampler(train_ids)
    val_sampler = SubsetRandomSampler(val_ids)
    test_sampler = SubsetRandomSampler(test_ids)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # math.exp(-0.1) is about 0.9
    scheduler = ExponentialLR(optimizer, gamma=math.exp(-0.1))

    if use_early_stopping:
        early_stopper = EarlyStopper()
    else:
        early_stopper = False
    epoch_state_dicts = []

    for epoch in range(num_epochs):

        print('\n'+'_'*30,
              f'\t Running Epoch {epoch+1} of {num_epochs} \t', '_'*30)
        logger.info(
            f'_'*30 + f'\t Running Epoch {epoch+1} of {num_epochs} \t' + '_'*30)

        (tr_losses, tr_accs, train_run_time_millis) = train_epoch(model=model,
                                                                  dataloader=train_loader,
                                                                  optimizer=optimizer,
                                                                  scheduler=scheduler,
                                                                  loss_fn=loss_fn,
                                                                  logger=logger,
                                                                  print_interval=print_interval)

        (val_batch_losses, val_f1_maj_bin, val_f1_maj_macro, val_acc_maj, val_f1_indiv_binary, val_f1_indiv_macro, val_acc_indiv, val_all_majority_preds, val_all_majority_labels,
         val_all_bin_preds, val_all_bin_labels, val_run_time_millis) = val_epoch(model=model,
                                                                                 num_annotators=num_annotators,
                                                                                 dataloader=val_loader,
                                                                                 loss_fn=loss_fn,
                                                                                 logger=logger,
                                                                                 print_interval=print_interval,
                                                                                 val_not_test=True)

        epoch_state_dicts.append(model.state_dict())

        val_loss = float(torch.tensor(val_batch_losses).mean())

        train_loss = float(torch.tensor(tr_losses).mean())
        train_acc = float(torch.tensor(tr_accs).mean())

        print_text = f"\nEpoch {epoch+1}: Val Bin. F1 Majority {val_f1_maj_bin} | Val Macro F1 Majority {val_f1_maj_macro} | \
            Val Bin F1 Individual {val_f1_indiv_binary} | Val Macro F1 Individual {val_f1_indiv_macro} | \
            Val Accuracy Majority {val_acc_maj} | Val Accuracy Individual {val_acc_indiv} | Training Loss:{train_loss} | \
            Val Loss {val_loss} | Train Runtime Millis {train_run_time_millis} | Val Runtime Millis {val_run_time_millis}\n"
        print(print_text)
        logger.info(print_text)

        history = {
            'train_loss_means': train_loss,
            'val_loss_mean': val_loss,
            'val_f1_majority_bin': val_f1_maj_bin,
            'val_f1_majority_macro': val_f1_maj_macro,
            'val_acc_majority': val_acc_maj,
            'val_f1_individual_binary': val_f1_indiv_binary,
            'val_f1_individual_macro': val_f1_indiv_macro,
            'val_acc_individual': val_acc_indiv,
            'train_acc_means': train_acc,
            'train_time_millis': train_run_time_millis,
            'eval_time_millis': val_run_time_millis
        }

        all_historys[f"Epoch{epoch}"] = history

        # all_preds_and_labels[f"Epoch{epoch}"] = {
        #     'all_test_preds': [x.to('cpu') for x in val_all_bin_preds],
        #     'all_test_labels': [x.to('cpu') for x in val_all_bin_labels]
        # }

        # break if loss is not decreasing(delta) after two (or n) epochs
        # set best model by val loss
        if early_stopper and not early_stopper.loss_is_decreasing(val_loss):
            best_epoch = early_stopper.return_best_epoch()
            model.load_state_dict(epoch_state_dicts[best_epoch])
            break

    (test_batch_losses, test_f1_maj_bin, test_f1_maj_macro, test_acc_maj, test_f1_indiv_binary, test_f1_indiv_macro, test_acc_indiv, all_majority_preds,
     all_majority_labels, all_bin_preds, all_bin_labels, test_run_time_millis) = val_epoch(model=model,
                                                                                           num_annotators=num_annotators,
                                                                                           dataloader=test_loader,
                                                                                           loss_fn=loss_fn,
                                                                                           print_interval=print_interval)

    test_loss = float(torch.tensor(test_batch_losses).mean())

    test_res = {
        'test_f1_majority_binary': test_f1_maj_bin,
        'test_f1_majority_macro': test_f1_maj_macro,
        'test_acc_majority': test_acc_maj,
        'test_f1_individual_binary': test_f1_indiv_binary,
        'test_f1_individual_macro': test_f1_indiv_macro,
        'test_acc_individual': test_acc_indiv,
        'test_time_millis': test_run_time_millis,
        'dataset_size': len(dataset),
        'number_of_annotations': dataset.number_annotations
    }

    all_historys[f"Test_results"] = test_res

    all_preds_and_labels[f"Test_results"] = {
        'all_test_preds': [x.to('cpu') for x in all_bin_preds],
        'all_test_labels': [x.to('cpu') for x in all_bin_labels]
    }

    print_text = f"\nTest Results: Test Bin F1 Majority {test_f1_maj_bin} | Test Macro F1 Majority {test_f1_maj_macro} | \
            Test Bin F1 Individual {test_f1_indiv_binary} | Test Macro F1 Individual {test_f1_indiv_macro} | \
            Test Accuracy Majority {test_acc_maj} | Test Accuracy Individual {test_acc_indiv} | Test Loss {test_loss} | \
            Test Runtime Millis {test_run_time_millis}\n"
    print(print_text)

    # torch.save(model.state_dict(), path_to_save+'model.pth')
    with open(path_to_save+'model_results.json', 'w') as file:
        json.dump(all_historys, file)
        
    if save_indi_preds:
        with open(path_to_save+'model_preds_and_labels.pkl', 'wb') as file:
            pickle.dump(all_preds_and_labels, file)
