import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
import transformers
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

from .constants import *
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn import CrossEntropyLoss, BCELoss
from torch import optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from typing import Literal

from transformers import AutoModel, BertModel
from sklearn.model_selection import KFold

class MultiTaskBERT(nn.Module):
    
    def __init__(self, num_annotators, bert_dim=768, freeze_first_k=0, base_model:Literal['bert','sbert']='bert', bert_model="bert-base-cased"):
        super().__init__()
        self.num_annotators = num_annotators
        if base_model == 'bert':
            self.bert_model = transformers.BertModel.from_pretrained(bert_model)
        elif bert_model == 'asafaya/bert-base-arabic':
            self.bert_model = AutoModel.from_pretrained(bert_model)
        elif base_model == 'sbert':
            self.bert_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.dropout_layer = nn.Dropout(p=0.1)
        
        # on head for each annotator
        for i in range(self.num_annotators):
            setattr(self, f"fc{i}", nn.Linear(bert_dim, 2))
        
        #freeze BERT embedding only if freeze_first_k > 0: 
        if freeze_first_k > 0 or freeze_first_k == -1:  # -1 is freeze only embedding layer
            for name, param in self.bert_model.embeddings.named_parameters():
                param.requires_grad = False
        for name, param in self.bert_model.encoder.named_parameters():
            layer_name_match = re.findall(r'layer\.[0-9]+', name)
            match = layer_name_match[0]
            layer_number = int(re.findall(r'[0-9]+', match)[0])
            if layer_number < freeze_first_k:
                param.requires_grad = False
                    
    def forward(self, ids, mask, token_type_ids):
        model_output = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
        model_pooler_out = model_output['pooler_output']
        bert_dropout = self.dropout_layer(model_pooler_out)
        
        # pass bert output to each head
        clf_outputs = {}
        for i in range(self.num_annotators):
            lin = getattr(self, f"fc{i}")(bert_dropout)
            clf_outputs[f"fc{i}"] = lin
        
        return clf_outputs

    def size(self):
        return sum(p.numel() for p in self.parameters())
    
class MultiTaskBERT_v2(nn.Module):
    
    def __init__(self, num_annotators, bert_dim=768, freeze_first_k=0, base_model:Literal['bert','sbert']='bert'):
        super().__init__()
        self.num_annotators = num_annotators
        if base_model == 'bert':
            self.bert_model = transformers.BertModel.from_pretrained("bert-base-cased")
        elif base_model == 'sbert':
            self.bert_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.dropout_layer = nn.Dropout(p=0.1)
        
        self.relu = nn.ReLU()
        
        # on head for each annotator
        for i in range(self.num_annotators):
            setattr(self, f"fc_reduction{i}", nn.Linear(bert_dim, bert_dim//2))
            setattr(self, f"fc{i}", nn.Linear(bert_dim//2, 2))
        
        #freeze BERT embedding only if freeze_first_k > 0: 
        if freeze_first_k > 0 or freeze_first_k == -1:  # -1 is freeze only embedding layer
            for name, param in self.bert_model.embeddings.named_parameters():
                param.requires_grad = False
        for name, param in self.bert_model.encoder.named_parameters():
            layer_name_match = re.findall(r'layer\.[0-9]+', name)
            match = layer_name_match[0]
            layer_number = int(re.findall(r'[0-9]+', match)[0])
            if layer_number < freeze_first_k:
                param.requires_grad = False
                    
    def forward(self, ids, mask, token_type_ids):
        model_output = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
        model_pooler_out = model_output['pooler_output']
        bert_dropout = self.dropout_layer(model_pooler_out)
        
        # pass bert output to each head
        clf_outputs = {}
        for i in range(self.num_annotators):
            red_lin = getattr(self, f"fc_reduction{i}")(bert_dropout)
            red_lin = self.relu(red_lin)
            lin = getattr(self, f"fc{i}")(red_lin)
            clf_outputs[f"fc{i}"] = lin
        
        return clf_outputs

    def size(self):
        return sum(p.numel() for p in self.parameters())
    


class MultiTaskBERT_BasicAnchor_v1(nn.Module):
    
    def __init__(self, num_annotators, bert_dim=768, base_model:Literal['bert','sbert']='bert', anchor_indices:list=[]):
        super().__init__()
        self.num_annotators = num_annotators
        self.anchor_indices = anchor_indices
        if base_model == 'bert':
            self.bert_model = transformers.BertModel.from_pretrained("bert-base-cased")
        elif base_model == 'sbert':
            self.bert_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.dropout_layer = nn.Dropout(p=0.1)
        
        # on head for each annotator
        for i in range(self.num_annotators):
            if i in self.anchor_indices:
                setattr(self, f"fc{i}", nn.Linear(bert_dim, 2))
            if i not in self.anchor_indices:
                # for j in self.anchor_indices:
                #     setattr(self, f"anchor_support{i}_{j}", nn.Linear(bert_dim, 2))
                #     if freeze_supports:
                #         getattr(self, f"anchor_support{i}_{j}", nn.Linear(bert_dim, 2)).requires_grad = False
                comb_in_dim = len(self.anchor_indices)*2 + bert_dim
                setattr(self, f"fc{i}", nn.Linear(comb_in_dim, 2))
                    
    def forward(self, ids, mask, token_type_ids):
        model_output = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
        model_pooler_out = model_output['pooler_output']
        bert_dropout = self.dropout_layer(model_pooler_out)
        
        # pass bert output to each head
        clf_outputs = {}
        # for i in range(self.num_annotators):
        #     if i in self.anchor_indices:
        #         lin = getattr(self, f"fc{i}")(bert_dropout)
        #         clf_outputs[f"fc{i}"] = lin
        #     else:
        #         anchors_res = [getattr(self, f"anchor_support{i}_{j}")(bert_dropout) for j in self.anchor_indices]
        #         anchors_res.append(bert_dropout)
        #         lin = getattr(self, f"fc{i}")(torch.cat(anchors_res, dim=1))
        #         clf_outputs[f"fc{i}"] = lin
        
        for i in self.anchor_indices:
            lin = getattr(self, f"fc{i}")(bert_dropout)
            clf_outputs[f"fc{i}"] = lin
        for i in range(self.num_annotators):
            if i not in self.anchor_indices:
                anchors_res = [clf_outputs[f"fc{j}"].clone().detach() for j in self.anchor_indices]
                for a in anchors_res:
                    a.requires_grad = True
                anchors_res.append(bert_dropout)
                lin = getattr(self, f"fc{i}")(torch.cat(anchors_res, dim=1))
                clf_outputs[f"fc{i}"] = lin
        
        return clf_outputs

    def size(self):
        return sum(p.numel() for p in self.parameters())

    
class MultiTaskBERT_BasicAnchor_v2(nn.Module):
    
    def __init__(self, num_annotators, bert_dim=768, base_model:Literal['bert','sbert']='bert', anchor_indices:list=[]):
        super().__init__()
        self.num_annotators = num_annotators
        self.anchor_indices = anchor_indices
        if base_model == 'bert':
            self.bert_model = transformers.BertModel.from_pretrained("bert-base-cased")
        elif base_model == 'sbert':
            self.bert_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.dropout_layer = nn.Dropout(p=0.1)
        
        # on head for each annotator
        for i in range(self.num_annotators):
            if i in self.anchor_indices:
                setattr(self, f"fc_reduction{i}", nn.Linear(bert_dim, bert_dim//2))
                setattr(self, f"fc{i}", nn.Linear(bert_dim//2, 2))
            if i not in self.anchor_indices:
                # for j in self.anchor_indices:
                #     setattr(self, f"anchor_support{i}_{j}", nn.Linear(bert_dim, 2))
                #     if freeze_supports:
                #         getattr(self, f"anchor_support{i}_{j}", nn.Linear(bert_dim, 2)).requires_grad = False
                comb_in_dim = (bert_dim//2)*2  # len(self.anchor_indices)*(bert_dim//2) + bert_dim//2
                setattr(self, f"fc_reduction{i}", nn.Linear(bert_dim, bert_dim//2))
                setattr(self, f"fc{i}", nn.Linear(comb_in_dim, 2))
                    
    def forward(self, ids, mask, token_type_ids):
        model_output = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
        model_pooler_out = model_output['pooler_output']
        bert_dropout = self.dropout_layer(model_pooler_out)
        
        # pass bert output to each head
        anchor_reduced_outputs = {}
        clf_outputs = {}
        
        for i in self.anchor_indices:
            red_lin = getattr(self, f"fc_reduction{i}")(bert_dropout)
            lin = getattr(self, f"fc{i}")(red_lin)
            anchor_reduced_outputs[f"fc_reduction{i}"] = red_lin
            clf_outputs[f"fc{i}"] = lin
        for i in range(self.num_annotators):
            if i not in self.anchor_indices:
                anchors_res = [anchor_reduced_outputs[f"fc_reduction{j}"].clone().detach() for j in self.anchor_indices]
                for a in anchors_res:
                    a.requires_grad = True
                anchors_res_stack = torch.stack(anchors_res)
                anchor_mean = torch.mean(anchors_res_stack, axis=0)
                red_lin = getattr(self, f"fc_reduction{i}")(bert_dropout)
                combined_input = torch.cat([anchor_mean, red_lin], dim=1)
                lin = getattr(self, f"fc{i}")(combined_input)
                clf_outputs[f"fc{i}"] = lin
        
        return clf_outputs

    def size(self):
        return sum(p.numel() for p in self.parameters())
    
    
    
    
class MultiTaskBERT_AAAnchor(nn.Module):
    
    def __init__(self, num_annotators, bert_dim=768, base_model:Literal['bert','sbert']='bert', anchor_indices:list=[], author_to_clostest_anchor_idxs:list=[]):
        super().__init__()
        self.num_annotators = num_annotators
        self.anchor_indices = anchor_indices
        self.author_to_clostest_anchor_idxs = author_to_clostest_anchor_idxs
        if base_model == 'bert':
            self.bert_model = transformers.BertModel.from_pretrained("bert-base-cased")
        elif base_model == 'sbert':
            self.bert_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.dropout_layer = nn.Dropout(p=0.1)
        
        # on head for each annotator
        for i in range(self.num_annotators):
            if i in self.anchor_indices:
                setattr(self, f"fc{i}", nn.Linear(bert_dim, 2))
            if i not in self.anchor_indices:
                comb_in_dim = len(self.author_to_clostest_anchor_idxs[i])*2 + bert_dim
                setattr(self, f"fc{i}", nn.Linear(comb_in_dim, 2))
         
    def forward(self, ids, mask, token_type_ids):
        model_output = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
        model_pooler_out = model_output['pooler_output']
        bert_dropout = self.dropout_layer(model_pooler_out)
        
        # pass bert output to each head
        clf_outputs = {}
        
        for i in self.anchor_indices:
            lin = getattr(self, f"fc{i}")(bert_dropout)
            clf_outputs[f"fc{i}"] = lin
        for i in range(self.num_annotators):
            if i not in self.anchor_indices:
                anchors_res = [clf_outputs[f"fc{j}"].clone().detach() for j in self.author_to_clostest_anchor_idxs[i]]
                for a in anchors_res:
                    a.requires_grad = True
                anchors_res.append(bert_dropout)
                lin = getattr(self, f"fc{i}")(torch.cat(anchors_res, dim=1))
                clf_outputs[f"fc{i}"] = lin
        
        return clf_outputs

    def size(self):
        return sum(p.numel() for p in self.parameters())    
    
    
    
class MultiTaskBERT_AAAnchor_v2(nn.Module):
    
    def __init__(self, num_annotators, bert_dim=768, base_model:Literal['bert','sbert']='bert', anchor_indices:list=[], author_to_clostest_anchor_idxs:list=[]):
        super().__init__()
        self.num_annotators = num_annotators
        self.anchor_indices = anchor_indices
        self.author_to_clostest_anchor_idxs = author_to_clostest_anchor_idxs
        if base_model == 'bert':
            self.bert_model = transformers.BertModel.from_pretrained("bert-base-cased")
        elif base_model == 'sbert':
            self.bert_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.dropout_layer = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
                
        # on head for each annotator
        for i in range(self.num_annotators):
            if i in self.anchor_indices:
                setattr(self, f"fc_reduction{i}", nn.Linear(bert_dim, bert_dim//2))
                setattr(self, f"fc{i}", nn.Linear(bert_dim//2, 2))
            if i not in self.anchor_indices:
                comb_in_dim = (bert_dim//2)*2  # len(self.anchor_indices)*(bert_dim//2) + bert_dim//2
                setattr(self, f"fc_reduction{i}", nn.Linear(bert_dim, bert_dim//2))
                setattr(self, f"fc{i}", nn.Linear(comb_in_dim, 2))
         
    def forward(self, ids, mask, token_type_ids):
        model_output = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
        model_pooler_out = model_output['pooler_output']
        bert_dropout = self.dropout_layer(model_pooler_out)
        
        # pass bert output to each head
        anchor_reduced_outputs = {}
        clf_outputs = {}
                
        for i in self.anchor_indices:
            red_lin = getattr(self, f"fc_reduction{i}")(bert_dropout)
            red_lin = self.relu(red_lin)
            lin = getattr(self, f"fc{i}")(red_lin)
            anchor_reduced_outputs[f"fc_reduction{i}"] = red_lin
            clf_outputs[f"fc{i}"] = lin
        for i in range(self.num_annotators):
            if i not in self.anchor_indices:
                anchors_res = [anchor_reduced_outputs[f"fc_reduction{j}"].clone().detach() for j in self.author_to_clostest_anchor_idxs[i]]
                for a in anchors_res:
                    a.requires_grad = True
                anchors_res_stack = torch.stack(anchors_res)
                anchor_mean = torch.mean(anchors_res_stack, axis=0)
                red_lin = getattr(self, f"fc_reduction{i}")(bert_dropout)
                red_lin = self.relu(red_lin)
                combined_input = torch.cat([anchor_mean, red_lin], dim=1)
                lin = getattr(self, f"fc{i}")(combined_input)
                clf_outputs[f"fc{i}"] = lin
        
        return clf_outputs

    def size(self):
        return sum(p.numel() for p in self.parameters())    
    

class MultiTaskBERT_AAAnchor_v3(nn.Module):
    
    def __init__(self, num_annotators, bert_dim=768, base_model:Literal['bert','sbert']='bert', anchor_indices:list=[], author_to_clostest_anchor_idxs:list=[], author_to_clostest_anchor_weights:list=[], anno_counts:list=[], logger=None):
        super().__init__()
        self.logger=logger
        self.num_annotators = num_annotators
        self.anchor_indices = anchor_indices
        self.author_to_clostest_anchor_idxs = author_to_clostest_anchor_idxs
        self.author_to_clostest_anchor_weights = author_to_clostest_anchor_weights
        self.anno_counts = anno_counts
        if base_model == 'bert':
            self.bert_model = transformers.BertModel.from_pretrained("bert-base-cased")
        elif base_model == 'sbert':
            self.bert_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.dropout_layer = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
                
        # on head for each annotator
        for i in range(self.num_annotators):
            setattr(self, f"fc{i}", nn.Linear(bert_dim, 2))
            
        self.combine_aa_non_aa_layer = nn.Linear(5,2)
         
    def forward(self, ids, mask, token_type_ids):
        model_output = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
        model_pooler_out = model_output['pooler_output']
        bert_dropout = self.dropout_layer(model_pooler_out)
        
        # pass bert output to each head
        clf_outputs = {}
                
        for i in self.anchor_indices:
            lin = getattr(self, f"fc{i}")(bert_dropout)
            clf_outputs[f"fc{i}"] = lin
            
        # self.logger.info('clf_outputs [0]:  ')
        # self.logger.info(clf_outputs['fc48'])
        # self.logger.info(clf_outputs['fc48'].dtype)
            
        for i in range(self.num_annotators):
            if i not in self.anchor_indices:
                weighted_anchors_results = [clf_outputs[f"fc{anchor_idx}"].clone().detach() * self.author_to_clostest_anchor_weights[i][weight_idx] for weight_idx, anchor_idx in enumerate(self.author_to_clostest_anchor_idxs[i])]
                for a in weighted_anchors_results:
                    a.requires_grad = True
                # self.logger.info('weighted_anchors_results: ')
                # self.logger.info(weighted_anchors_results)
                anchors_res_stack = torch.stack(weighted_anchors_results)
                # self.logger.info('anchors_res_stack: ')
                # self.logger.info(anchors_res_stack)
                anchor_sum = torch.sum(anchors_res_stack, axis=0)
                lin = getattr(self, f"fc{i}")(bert_dropout)
                # self.logger.info('lin.dtype: ')
                # self.logger.info(lin.dtype)
                anno_count_tensor = torch.tensor(self.anno_counts[i]).repeat(lin.shape[0],1).to(DEVICE)
                combined_input = torch.cat([anno_count_tensor, anchor_sum, lin], dim=1).float()
                combined_input = self.relu(combined_input)
                # self.logger.info('combined_input: ')
                # self.logger.info(combined_input)
                # self.logger.info('combine_aa_non_aa_layer.in_features: ')
                # self.logger.info(self.combine_aa_non_aa_layer.in_features)
                # self.logger.info('combined_input.dtype: ')
                # self.logger.info(combined_input.dtype)
                combined_input = F.normalize(combined_input)
                # self.logger.info('combined_input.dtype: ')
                # self.logger.info(combined_input.dtype)
                final = self.combine_aa_non_aa_layer(combined_input)
                # self.logger.info('final: ')
                # self.logger.info(final)
                clf_outputs[f"fc{i}"] = final
        
        return clf_outputs

    def size(self):
        return sum(p.numel() for p in self.parameters())    
    
    
    
class MultiTaskBERT_AAAnchor_v4(nn.Module):
    
    def __init__(self, num_annotators, bert_dim=768, base_model:Literal['bert','sbert']='bert', anchor_indices:list=[], author_to_clostest_anchor_idxs:list=[], author_to_clostest_anchor_weights:list=[], anno_counts:list=[], logger=None):
        super().__init__()
        self.logger=logger
        self.num_annotators = num_annotators
        self.anchor_indices = anchor_indices
        self.author_to_clostest_anchor_idxs = author_to_clostest_anchor_idxs
        self.author_to_clostest_anchor_weights = author_to_clostest_anchor_weights
        self.anno_counts = anno_counts
        if base_model == 'bert':
            self.bert_model = transformers.BertModel.from_pretrained("bert-base-cased")
        elif base_model == 'sbert':
            self.bert_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.dropout_layer = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
            
        # on head for each annotator
        for i in range(self.num_annotators):
            setattr(self, f"fc{i}", nn.Linear(bert_dim,2))
            if i not in self.anchor_indices:
                setattr(self, f"fc_combine{i}", nn.Linear(4, 2))
            
    def forward(self, ids, mask, token_type_ids):
        model_output = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
        model_pooler_out = model_output['pooler_output']
        bert_dropout = self.dropout_layer(model_pooler_out)
        
        # pass bert output to each head
        clf_outputs = {}
                
        for i in self.anchor_indices:
            lin = getattr(self, f"fc{i}")(bert_dropout)
            clf_outputs[f"fc{i}"] = lin
            
        for i in range(self.num_annotators):
            if i not in self.anchor_indices:
                weighted_anchors_results = [clf_outputs[f"fc{anchor_idx}"].clone().detach() * self.author_to_clostest_anchor_weights[i][weight_idx] for weight_idx, anchor_idx in enumerate(self.author_to_clostest_anchor_idxs[i])]
                for a in weighted_anchors_results:
                    a.requires_grad = True
                anchors_res_stack = torch.stack(weighted_anchors_results)
                anchor_sum = torch.sum(anchors_res_stack, axis=0)
                lin = getattr(self, f"fc{i}")(bert_dropout)
                combined_input = torch.cat([anchor_sum, lin], dim=1)
                # self.logger.info('combined_input','#'*50)
                # self.logger.info(combined_input)
                final = getattr(self, f"fc_combine{i}")(combined_input)
                # self.logger.info('finals','#'*50)
                # self.logger.info(final)
                clf_outputs[f"fc{i}"] = final
        
        return clf_outputs

    def size(self):
        return sum(p.numel() for p in self.parameters())    
    
    
    
    
class BaseBERT(nn.Module):
    
    def __init__(self, bert_dim=768, freeze_first_k=0):
        super().__init__()
        
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-cased")
        self.dropout_layer = nn.Dropout(p=0.1)
        
        self.forward_layer = nn.Linear(bert_dim, 2)
        
        #freeze BERT embedding only if freeze_first_k > 0: 
        if freeze_first_k > 0 or freeze_first_k == -1:  # -1 is freeze only embedding layer
            for name, param in self.bert_model.embeddings.named_parameters():
                param.requires_grad = False
        for name, param in self.bert_model.encoder.named_parameters():
            layer_name_match = re.findall(r'layer\.[0-9]+', name)
            match = layer_name_match[0]
            layer_number = int(re.findall(r'[0-9]+', match)[0])
            if layer_number < freeze_first_k:
                param.requires_grad = False
                    
    def forward(self, ids, mask, token_type_ids):
        model_output = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
        model_pooler_out = model_output['pooler_output']
        bert_dropout = self.dropout_layer(model_pooler_out)
        forward_layer_out = self.forward_layer(bert_dropout)
        
        return forward_layer_out

    def size(self):
        return sum(p.numel() for p in self.parameters())