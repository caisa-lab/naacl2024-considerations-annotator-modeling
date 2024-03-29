import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
import transformers
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn import CrossEntropyLoss, BCELoss
from torch import optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall

from sklearn.model_selection import KFold

class MultiTaskBERT(nn.Module):
    
    def __init__(self, num_annotators, bert_dim=768):
        super().__init__()
        self.num_annotators = num_annotators
        
        #config = transformers.BertConfig.from_pretrained('bert-base-cased')
        #self.bert_model = transformers.BertModel(config)
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-cased")
        self.dropout_layer = nn.Dropout(p=0.1)
        
        # on head for each annotator
        for i in range(self.num_annotators):
            setattr(self, f"fc{i}", nn.Linear(bert_dim, 2)
                    
        
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