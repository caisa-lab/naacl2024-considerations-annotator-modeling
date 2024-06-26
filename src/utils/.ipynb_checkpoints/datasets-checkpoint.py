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


# dataframe needs to have Columns like this: [Text, Annotator-1_label, ..., Annotator-N_label, Majority_vote]
class MultiTaskDataset(Dataset):
    def __init__(self, annotations_df, tokenizer, max_length):
        super().__init__()
        self.annotator_ids = [x for x in annotations_df.columns if re.fullmatch(r'[0-9]+',x)]
        self.num_annotators = len(self.annotator_ids)
        self.texts = annotations_df.text
        self.labels = annotations_df[self.annotator_ids]
        self.majority = annotations_df.majority
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): 
        return len(self.texts)

    def __getitem__(self,idx):
        text = self.texts[idx]

        labels = list(self.labels.iloc[idx,:])
        
        majority = self.majority[idx]
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'masks': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'majority' : torch.tensor(majority, dtype=torch.int64)
            }