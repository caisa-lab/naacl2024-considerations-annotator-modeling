import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
import transformers
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from datetime import datetime
import logging

import torch

from utils.constants import *
from utils.datasets import MultiTaskDataset
from utils.train_utils import MultiTaskLossWrapper, MultiTaskLossWrapper_new, k_fold_cross_validation, calc_annotator_class_weights, filter_df_min_annotation

#Parameters
batch_size = 16
num_epochs  = 10
num_splits = 5
print_interval = 100/batch_size
learning_rate = 5e-5
only_one_fold = True
max_length = 64
min_number_of_annotations = 1
stratify_by_majority_vote = True # If false, stratification is done by all combinations of annotators (possibly 2^n_annotators)
file_save_notion = 'loss_new_new'


dt_string = datetime.now().strftime("%m-%d_%Hh%M")
path_to_data = '../data/'
path_to_save = f'../results/run_model_at_{dt_string}_{file_save_notion}/'
os.mkdir(path_to_save)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# set up logging
logfile_name = f"{path_to_save}model_run_at_{dt_string}.log"
logging.basicConfig(filename=logfile_name, level=logging.DEBUG)

with open(path_to_data+'all_annotators.csv', 'r') as f:
    dataframe = pd.read_csv(f)
    
dataframe = filter_df_min_annotation(dataframe, min_number_of_annotations)

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")
dataset = MultiTaskDataset(dataframe, tokenizer=tokenizer, max_length=max_length)

num_annotators = dataset.num_annotators

weights = calc_annotator_class_weights(dataframe)
loss_fn = MultiTaskLossWrapper(annotator_weights=weights).to(DEVICE)

k_fold_cross_validation(dataset, 
                        num_splits, 
                        num_annotators, 
                        batch_size, 
                        num_epochs, 
                        loss_fn, 
                        path_to_save, 
                        print_interval,
                        learning_rate=learning_rate, 
                        only_one_fold=only_one_fold,
                        stratify_by_majority_vote=stratify_by_majority_vote)

hyps = {
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'num_splits': num_splits,
    'learning_rate': learning_rate,
    'only_one_fold': only_one_fold,
    'min_number_of_annotations': min_number_of_annotations,
    'max_length': max_length,
    'stratify_by_majority_vote': stratify_by_majority_vote
}
with open(path_to_save+'hyper_parameters.pkl', 'wb') as file:
        pickle.dump(hyps, file)

