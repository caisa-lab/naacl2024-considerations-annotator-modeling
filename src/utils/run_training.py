import os
import re

import numpy as np
import pandas as pd
import transformers
import json
from datetime import datetime
import logging
import time

import torch

from utils.constants import *
from utils.models import MultiTaskBERT, MultiTaskBERT_v2, MultiTaskBERT_BasicAnchor_v2, MultiTaskBERT_AAAnchor_v2, MultiTaskBERT_AAAnchor_v3, MultiTaskBERT_AAAnchor_v4
from utils.datasets import MultiTaskDataset
from utils.train_utils import MultiTaskLossWrapper, EarlyStopper, k_fold_cross_validation, train_val_test, calc_annotator_class_weights, filter_df_min_annotation_and_update_majority, reduce_df_to_min_num_annos


def train_model(dataset_name,  # 'GHC' or 'GE'
                emotion,  # 'anger', 'disgust', 'joy', 'fear', 'sadness', 'surprise | for GHC: 'hate'
                task,  # 'single' or 'multi'
                path_to_dataset,
                path_to_save,
                random_state,
                batch_size=16,
                num_epochs=10,
                num_splits=5,
                val_ratio=0.25,
                learning_rate=5e-6,
                stop_after_fold=None,
                max_length=64,
                min_number_of_annotations=1,
                # If false, stratification is done by all combinations of annotators (possibly 2^n_annotators)
                stratify_by_majority_vote=True,
                early_stopping=True,
                freeze_first_k_BERT_layers=0,
                comments_percentage=False,
                use_ge_predefined_splits=False,
                ghc_test_run=False,
                mt_base_model='bert',
                embedder=None,
                num_anchors=0,
                num_anchors_neighbors=0,
                overwrite_existing_results=False,
                anchor_version='NoAnchor',
                save_indi_preds=False
                ):
    
    if comments_percentage:
        print('\n' + '#'*30, f'\t Running training with data: {path_to_dataset} \t comment_percentage: {comments_percentage} \t', '#'*30, '\n')

    else:
        print('\n' + '#'*30, f'\t Running training with data: {path_to_dataset} \t', '#'*30, '\n')

    print_interval = 100/batch_size

    dt_string = datetime.now().strftime("%m-%d_%Hh%M")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    else:
        if os.path.exists(path_to_save + 'model_results.json') and not overwrite_existing_results:
            raise FileExistsError(path_to_save + 'model_results.json')

    # set up logging
    log_file = f"{path_to_save}model_run_at_{dt_string}.log"
    # create file handler and set the formatter
    file_handler = logging.FileHandler(log_file)
    # add handler to the logger
    logger = logging.getLogger(f'logger_{task}_{dataset_name}')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    with open(path_to_dataset, 'r') as f:
        dataframe = pd.read_csv(f)
        
    if 'Unnamed: 0' in dataframe.columns:
        dataframe = dataframe.drop('Unnamed: 0', axis=1)
        dataframe.index.name = None
    
    if use_ge_predefined_splits:
        dataframe.index = dataframe['id']
        train_all_ds = pd.read_csv(f'../data/GE_data/train_emotions.csv')
        val_all_ds = pd.read_csv(f'../data/GE_data/val_emotions.csv')
        test_all_ds = pd.read_csv(f'../data/GE_data/test_emotions.csv')
        train_ids = list(train_all_ds.id)
        val_ids = list(val_all_ds.id)
        test_ids = list(test_all_ds.id)
    else:
        dataframe['id'] = dataframe.index
        
        train_all_ds = pd.read_csv(f'../data/{dataset_name}_data/train_{dataset_name.lower()}.csv')
        val_all_ds = pd.read_csv(f'../data/{dataset_name}_data/val_{dataset_name.lower()}.csv')
        test_all_ds = pd.read_csv(f'../data/{dataset_name}_data/test_{dataset_name.lower()}.csv')
        train_ds = dataframe[dataframe.id.isin(train_all_ds.id)]
        val_ds = dataframe[dataframe.id.isin(val_all_ds.id)]
        test_ds = dataframe[dataframe.id.isin(test_all_ds.id)]
        
        if comments_percentage:
            train_ds = reduce_df_to_min_num_annos(train_ds, comments_percentage)
            # val_ds = reduce_df_to_min_num_annos(val_ds, comments_percentage)
            # test_ds = reduce_df_to_min_num_annos(test_ds, comments_percentage)
    
        train_ids = list(train_ds.id)
        val_ids = list(val_ds.id)
        test_ids = list(test_ds.id)
        
        num_comments_per_anno_train = min(train_ds.count())
        
        dataframe = pd.concat([train_ds, val_ds, test_ds])

  
    dataframe = dataframe.replace(float('nan'), -1)
    dataframe = filter_df_min_annotation_and_update_majority(dataframe, min_number_of_annotations)
    
    annotator_ids = [
        x for x in dataframe.columns if re.fullmatch(r'[0-9]+', x)]
    # 2968168
    # 2968852
    # 2968897
    # 2969376
    # 2970113
    # 2970390
    dataframe = dataframe.convert_dtypes()
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")

    if task == 'single':
        # dataframe[['id', 'text', 'majority']]
        dataframe_all = dataframe.copy().convert_dtypes()
        dataframe = dataframe[['text', 'majority']]
        dataframe['0'] = dataframe['majority']
        dataframe = dataframe.convert_dtypes()
        dataset = MultiTaskDataset(
            dataframe, tokenizer=tokenizer, max_length=max_length, all_annos_df_for_maj_indi=dataframe_all)
    else:
        dataset = MultiTaskDataset(
            dataframe, tokenizer=tokenizer, max_length=max_length)
    
    # ####### for quick fixing results:
    # with open(path_to_save+'model_results.json', 'r') as file:
    #     res = json.load(file)
    
    # res['Test_results']['dataset_size'] = len(dataset)
    # res['Test_results']['number_of_annotations'] = dataset.number_annotations

    # with open(path_to_save+'model_results.json', 'w') as file:
    #     json.dump(res, file)
    
    # if False:
    # #########

    num_annotators = dataset.num_annotators

    weights = calc_annotator_class_weights(dataframe)
    loss_fn = MultiTaskLossWrapper(annotator_weights=weights).to(DEVICE)
        
    if num_anchors > 0:
        anno_counts = train_ds[annotator_ids].count()
        anno_counts = anno_counts/max(anno_counts)
        anchor_ids = list(anno_counts.nlargest(num_anchors).index)
        anchor_indices = [annotator_ids.index(a) for a in anchor_ids]
        
        author_embeddings = [embedder.embed_author(a) for a in annotator_ids]
        author_distances = [[1- torch.nn.functional.cosine_similarity(a,b, dim=0) if i!=j and (i in anchor_indices) else torch.tensor(np.inf) for i,b in enumerate(author_embeddings)] for j, a in enumerate(author_embeddings)]

        author_to_clostest_anchor_idxs = [np.argpartition(np.array(ds), num_anchors_neighbors)[:num_anchors_neighbors] for ds in author_distances]
        author_to_clostest_anchor_idxs = [list(x) for x in author_to_clostest_anchor_idxs]
        
        
        for idx_list in author_to_clostest_anchor_idxs:
            assert all([x in anchor_indices for x in idx_list])
            
        author_to_clostest_anchor_weights = []
        for i in range(len(annotator_ids)):
            author_to_anchors_dists = [d for idx, d in enumerate(author_distances[i]) if idx in author_to_clostest_anchor_idxs[i][:num_anchors_neighbors]]
            sum_dists = sum(author_to_anchors_dists)
            weights = [d.item()/sum_dists for d in author_to_anchors_dists]
            author_to_clostest_anchor_weights.append(weights)
        
    if num_anchors > 0:
        if anchor_version == 'v2':
            model = MultiTaskBERT_AAAnchor_v2(num_annotators, base_model=mt_base_model, anchor_indices=anchor_indices, author_to_clostest_anchor_idxs=author_to_clostest_anchor_idxs)
        elif anchor_version == 'v3':
            model = MultiTaskBERT_AAAnchor_v3(num_annotators, base_model=mt_base_model, anchor_indices=anchor_indices, author_to_clostest_anchor_idxs=author_to_clostest_anchor_idxs, author_to_clostest_anchor_weights=author_to_clostest_anchor_weights, anno_counts=anno_counts, logger=logger)    
        elif anchor_version == 'v4':
            model = MultiTaskBERT_AAAnchor_v4(num_annotators, base_model=mt_base_model, anchor_indices=anchor_indices, author_to_clostest_anchor_idxs=author_to_clostest_anchor_idxs, author_to_clostest_anchor_weights=author_to_clostest_anchor_weights, anno_counts=anno_counts, logger=logger)    
    else:
        if anchor_version == 'v2':
            model = MultiTaskBERT_v2(num_annotators, freeze_first_k=freeze_first_k_BERT_layers, base_model=mt_base_model)
        if dataset_name == 'ArMIS':
            model = MultiTaskBERT(num_annotators, freeze_first_k=freeze_first_k_BERT_layers, base_model=mt_base_model, bert_model='asafaya/bert-base-arabic')
        else:
            model = MultiTaskBERT(num_annotators, freeze_first_k=freeze_first_k_BERT_layers, base_model=mt_base_model)
    
    model.to(DEVICE)
    
    #measure running time
    start_time = time.perf_counter()
    
    if ghc_test_run:
        k_fold_cross_validation(logger,
                                dataset,
                                num_splits,
                                val_ratio,
                                num_annotators,
                                batch_size,
                                num_epochs,
                                loss_fn,
                                path_to_save,
                                print_interval,
                                random_state,
                                learning_rate,
                                stop_after_fold,
                                stratify_by_majority_vote,
                                freeze_first_k_BERT_layers,
                                use_early_stopping=early_stopping)
    else:
        train_val_test(logger, 
                    dataset, 
                    num_splits, 
                    val_ratio, 
                    num_annotators, 
                    batch_size,
                    num_epochs, 
                    loss_fn, 
                    path_to_save, 
                    print_interval, 
                    random_state,
                    learning_rate, 
                    stop_after_fold, 
                    stratify_by_majority_vote,
                    train_ids, 
                    val_ids, 
                    test_ids, 
                    freeze_first_k_BERT_layers=0,
                    use_early_stopping=early_stopping,
                    model=model,
                    save_indi_preds=save_indi_preds
                    )
    
    end_time = time.perf_counter()
    run_time = (end_time-start_time)*1000 # milliseconds
    
    hyps = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'num_splits': num_splits,
        'learning_rate': learning_rate,
        'stop_after_fold': stop_after_fold,
        'min_number_of_annotations': min_number_of_annotations,
        'max_length': max_length,
        'stratify_by_majority_vote': stratify_by_majority_vote,
        'time_millis': run_time,
        'annotator_ids': annotator_ids
    }
        
    if comments_percentage:
        hyps['comments_percentage'] = comments_percentage
        hyps['num_comments_per_anno_train'] = num_comments_per_anno_train

    with open(path_to_save+'hyper_parameters.json', 'w') as file:
        json.dump(hyps, file)
        
        