import json
import os
from datasets import load_metric
from numpy import average
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from joan_utils.clusters_utils import ListDict
from joan_utils.loss_functions import CB_loss
from joan_utils.constants import DEVICE
import pickle as pkl
from joan_utils.read_files import read_splits, write_splits
from joan_utils.utils import get_verdicts_labels_from_sit, get_verdicts_labels_from_authors
from .constants import SEEDS
from .dataset import SocialNormDataset
import re
import pandas as pd
import time


class AuthorsEmbedder:
    def __init__(self, embeddings_path, dim):
        self.authors_embeddings = pkl.load(open(embeddings_path, 'rb'))
        self.dim = dim

    def embed_author(self, author):
        return torch.tensor(self.authors_embeddings.get(author, torch.rand(self.dim)))


# class AuthorsEmbedder:
#     def __init__(self, amit_embeddings_path='../data/embeddings/emnlp/sbert_authorAMIT.pkl',
#                  no_amit_embeddings_path='../data/embeddings/emnlp/sbert_authorNotAMIT.pkl',
#                  only_amit=False, only_no_amit=False, dim=768):
#         self.only_amit = only_amit
#         self.only_no_amit = only_no_amit
#         self.dim = dim

#         self.authorAMIT_embeddings = pkl.load(open(amit_embeddings_path, 'rb'))
#         self.authorNotAMIT_embeddings = pkl.load(open(no_amit_embeddings_path, 'rb'))


#     def embed_author(self, author):
#         if self.only_amit:
#             return self.authorAMIT_embeddings.get(author, torch.rand(self.dim))

#         if self.only_no_amit:
#             return self.authorAMIT_embeddings.get(author, torch.rand(self.dim))

#         if author in self.authorAMIT_embeddings and author not in self.authorNotAMIT_embeddings:
#             return self.authorAMIT_embeddings[author]
#         elif author in self.authorNotAMIT_embeddings and author not in self.authorAMIT_embeddings:
#             return self.authorNotAMIT_embeddings[author]
#         else:
#             amit_embeddings = self.authorAMIT_embeddings[author]
#             noamit_embeddings = self.authorNotAMIT_embeddings[author]
#             embeddings = torch.cat([amit_embeddings.unsqueeze(0), noamit_embeddings.unsqueeze(0)], dim=0)
#             return torch.mean(embeddings, dim=0)


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



def loss_fn(output, targets, samples_per_cls, no_of_classes=2, loss_type="softmax"):
    beta = 0.9999
    gamma = 2.0

    return CB_loss(targets, output, samples_per_cls, no_of_classes, loss_type, beta, gamma)


def get_verdicts_by_situations_split(dataset, random_state):
    all_situations = set(dataset.postIdToId.keys())
    if isinstance(dataset, SocialNormDataset):
        annotated_situations = json.load(
            open('../data/conflict_aspect_annotations.json', 'r'))
        annotated_situations = set(annotated_situations['data'].keys())
        all_situations = list(all_situations.difference(annotated_situations))
    else:
        all_situations = list(all_situations)

    train_situations, test_situations = train_test_split(
        all_situations, test_size=0.18, random_state=random_state)
    train_situations, val_situations = train_test_split(
        train_situations, test_size=0.15, random_state=random_state)
    
    if isinstance(dataset, SocialNormDataset):
        test_situations.extend(list(annotated_situations))

    postToVerdicts = ListDict()
    for v, s in dataset.verdictToParent.items():
        # if dataset.verdictToTokensLength[v] > 5:
        postToVerdicts.append(s, v)

    train_verdicts, train_labels = get_verdicts_labels_from_sit(
        dataset, train_situations, postToVerdicts)
    val_verdicts, val_labels = get_verdicts_labels_from_sit(
        dataset, val_situations, postToVerdicts)
    test_verdicts, test_labels = get_verdicts_labels_from_sit(
        dataset, test_situations, postToVerdicts)
    return train_verdicts, train_labels, val_verdicts, val_labels, test_verdicts, test_labels

def get_verdicts_by_situations_split_by_predefined_splits(dataset, train_parent_ids, val_parent_ids, test_parent_ids, annotator_ids, train_ds, random_state, author_to_id=False):
    all_situations = set(dataset.postIdToId.keys())
    
    if isinstance(dataset, SocialNormDataset):
        annotated_situations = json.load(
            open('../data/conflict_aspect_annotations.json', 'r'))
        annotated_situations = set(annotated_situations['data'].keys())
        all_situations = list(all_situations.difference(annotated_situations))
    else:
        all_situations = list(all_situations)
    
    #postIdToParent = {pId: dataset.idToParent[dataset.postIdToId[pId]] for pId in all_situations}
    
    train_situations = [sit for sit  in all_situations if sit in train_parent_ids]
    val_situations = [sit for sit  in all_situations if sit in val_parent_ids]
    test_situations = [sit for sit  in all_situations if sit in test_parent_ids]
    
    if isinstance(dataset, SocialNormDataset):
        test_situations.extend(list(annotated_situations))

    postToVerdicts = ListDict()
    for v, s in dataset.verdictToParent.items():
        # if dataset.verdictToTokensLength[v] > 5:
        postToVerdicts.append(s, v)

    train_verdicts, train_labels = get_verdicts_labels_from_sit(
        dataset, train_situations, postToVerdicts)
    val_verdicts, val_labels = get_verdicts_labels_from_sit(
        dataset, val_situations, postToVerdicts)
    test_verdicts, test_labels = get_verdicts_labels_from_sit(
        dataset, test_situations, postToVerdicts)
    
    if author_to_id:
        train_ds.index = train_ds.parent_id 
    
    train_verdicts_new, train_labels_new = [],[]
    for i, verdict in enumerate(train_verdicts):
        author_id = dataset.verdictToAuthor[verdict]
        if author_to_id:
            author_id = author_to_id[author_id]
        post_id = dataset.verdictToParent[verdict]
        # if author_id in annotator_ids and post_id in train_ds.index and not pd.isna(train_ds.loc[post_id,str(author_id)]):
        if author_id in annotator_ids:
            if post_id in train_parent_ids:
                if not pd.isna(train_ds.loc[post_id,str(author_id)]):
                    train_verdicts_new.append(verdict)
                    train_labels_new.append(train_labels[i])

    return train_verdicts_new, train_labels_new, val_verdicts, val_labels, test_verdicts, test_labels


def get_verdicts_by_author_split(dataset, random_state):
    if not os.path.exists('../data/splits/train_author.txt'):
        all_authors = list(dataset.authorsToVerdicts.keys())
        train_authors, test_authors = train_test_split(
            all_authors, test_size=0.2, random_state=random_state)
        train_authors, val_authors = train_test_split(
            train_authors, test_size=0.14, random_state=random_state)
        write_splits('../data/splits/train_author.txt', train_authors)
        write_splits('../data/splits/val_author.txt', val_authors)
        write_splits('../data/splits/test_author.txt', test_authors)
    else:
        print("Reading authors splits.")
        train_authors = read_splits('../data/splits/train_author.txt')
        val_authors = read_splits('../data/splits/val_author.txt')
        test_authors = read_splits('../data/splits/test_author.txt')
        # train_authors.remove('Judgement_Bot_AITA')

    train_verdicts, train_labels = get_verdicts_labels_from_authors(
        dataset, train_authors)
    val_verdicts, val_labels = get_verdicts_labels_from_authors(
        dataset, val_authors)
    test_verdicts, test_labels = get_verdicts_labels_from_authors(
        dataset, test_authors)
    return train_verdicts, train_labels, val_verdicts, val_labels, test_verdicts, test_labels


def evaluate(dataloader, model, graph_model, data, embedder, USE_AUTHORS, dataset, author_encoder, samples_per_class_train, loss_type, return_predictions=False):

    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")

    model.eval()
    if USE_AUTHORS and author_encoder == 'graph':
        graph_model.eval()

    all_ids = ['verdicts']
    all_pred = ['predictions']
    all_labels = ['gold labels']
    all_post_ids = ['post_ids']
    all_authors = ['authors']
    
    losses = []
    
    start_time = time.perf_counter()

    for batch in dataloader:
        verdicts_index = batch.pop("index")
        author_node_idx = batch.pop("author_node_idx")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.no_grad():
            if USE_AUTHORS and (author_encoder == 'average' or author_encoder == 'attribution'):
                authors_embeddings = torch.stack([embedder.embed_author(
                    dataset.verdictToAuthor[dataset.idToVerdict[index.item()]]) for index in verdicts_index]).to(DEVICE)
                logits = model(batch, authors_embeddings)
            elif USE_AUTHORS and author_encoder == 'graph':
                graph_output = graph_model(
                    data.x.to(DEVICE), data.edge_index.to(DEVICE))
                authors_embeddings = graph_output[author_node_idx.to(DEVICE)]
                logits = model(batch, authors_embeddings)
            else:
                logits = model(batch)
                
        loss = loss_fn(logits, labels, samples_per_class_train, loss_type=loss_type)
        losses.append(loss.item())

        predictions = torch.argmax(logits, dim=-1)
        accuracy_metric.add_batch(predictions=predictions, references=labels)
        f1_metric.add_batch(predictions=predictions, references=labels)
        all_pred.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_ids.extend([dataset.idToVerdict[idx]
                       for idx in verdicts_index.numpy()])
        all_post_ids.extend([dataset.idToParent[idx]
                       for idx in verdicts_index.numpy()])
        all_authors.extend([dataset.verdictToAuthor[dataset.idToVerdict[idx]]
                       for idx in verdicts_index.numpy()])

    end_time = time.perf_counter()
    run_time = (end_time - start_time) * 1000  # milliseconds
    
    loss_mean = float(torch.tensor(losses).mean())
    
    # calc majority F1s
    all_post_ids_np = np.array(all_post_ids[1:])
    all_pred_np = np.array(all_pred[1:])
    all_labels_np = np.array(all_labels[1:])
    all_post_ids_set_list = list(set(all_post_ids_np))
    all_post_ids_occurence_idx_lists = [[i for i,e in enumerate(all_post_ids_np) if e == set_e] for set_e in all_post_ids_set_list]
    majority_preds = []
    majority_labels = []

    for i in range(len(all_post_ids_occurence_idx_lists)):
        idxs = all_post_ids_occurence_idx_lists[i]
        preds = all_pred_np[idxs]
        labels = all_labels_np[idxs]
        majority_pred = 0 if preds.mean()<0.5 else 1
        majority_label = 0 if labels.mean()<0.5 else 1
        majority_preds.append(majority_pred)
        majority_labels.append(majority_label)

    if return_predictions:
        jsonable_all_preds = [all_pred[0]] + [int(x) for x in all_pred[1:]]
        jsonable_all_labels = [all_labels[0]] + [int(x) for x in all_labels[1:]]
        return {'accuracy': accuracy_metric.compute()['accuracy'],
                'f1_weighted': f1_metric.compute(average='weighted')['f1'],
                'macro': f1_score(all_labels[1:], all_pred[1:], average='macro'),
                'micro': f1_score(all_labels[1:], all_pred[1:], average='micro'),
                'binary': f1_score(all_labels[1:], all_pred[1:], average='binary'),
                'accuracy_majority': accuracy_score(majority_labels, majority_preds),
                'macro_majority': f1_score(majority_labels, majority_preds, average='macro'),
                'binary_majority': f1_score(majority_labels, majority_preds, average='binary'),
                'results': list(zip(all_ids, all_authors, jsonable_all_preds, jsonable_all_labels)),
                'run_time_millis': run_time,
                'loss_mean': loss_mean}

    return {'accuracy': accuracy_metric.compute()['accuracy'],
            'f1_weighted': f1_metric.compute(average='weighted')['f1'],
            'macro': f1_score(all_labels[1:], all_pred[1:], average='macro'),
            'micro': f1_score(all_labels[1:], all_pred[1:], average='micro'),
            'binary': f1_score(all_labels[1:], all_pred[1:], average='binary'),
            'accuracy_majority': accuracy_score(majority_labels, majority_preds),
            'macro_majority': f1_score(majority_labels, majority_preds, average='macro'),
            'binary_majority': f1_score(majority_labels, majority_preds, average='binary'),
            'run_time_millis': run_time,
            'loss_mean': loss_mean}


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def create_author_graph(graphData, dataset, authors_embeddings, authorToAuthor, limit_connections=100):
    leave_out = {'Judgement_Bot_AITA'}
    for author, _ in dataset.authorsToVerdicts.items():
        if author not in leave_out:
            graphData.addNode(author, 'author',
                              authors_embeddings[author], None, None)

    # Add author to author edges
    source = []
    target = []
    for author, neighbors in tqdm(authorToAuthor.items()):
        neighbors.sort(key=lambda x: x[1], reverse=True)
        if len(neighbors) > limit_connections:
            neighbors = neighbors[:limit_connections]

        for neighbor in neighbors:
            # neighbor[0] = author, neighbor[1] = number_of_connections
            if author in graphData.nodesToId and neighbor[0] in graphData.nodesToId:
                source.append(graphData.nodesToId[author])
                target.append(graphData.nodesToId[neighbor[0]])

    return graphData, torch.tensor([source, target], dtype=torch.long)
