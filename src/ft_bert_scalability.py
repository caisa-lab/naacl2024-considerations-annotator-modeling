import glob
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizerFast, DataCollatorWithPadding, AdamW, get_scheduler
from datasets import DatasetDict, Dataset, Features, Value
from torch_geometric.data import Data
from torch.utils.data import DataLoader, SubsetRandomSampler


import pickle

from utils.train_utils import calc_annotator_class_weights, create_composite_embeddings

from joan_utils.dataset import GraphData, SocialNormDataset, VerdictDataset, RegularDataset
from joan_utils.read_files import *
from joan_utils.utils import *
from joan_utils.loss_functions import *
from joan_utils.train_utils import *
from joan_utils.models import GAT, JudgeBert, SentBertClassifier, SentBertClassifier_Anchor_v4, SentBertClassifier_Anchor_v2
from joan_utils.constants import *
from tqdm import tqdm
from argparse import ArgumentParser
import logging
from datetime import datetime
import time

from utils.train_utils import EarlyStopper

from joan_utils.train_utils import filter_df_min_annotation_and_update_majority, reduce_df_to_min_num_annos

# TIMESTAMP = get_current_timestamp()

# '../data/embeddings/emnlp/sbert_authorAMIT.pkl'


############ run ############

# CUDA_VISIBLE_DEVICES=3 /app/home/neuendob/anaconda3/envs/master/bin/

##########  NORMAL ###########

# Authors averaging
# python ft_bert_scalability.py --use_authors='true' --author_encoder='average' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --user_dim=768 --model_name='sbert' --split_type='sit' --authors_embedding_path='../data/embeddings/emnlp/sbert_authorAMIT.pkl' --results_dir='../results/scalability' --dataset_name='social_norms'

# Authors attribution
# python ft_bert_scalability.py --use_authors='true' --author_encoder='attribution' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --user_dim=13484 --model_name='sbert' --split_type='sit' --authors_embedding_path='../data/embeddings/emnlp/attribution/sit_mlp_prediction.pkl' --results_dir='../results/scalability' --dataset_name='social_norms'

# ID:
# python ft_bert_scalability.py --use_authors='false' --author_encoder='user_id' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --model_name='sbert' --split_type='sit' --authors_embedding_path='Not_needed' --results_dir='../results/scalability' --dataset_name='social_norms'

# baseline for sc:
# python ft_bert_scalability.py --use_authors='false' --author_encoder='none' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --model_name='sbert' --split_type='sit' --authors_embedding_path='Not_needed' --results_dir='../results/scalability' --dataset_name='social_norms'

# ID for GE:
# python ft_bert_scalability.py --use_authors='false' --author_encoder='user_id' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --model_name='sbert' --split_type='sit' --authors_embedding_path='Not_needed' --results_dir='../results/scalability' --dataset_name='GE' --situation='text'

# ID for GHC:
# python ft_bert_scalability.py --use_authors='false' --author_encoder='user_id' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --model_name='sbert' --split_type='sit' --authors_embedding_path='Not_needed' --results_dir='../results/scalability' --dataset_name='GHC' --situation='text'

# baseline for GHC:
# python ft_bert_scalability.py --use_authors='false' --author_encoder='none' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --model_name='sbert' --split_type='sit' --authors_embedding_path='Not_needed' --results_dir='../results/scalability' --dataset_name='GHC' --situation='text'

# baseline for GE:
# python ft_bert_scalability.py --use_authors='false' --author_encoder='none' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --model_name='sbert' --split_type='sit' --authors_embedding_path='Not_needed' --results_dir='../results/scalability' --dataset_name='GE' --situation='text'


########### WITH PERCENTAGE ########

# Authors averaging
# python ft_bert_scalability.py --use_authors='true'  --author_encoder='average' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --user_dim=768 --model_name='sbert' --split_type='sit' --authors_embedding_path='../data/embeddings/emnlp/sbert_authorAMIT.pkl' --results_dir='../results/comment_scalability' --dataset_name='social_norms' --comment_scalability=True

# Authors attribution
# python ft_bert_scalability.py --use_authors='true'  --author_encoder='attribution' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --user_dim=13484 --model_name='sbert' --split_type='sit' --authors_embedding_path='../data/embeddings/emnlp/attribution/sit_mlp_prediction.pkl' --results_dir='../results/comment_scalability' --dataset_name='social_norms' --comment_scalability=True

# ID:
# python ft_bert_scalability.py --use_authors='false' --author_encoder='user_id' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --model_name='sbert' --split_type='sit' --authors_embedding_path='Not_needed' --results_dir='../results/comment_scalability' --dataset_name='social_norms' --comment_scalability=True

# baseline for sc:
# python ft_bert_scalability.py --use_authors='false' --author_encoder='none' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --model_name='sbert' --split_type='sit' --authors_embedding_path='Not_needed' --results_dir='../results/comment_scalability' --dataset_name='social_norms' --comment_scalability=True

# ID for GE:
# python ft_bert_scalability.py --use_authors='false' --author_encoder='user_id' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --model_name='sbert' --split_type='sit' --authors_embedding_path='Not_needed' --results_dir='../results/comment_scalability' --dataset_name='GE' --situation='text' --comment_scalability=True

# ID for GHC:
# python ft_bert_scalability.py --use_authors='false' --author_encoder='user_id' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --model_name='sbert' --split_type='sit' --authors_embedding_path='Not_needed' --results_dir='../results/comment_scalability' --dataset_name='GHC' --situation='text' --comment_scalability=True

# baseline for GHC:
# python ft_bert_scalability.py --use_authors='false' --author_encoder='none' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --model_name='sbert' --split_type='sit' --authors_embedding_path='Not_needed' --results_dir='../results/comment_scalability' --dataset_name='GHC' --situation='text' --comment_scalability=True

# baseline for GE:
# python ft_bert_scalability.py --use_authors='false' --author_encoder='none' --loss_type='focal' --num_epochs=10 --sbert_model='sentence-transformers/all-distilroberta-v1' --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --model_name='sbert' --split_type='sit' --authors_embedding_path='Not_needed' --results_dir='../results/comment_scalability' --dataset_name='GE' --situation='text' --comment_scalability=True

parser = ArgumentParser()
parser.add_argument("--use_authors", dest="use_authors", required=True, type=str2bool)
parser.add_argument("--author_encoder", dest="author_encoder", required=True, type=str) # ['average', 'priming', 'graph', 'none']

parser.add_argument("--split_type", dest="split_type", required=True, type=str) # ['author', 'sit', 'verdicts']
parser.add_argument("--situation", dest="situation", default='title', type=str) # ['text', 'title']
parser.add_argument("--sbert_model", dest="sbert_model", default='sentence-transformers/all-distilroberta-v1', type=str)
parser.add_argument("--authors_embedding_path", dest="authors_embedding_path", required=True, type=str)
parser.add_argument("--sbert_dim", dest="sbert_dim", default=768, type=int)
parser.add_argument("--user_dim", dest="user_dim", default=768, type=int)
parser.add_argument("--graph_dim", dest="graph_dim", default=384, type=int)
parser.add_argument("--concat", dest="concat", default='true', type=str2bool)
parser.add_argument("--num_epochs", dest="num_epochs", default=5, type=int)
parser.add_argument("--learning_rate", dest="learning_rate", default=1e-4, type=float)
parser.add_argument("--batch_size", dest="batch_size", default=32, type=int)
parser.add_argument("--loss_type", dest="loss_type", default='softmax', type=str)
parser.add_argument("--verdicts_dir", dest="verdicts_dir", default='../data/verdicts', type=str)
parser.add_argument("--bert_tok", dest="bert_tok", default='bert-base-uncased', type=str)
parser.add_argument("--dirname", dest="dirname", type=str, default='../data/amit_filtered_history')
parser.add_argument("--model_name", dest="model_name", type=str, required=True) # ['judge_bert', 'sbert'] otherwise exception
parser.add_argument("--dataset_name", dest="dataset_name", type=str, required=True) # ['social_norms', "GE'anger",..., 'GHC']
parser.add_argument("--comment_scalability", dest="comment_scalability", type=str2bool, default=False)
parser.add_argument("--seed_i", dest="seed_i", type=int, default=False)
parser.add_argument("--run_full_SC_ds_to_test_implementation", dest="run_full_SC_ds_to_test_implementation", type=str2bool, default=False)
parser.add_argument("--run_full_GE_GHC_ds_to_test_implementation", dest="run_full_GE_GHC_ds_to_test_implementation", type=str2bool, default=False)
parser.add_argument("--use_full", dest="use_full", type=str2bool, default=False)

parser.add_argument("--num_anchors", dest="num_anchors", type=int, default=0)
parser.add_argument("--num_anchors_neighbors", dest="num_anchors_neighbors", type=int, default=0)
parser.add_argument("--anchor_version", dest="anchor_version", type=str, default='v2')
parser.add_argument("--num_annos", dest="num_annos", type=int, default=0)
parser.add_argument("--overwrite_existing_results", dest="overwrite_existing_results", type=str2bool, default=False)
parser.add_argument("--use_early_stopping", dest="use_early_stopping", type=str2bool, default=False)
parser.add_argument("--save_indi_preds", dest="save_indi_preds", type=str2bool, default=False)

if __name__ == '__main__':
    
    args = parser.parse_args()
    print_args(args, logging)
    dirname = args.dirname
    bert_checkpoint = args.bert_tok
    model_name = args.model_name
    dataset_name = args.dataset_name
    verdicts_dir = args.verdicts_dir
    graph_dim = args.graph_dim
    authors_embedding_path = args.authors_embedding_path
    USE_AUTHORS = args.use_authors
    author_encoder = args.author_encoder
    
    comment_scalability = args.comment_scalability
    seed_i = args.seed_i
    run_full_SC_ds_to_test_implementation = args.run_full_SC_ds_to_test_implementation
    run_full_GE_GHC_ds_to_test_implementation = args.run_full_GE_GHC_ds_to_test_implementation
    user_dim = args.user_dim
    
    num_anchors = args.num_anchors
    num_anchors_neighbors = args.num_anchors_neighbors
    overwrite_existing_results = args.overwrite_existing_results
    anchor_version = args.anchor_version
    
    use_early_stopping = args.use_early_stopping
    
    save_indi_preds = args.save_indi_preds
    
    use_full = args.use_full
    full_str = ''
    full_path_str = ''
    if use_full:
        full_str = '*full'
        full_path_str = 'ge_ghc_full_ds_scalability/'
        
    assert (not use_full) or ('GE' in dataset_name or 'GHC' in dataset_name), 'SC is full ds each time anyway'
    
    if USE_AUTHORS:
        assert author_encoder in {'average', 'graph', 'attribution', 'composite', 'compositeUid'}
    else:
        assert author_encoder.lower() in ['none', 'priming', 'user_id', 'composite', 'compositeuid']
    
    assert dataset_name in ['ArMIS', 'ConvAbuse' , 'HSBrexit', 'MD','social_norms', 'GHC', 'GE', "GE'anger", "GE'disgust", "GE'fear", "GE'joy", "GE'sadness", "GE'surprise", "SC"]
    if dataset_name in ['social_norms','SC']:
        dataset_name = 'social_norms'
        ds_name_short = 'SC'
    else:
        ds_name_short = dataset_name

    if author_encoder == 'user_id':
        model_name_short = 'uid'
    elif author_encoder == 'average':
        model_name_short = 'ae'
    elif author_encoder == 'none':
        model_name_short = 'sbertbase'
    elif author_encoder == 'attribution':
        model_name_short = 'aa'
    elif author_encoder == 'composite':
        model_name_short = 'comp'
    elif author_encoder == 'compositeUid':
        model_name_short = 'compUid'
    
    split_type = args.split_type
    
    comment_num_annos_list = [14,50]
    num_annos_list = []
    comments_percentage_list = [False]
    
    if args.num_annos == 0:
        num_annos_GE = [6, 8, 10, 12, 14, 16, 18] + list(range(22, 83, 4))
        num_annos_MD = [6, 8, 10, 12, 14, 16, 18] + list(range(22, 83, 4)) + list(range(100,601,100)) + [682]
        num_annos_SC = [6, 8, 10, 12, 14, 16, 18] + list(range(22, 83, 4)) + list(range(100, 1000, 100)) + list(range(1000, 2510, 300))
        num_annos_GHC = [6, 8, 10, 12, 14, 16, 18]
        num_annos_ConvAbuse = [1,2,3,4,5,6,7,8]
        num_annos_ArMIS = [1,2,3]
        num_annos_HSBrexit = [1,2,3,4,5,6]
    else:
        num_annos_GE = [args.num_annos]
        num_annos_MD = [args.num_annos]
        num_annos_SC = [args.num_annos]
        num_annos_GHC = [args.num_annos]
        num_annos_ConvAbuse = [args.num_annos]
        num_annos_ArMIS = [args.num_annos]
        num_annos_HSBrexit = [args.num_annos]
    
    GE_comments_percentage_list = [float(f'0.0{x}') for x in range(2,10,2)] + [float(f'0.{x}') for x in range(1,10)] + [1.0]
    GHC_comments_percentage_list = [float(f'0.0{x}') for x in range(2,10,2)] + [float(f'0.{x}') for x in range(1,10)] + [1.0]
    SC_comments_percentage_list = [float(f'0.{x}') for x in range(1,10)] + [1.0]
    funkes_ds_comments_percentage_list = SC_comments_percentage_list
    
    learning_rate = None
    emotion = None
    
    if 'GE' in args.dataset_name:
        num_annos_list = num_annos_GE
        if comment_scalability:
            comments_percentage_list = GE_comments_percentage_list
            if args.num_annos == 0:
                num_annos_list = comment_num_annos_list
        learning_rate = 5e-6
        if not "'" in args.dataset_name:
            raise Exception('missing Emotion')
        emotion = args.dataset_name.split("'")[-1]
        # dataset_name = 'GE'
    elif args.dataset_name == 'GHC':
        num_annos_list = num_annos_GHC
        if comment_scalability:
            comments_percentage_list = GHC_comments_percentage_list  
            if args.num_annos == 0:
                num_annos_list = comment_num_annos_list[:1]
        learning_rate = 1e-5 
        emotion = 'not_needed'
        dataset_name = 'GHC'
    elif args.dataset_name == 'SC':
        num_annos_list = num_annos_SC
        if comment_scalability:
            comments_percentage_list = SC_comments_percentage_list 
            if args.num_annos == 0:
                num_annos_list = comment_num_annos_list
        learning_rate = 2e-5 
        emotion = 'not_needed'
        dataset_name = 'SC'
    elif args.dataset_name == 'MD':
        num_annos_list = num_annos_MD
        if comment_scalability:
            comments_percentage_list = funkes_ds_comments_percentage_list 
            if args.num_annos == 0:
                num_annos_list = comment_num_annos_list
        learning_rate = 3e-5 
        emotion = 'not_needed'
        dataset_name = 'MD'
    elif args.dataset_name == 'HSBrexit':
        num_annos_list = num_annos_HSBrexit
        if comment_scalability:
            comments_percentage_list = funkes_ds_comments_percentage_list 
        learning_rate = 5e-5 
    elif args.dataset_name == 'ConvAbuse':
        num_annos_list = num_annos_ConvAbuse
        if comment_scalability:
            comments_percentage_list = funkes_ds_comments_percentage_list 
        learning_rate = 5e-5 
    elif args.dataset_name == 'ArMIS':
        num_annos_list = num_annos_ArMIS
        if comment_scalability:
            comments_percentage_list = funkes_ds_comments_percentage_list 
        learning_rate = 1e-5 
    else:
        raise Exception('wrong ds name: ', args.dataset_name)

    assert type(seed_i) == int
    
    if not type(seed_i) == int:
        seed_idx_list = [0,1,2,3,4]
    else:
        seed_idx_list = [seed_i]
    
    
    if run_full_SC_ds_to_test_implementation: #compare performance with joans
        seed_idx_list = [0]
        num_annos_list = ['ALL']
        # comments_percentage_list = [False]
        # comment_scalability = False
    elif run_full_GE_GHC_ds_to_test_implementation:
        seed_idx_list = [0,1,2,3,4]
        if 'GE' in dataset_name:
            num_annos_list = [50, 82]
        if dataset_name == 'GHC':
            num_annos_list = [18]
        # comments_percentage_list = [True]
        # comment_scalability = False
        
    for seed_idx in seed_idx_list:
        seed = SEEDS[seed_idx]
        
        for num_annos in num_annos_list:
            
            for comments_percentage in comments_percentage_list:
                
                if author_encoder == 'attribution':
                    user_dim = num_annos
                    if comment_scalability:
                        authors_embedding_path = f'../data/embeddings/emnlp/attribution/sit_mlp_prediction_{num_annos}_annos_{int(comments_percentage*100)}_percentage.pkl'
                    elif num_annos in [14,50]:
                        authors_embedding_path = f'../data/embeddings/emnlp/attribution/sit_mlp_prediction_{num_annos}_annos_{100}_percentage.pkl'
                    else:
                        authors_embedding_path = f'../data/embeddings/emnlp/attribution/sit_mlp_prediction_{num_annos}_annos.pkl'

                anchor_str = ''
                if num_anchors > 0:
                    anchor_str = f'-aa-anchor-{anchor_version}-{num_anchors}-{num_anchors_neighbors}'
                
                if comment_scalability:
                    path_to_save = f'../results/perso_comment_scalability_{seed_idx}/{ds_name_short}-{model_name_short}{anchor_str}_{int(comments_percentage*100)}_percentage_{num_annos}_annos/'
                else:
                    path_to_save = f'../results/{full_path_str}perso_scalability_{seed_idx}/{ds_name_short}-{model_name_short}{anchor_str}_{num_annos}_annos/'
                
                parent_foler_path = os.path.join(*path_to_save.strip('/').split('/')[:-1])
                if not os.path.isdir(parent_foler_path):
                    os.mkdir(parent_foler_path)
                
                if os.path.exists(path_to_save+"model_results.json") and not overwrite_existing_results:
                    print("FILE ALREADY EXISTS: ", path_to_save+"model_results.json")
                    continue
                
                if not os.path.exists(path_to_save):
                    os.mkdir(path_to_save)
                else:
                    if os.path.exists(path_to_save + 'model_results.json') and not overwrite_existing_results:
                        raise FileExistsError(path_to_save + 'model_results.json')
                
                dt_string = datetime.now().strftime("%m-%d_%Hh%M")
                # set up logging
                '/home/neuendo4/Master/Master_Thesis/results/ge_ghc_full_ds_scalability/perso_comment_scalability_3/sc-aa_14_annos/model_run_at_09-08_11h33.log'
                log_file = f"{path_to_save}model_run_at_{dt_string}.log"
                # create file handler and set the formatter
                file_handler = logging.FileHandler(log_file)
                # add handler to the logger
                logger = logging.getLogger(f'logger_{ds_name_short}_{model_name_short}_{num_annos}_annos')
                logger.setLevel(logging.DEBUG)
                logger.addHandler(file_handler)
                
                print('#'*100)
                if comment_scalability:
                    print_string = f"RUNNING with {num_annos} Annotators and {comments_percentage} comment percentage:"
                else:
                    print_string = f"RUNNING with {num_annos} Annotators:"
                print(print_string)
                logging.info(print_string)
                print('#'*100)                

                TIMESTAMP = get_current_timestamp()
                checkpoint_dir = os.path.join(path_to_save, f'{TIMESTAMP}_best_model_sampled.pt')
                graph_checkpoint_dir = os.path.join(path_to_save, f'{TIMESTAMP}_best_graphmodel.pt')

                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                    
                
                logger.info("Device {}".format(DEVICE))
                assert DEVICE == torch.device('cuda')
                
                
                # load and process data
                if dataset_name in ['social_norms', 'SC']:
                    path_to_data = '../data/'
                    social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts_and_authors.gzip', compression='gzip')
                    social_comments = pd.read_csv(path_to_data+'social_norms_clean.csv')
                    
                    anno_ids = []
                    if not run_full_SC_ds_to_test_implementation:
                        sc_traversed_filtered = pd.read_csv(f'../data/sc_filtered_from_bela/sc_{num_annos}_annos_filtered.csv')
                        
                        dataframe  = sc_traversed_filtered
                        dataframe['id'] = dataframe.index 
                        ds_name = 'sc'
                        train_all_ds = pd.read_csv(f'../data/{ds_name.upper()}_data/train_{ds_name}.csv')
                        val_all_ds = pd.read_csv(f'../data/{ds_name.upper()}_data/val_{ds_name}.csv')
                        test_all_ds = pd.read_csv(f'../data/{ds_name.upper()}_data/test_{ds_name}.csv')
                        
                        train_ds = dataframe[dataframe.id.isin(train_all_ds.id)]
                        val_ds = dataframe[dataframe.id.isin(val_all_ds.id)]
                        test_ds = dataframe[dataframe.id.isin(test_all_ds.id)]
                        
                        if comment_scalability:
                            train_ds = reduce_df_to_min_num_annos(train_ds, comments_percentage)
                            # val_ds = reduce_df_to_min_num_annos(val_ds, comments_percentage)
                            # test_ds = reduce_df_to_min_num_annos(test_ds, comments_percentage)
                        
                        train_parent_ids = list(train_ds.parent_id)
                        val_parent_ids = list(val_ds.parent_id)
                        test_parent_ids = list(test_ds.parent_id)
                        
                        num_comments_per_anno_train = min(train_ds.count())
                
                        dataframe = dataframe.replace(float('nan'), -1)
                        dataframe = filter_df_min_annotation_and_update_majority(dataframe, 1)
                        
                        sc_traversed_filtered = dataframe
                        
                        anno_ids = [int(x) for x in sc_traversed_filtered.columns if re.fullmatch(r'[0-9]*',x)]

                        with open(f'{path_to_data}sc_filtered_from_bela/author_mapping.json', 'r') as f:
                            author_to_id = json.load(f)
                            
                        id_to_author_name = list(author_to_id.keys())
                        authors_filtered = [id_to_author_name[i] for i in anno_ids]

                        social_comments = social_comments[social_comments.author_name.isin(authors_filtered)]
                        
                        social_chemistry = social_chemistry[social_chemistry.post_id.isin(sc_traversed_filtered.parent_id)]
                        social_comments = social_comments[social_comments.parent_id.isin(sc_traversed_filtered.parent_id)]
                        
                        social_comments = social_comments[social_comments.parent_id.isin(social_chemistry.post_id)]
                        social_chemistry = social_chemistry[social_chemistry.post_id.isin(social_comments.parent_id)]
                        
                    dataset = SocialNormDataset(social_comments, social_chemistry)
                    annotator_ids = anno_ids
                    
                elif dataset_name in ['ArMIS', 'ConvAbuse' , 'HSBrexit', 'MD', 'GHC', 'GE', "GE'anger", "GE'disgust", "GE'fear", "GE'joy", "GE'sadness", "GE'surprise"]:
                    
                    dataframe = pd.read_csv(f'../data/matching_sizes_data/{dataset_name}{full_str}_{num_annos}_annos_filtered.csv')
                    dataframe['id'] = dataframe.index 
                    
                    ############################################
                    print('1', dataframe.shape)
                    
                    if dataset_name in ['ArMIS', 'ConvAbuse' , 'HSBrexit', 'MD', 'GHC']:
                        train_all_ds = pd.read_csv(f'../data/{dataset_name}_data/train_{dataset_name.lower()}.csv')
                        val_all_ds = pd.read_csv(f'../data/{dataset_name}_data/val_{dataset_name.lower()}.csv')
                        test_all_ds = pd.read_csv(f'../data/{dataset_name}_data/test_{dataset_name.lower()}.csv')
                    else:
                        emotion = dataset_name.split("'")[-1]
                        ds_name = dataset_name.split("'")[0]
                        train_all_ds = pd.read_csv(f'../data/{ds_name}_data/{emotion}_train_ge.csv')
                        val_all_ds = pd.read_csv(f'../data/{ds_name}_data/{emotion}_val_ge.csv')
                        test_all_ds = pd.read_csv(f'../data/{ds_name}_data/{emotion}_test_ge.csv')
                    
                    ############################################
                    print('2', train_all_ds.shape)
                    
                    train_ds = dataframe[dataframe.id.isin(train_all_ds.id)]
                    val_ds = dataframe[dataframe.id.isin(val_all_ds.id)]
                    test_ds = dataframe[dataframe.id.isin(test_all_ds.id)]
                    
                    ############################################
                    print('3', train_ds.shape)
                    
                    if comment_scalability:
                        train_ds = reduce_df_to_min_num_annos(train_ds, comments_percentage)
                        # val_ds = reduce_df_to_min_num_annos(val_ds, comments_percentage)
                        # test_ds = reduce_df_to_min_num_annos(test_ds, comments_percentage)
                    
                    train_parent_ids = list(train_ds.id)
                    val_parent_ids = list(val_ds.id)
                    test_parent_ids = list(test_ds.id)
                    
                    num_comments_per_anno_train = min(train_ds.count())

                    dataframe = pd.concat([train_ds, val_ds, test_ds])
                    dataframe = dataframe.replace(float('nan'), -1)
                    dataframe = filter_df_min_annotation_and_update_majority(dataframe, 1)
                        
                    dataset = RegularDataset(dataframe)
                    annotator_ids = [
                        x for x in dataframe.columns if re.fullmatch(r'[0-9]+', x)]
                    
                else:
                    raise Exception('Wrong dataset name.')
                
                print('#'*100)
                print('datafrmae, dataset:')
                print(dataframe.shape)
                print(dataset.number_of_posts)
                
                
                #####################
                # calculations for anchors
                
                if num_anchors > 0:
                    weights = calc_annotator_class_weights(dataframe)
                    
                    if comment_scalability:
                        authors_embedding_path = f'../data/embeddings/emnlp/attribution/sit_mlp_prediction_{14}_annos_{int(comments_percentage*100)}_percentage.pkl'
                    elif num_anchors in [14,50]:
                        authors_embedding_path = f'../data/embeddings/emnlp/attribution/sit_mlp_prediction_{num_anchors}_annos_{100}_percentage.pkl'
                    else:
                        authors_embedding_path = f'../data/embeddings/emnlp/attribution/sit_mlp_prediction_{num_anchors}_annos.pkl'

                    user_dim_for_anchors = num_anchors
                    anchor_embedder = AuthorsEmbedder(embeddings_path=authors_embedding_path, dim=user_dim_for_anchors)
                    
                    auth_ids_to_idx = {id:idx for idx, id in enumerate(annotator_ids)}
                    
                    annotator_ids_as_str = [x for x in dataframe.columns if re.fullmatch(r'[0-9]+', x)]
                    
                    anno_counts = train_ds[annotator_ids_as_str].count()
                    anchor_ids = list(anno_counts.nlargest(num_anchors).index)
                    anchor_indices = [annotator_ids_as_str.index(a) for a in anchor_ids]
                    
                    anchor_names = [id_to_author_name[int(i)] for i  in anchor_ids]

                    author_embeddings = [anchor_embedder.embed_author(a) for a in annotator_ids_as_str]
                    author_distances = [[1- torch.nn.functional.cosine_similarity(a,b, dim=0) if i!=j and (i in anchor_indices) else torch.tensor(np.inf) for i,b in enumerate(author_embeddings)] for j, a in enumerate(author_embeddings)]

                    author_idx_to_clostest_anchor_idxs = [np.argsort(np.array(ds))[:num_anchors_neighbors] for ds in author_distances]
                    author_idx_to_clostest_anchor_idxs = [list(x) for x in author_idx_to_clostest_anchor_idxs]
                    
                    author_idx_to_clostest_anchor_ids = [[int(annotator_ids_as_str[idx]) for idx in idxs_list] for idxs_list in author_idx_to_clostest_anchor_idxs]
                    
                    
                    for idx_list in author_idx_to_clostest_anchor_idxs:
                        assert all([x in anchor_indices for x in idx_list])
                        
                    author_idx_to_clostest_anchor_weights = []
                    for i in range(len(annotator_ids_as_str)):
                        author_to_anchors_dists = [d for idx, d in enumerate(author_distances[i]) if idx in author_idx_to_clostest_anchor_idxs[i][:num_anchors_neighbors]]
                        sum_dists = sum(author_to_anchors_dists)
                        weights = [d.item()/sum_dists for d in author_to_anchors_dists]
                        author_idx_to_clostest_anchor_weights.append(weights)
                
                #####################
                
                
                if split_type == 'sit':
                    logger.info("Split type {}".format(split_type))
                    if dataset_name in ['social_norms', 'SC']:
                        if run_full_SC_ds_to_test_implementation:
                            train_verdicts, train_labels, val_verdicts, val_labels, test_verdicts, test_labels = get_verdicts_by_situations_split(dataset, seed)
                        else:
                            train_verdicts, train_labels, val_verdicts, val_labels, test_verdicts, test_labels = get_verdicts_by_situations_split_by_predefined_splits(dataset, train_parent_ids, val_parent_ids, test_parent_ids, annotator_ids, train_ds, seed, author_to_id)
                    else:
                        train_verdicts, train_labels, val_verdicts, val_labels, test_verdicts, test_labels = get_verdicts_by_situations_split_by_predefined_splits(dataset, train_parent_ids, val_parent_ids, test_parent_ids, annotator_ids, train_ds, seed)
                else:
                    raise Exception("Split type is wrong, it should be either sit or author")   
            
                train_size_stats = "Training Size: {}, NTA labels {}, YTA labels {}".format(len(train_verdicts), train_labels.count(0), train_labels.count(1))
                logger.info(train_size_stats)
                val_size_stats = "Validation Size: {}, NTA labels {}, YTA labels {}".format(len(val_verdicts), val_labels.count(0), val_labels.count(1))
                logger.info(val_size_stats)
                test_size_stats = "Test Size: {}, NTA labels {}, YTA labels {}".format(len(test_verdicts), test_labels.count(0), test_labels.count(1))
                logger.info(test_size_stats)
                
                if USE_AUTHORS and (author_encoder == 'average' or author_encoder == 'attribution'):
                    embedder = AuthorsEmbedder(embeddings_path=authors_embedding_path, dim=args.user_dim)
                else:
                    embedder = None
                
                
                ## set raw data which is used to create dataset instances etc.
                raw_dataset = {'train': {'index': [], 'text': [], 'label': [], 'author_node_idx': []}, 
                        'val': {'index': [], 'text': [], 'label': [], 'author_node_idx': []}, 
                        'test': {'index': [], 'text': [], 'label': [], 'author_node_idx': []}}
                
                raw_dataset_anchor_data = {'train': {'index': [], 'anchor_inputs': [], 'weights': [], 'is_anchor_list': []}, 
                        'val': {'index': [], 'anchor_inputs': [], 'weights': [], 'is_anchor_list': []}, 
                        'test': {'index': [], 'anchor_inputs': [], 'weights': [], 'is_anchor_list': []}}
                
                authors_tokens_set = set()

                for i, verdict in enumerate(train_verdicts):
                    if args.situation == 'text':
                        situation_title = dataset.postIdToText[dataset.verdictToParent[verdict]]
                    else:
                        assert args.situation == 'title', print(args.situation)
                        situation_title = dataset.postIdToTitle[dataset.verdictToParent[verdict]]
                    
                    if situation_title != '' and situation_title is not None and verdict in dataset.verdictToAuthor:
                        author = dataset.verdictToAuthor[verdict]
                        
                        
                        if author != 'Judgement_Bot_AITA':
                            raw_dataset['train']['index'].append(dataset.verdictToId[verdict])
                            raw_dataset_anchor_data['train']['index'].append(dataset.verdictToId[verdict])
                            
                            if author_encoder == 'user_id' or author_encoder == 'compositeUid':
                                author = dataset.verdictToAuthor[verdict]
                                author_token = '[' +author +']'
                                authors_tokens_set.add(author_token)
                                raw_dataset['train']['text'].append(author_token + ' [SEP] ' + situation_title)
                                
                                #######
                                if num_anchors > 0:
                                    auth_id = author_to_id[author]
                                    auth_idx = auth_ids_to_idx[auth_id]
                                    closest_anchor_ids = author_idx_to_clostest_anchor_ids[auth_idx]
                                    closest_anchor_names = [id_to_author_name[i] for i in closest_anchor_ids]
                                    weights = author_idx_to_clostest_anchor_weights[auth_idx]
                                    
                                    anchor_inputs = []
                                    for anchor_name in closest_anchor_names:
                                        anchor_token = '[' +anchor_name +']'
                                        anchor_inputs.append(author_token + ' [SEP] ' + situation_title)
                                        
                                    raw_dataset_anchor_data['train']['anchor_inputs'].append(anchor_inputs)
                                    raw_dataset_anchor_data['train']['weights'].append(weights)
                                    raw_dataset_anchor_data['train']['is_anchor_list'].append(author in anchor_names)
                                #######
                                
                            # elif author_encoder == 'compositeUid':
                            #     author = dataset.verdictToAuthor[verdict]
                            #     author_token = '[' +author +']'
                            #     comp_token = '[COMP_' +author +']'
                            #     authors_tokens_set.add(author_token)
                            #     raw_dataset['train']['text'].append(author_token + comp_token + ' [SEP] ' + situation_title)
                            # elif author_encoder == 'composite':
                            #     author = dataset.verdictToAuthor[verdict]
                            #     comp_token = '[COMP_' +author +']'
                            #     raw_dataset['train']['text'].append(comp_token + ' [SEP] ' + situation_title)
                                
                            else:
                                raw_dataset['train']['text'].append(situation_title)
                                
                            raw_dataset['train']['label'].append(train_labels[i])
                            raw_dataset['train']['author_node_idx'].append(-1)
                                
                            assert train_labels[i] == dataset.verdictToLabel[verdict] 
                    
                for i, verdict in enumerate(val_verdicts):
                    if args.situation == 'text':
                        situation_title = dataset.postIdToText[dataset.verdictToParent[verdict]]
                    else:
                        assert args.situation == 'title', print(args.situation)
                        situation_title = dataset.postIdToTitle[dataset.verdictToParent[verdict]]
                        
                    if situation_title != '' and situation_title is not None and verdict in dataset.verdictToAuthor:
                        author = dataset.verdictToAuthor[verdict]
                        
                        if author != 'Judgement_Bot_AITA': 
                            raw_dataset['val']['index'].append(dataset.verdictToId[verdict])
                            raw_dataset_anchor_data['val']['index'].append(dataset.verdictToId[verdict])
                            if author_encoder == 'user_id':
                                author = dataset.verdictToAuthor[verdict]
                                author_token = '[' +author +']'
                                authors_tokens_set.add(author_token)
                                raw_dataset['val']['text'].append(author_token + ' [SEP] ' + situation_title)
                                
                                #######
                                if num_anchors > 0:
                                    auth_id = author_to_id[author]
                                    auth_idx = auth_ids_to_idx[auth_id]
                                    closest_anchor_ids = author_idx_to_clostest_anchor_ids[auth_idx]
                                    closest_anchor_names = [id_to_author_name[i] for i in closest_anchor_ids]
                                    weights = author_idx_to_clostest_anchor_weights[auth_idx]
                                    
                                    anchor_inputs = []
                                    for anchor_name in closest_anchor_names:
                                        anchor_token = '[' +anchor_name +']'
                                        anchor_inputs.append(author_token + ' [SEP] ' + situation_title)
                                        
                                    raw_dataset_anchor_data['val']['anchor_inputs'].append(anchor_inputs)
                                    raw_dataset_anchor_data['val']['weights'].append(weights)
                                    raw_dataset_anchor_data['val']['is_anchor_list'].append(author in anchor_names)
                                #######
                            elif author_encoder == 'compositeUid':
                                author = dataset.verdictToAuthor[verdict]
                                author_token = '[' +author +']'
                                comp_token = '[COMP_' +author +']'
                                authors_tokens_set.add(author_token)
                                raw_dataset['val']['text'].append(author_token + comp_token + ' [SEP] ' + situation_title)
                            elif author_encoder == 'composite':
                                author = dataset.verdictToAuthor[verdict]
                                comp_token = '[COMP_' +author +']'
                                raw_dataset['val']['text'].append(comp_token + ' [SEP] ' + situation_title)
                            else:
                                raw_dataset['val']['text'].append(situation_title)
                            
                            raw_dataset['val']['label'].append(val_labels[i])
                            raw_dataset['val']['author_node_idx'].append(-1)
                            
                            assert val_labels[i] == dataset.verdictToLabel[verdict]          
                    
                for i, verdict in enumerate(test_verdicts):
                    if args.situation == 'text':
                        situation_title = dataset.postIdToText[dataset.verdictToParent[verdict]]
                    else:
                        assert args.situation == 'title', print(args.situation)
                        situation_title = dataset.postIdToTitle[dataset.verdictToParent[verdict]]
                    
                    if situation_title != '' and situation_title is not None and verdict in dataset.verdictToAuthor:
                        author = dataset.verdictToAuthor[verdict]
                        
                        if author != 'Judgement_Bot_AITA': 
                            raw_dataset['test']['index'].append(dataset.verdictToId[verdict])
                            raw_dataset_anchor_data['test']['index'].append(dataset.verdictToId[verdict])
                            if author_encoder == 'user_id':
                                author = dataset.verdictToAuthor[verdict]
                                author_token = '[' +author +']'
                                authors_tokens_set.add(author_token)
                                raw_dataset['test']['text'].append(author_token + ' [SEP] ' + situation_title)
                                
                                #######
                                if num_anchors > 0:
                                    auth_id = author_to_id[author]
                                    auth_idx = auth_ids_to_idx[auth_id]
                                    closest_anchor_ids = author_idx_to_clostest_anchor_ids[auth_idx]
                                    closest_anchor_names = [id_to_author_name[i] for i in closest_anchor_ids]
                                    weights = author_idx_to_clostest_anchor_weights[auth_idx]
                                    
                                    anchor_inputs = []
                                    for anchor_name in closest_anchor_names:
                                        anchor_token = '[' +anchor_name +']'
                                        anchor_inputs.append(author_token + ' [SEP] ' + situation_title)
                                        
                                    raw_dataset_anchor_data['test']['anchor_inputs'].append(anchor_inputs)
                                    raw_dataset_anchor_data['test']['weights'].append(weights)
                                    raw_dataset_anchor_data['test']['is_anchor_list'].append(author in anchor_names)
                                #######
                            elif author_encoder == 'compositeUid':
                                author = dataset.verdictToAuthor[verdict]
                                author_token = '[' +author +']'
                                comp_token = '[COMP_' +author +']'
                                authors_tokens_set.add(author_token)
                                raw_dataset['test']['text'].append(author_token + comp_token + ' [SEP] ' + situation_title)
                            elif author_encoder == 'composite':
                                author = dataset.verdictToAuthor[verdict]
                                comp_token = '[COMP_' +author +']'
                                raw_dataset['test']['text'].append(comp_token + ' [SEP] ' + situation_title)
                            else:
                                raw_dataset['test']['text'].append(situation_title)
                                
                            raw_dataset['test']['label'].append(test_labels[i])
                            raw_dataset['test']['author_node_idx'].append(-1)
                            
                            assert test_labels[i] == dataset.verdictToLabel[verdict] 
                            

                if model_name == 'sbert':
                    logger.info("Training with SBERT, model name is {}".format(model_name))
                    tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)
                    if num_anchors <= 0:
                        if args.dataset_name == 'ArMIS':
                            model = SentBertClassifier(users_layer=USE_AUTHORS, user_dim=user_dim, sbert_model='asafaya/bert-base-arabic', sbert_dim=args.sbert_dim)
                        else:
                            model = SentBertClassifier(users_layer=USE_AUTHORS, user_dim=user_dim, sbert_model=args.sbert_model, sbert_dim=args.sbert_dim)
                    elif anchor_version == "v2":
                        model = SentBertClassifier_Anchor_v2(sbert_model=args.sbert_model, sbert_dim=args.sbert_dim)
                    elif anchor_version == "v4":
                        model = SentBertClassifier_Anchor_v4(sbert_model=args.sbert_model, sbert_dim=args.sbert_dim)
                else:
                    raise Exception('Wrong model name')
                
                # add author tokens to tokenizer
                tokenizer.add_tokens(list(authors_tokens_set))
                model.model.resize_token_embeddings(len(tokenizer))
                
                # for composite embedding
                ##################
                if author_encoder in ['composite', 'compositeUid']:
                    composite_embeddings = create_composite_embeddings(dataframe)
                    
                    comp_tok_list = [f'[COMP_{anno_id}]' for anno_id in composite_embeddings.keys()]
                    comp_embedding_list = list(composite_embeddings.values())
                        
                    tokenizer.add_tokens(comp_tok_list)
                    model.model.resize_token_embeddings(len(tokenizer))
                    
                    for i, comp_emb in enumerate(comp_embedding_list):
                        idx = len(tokenizer) - len(comp_embedding_list) + i
                        with torch.no_grad():
                            model.model.embeddings.word_embeddings.weight[idx] =  torch.tensor(comp_emb)
                ##################
                
                model.to(DEVICE)
                
                ds = DatasetDict()

                for split, d in raw_dataset.items():
                    ds[split] = Dataset.from_dict(mapping=d, features=Features({'label': Value(dtype='int64'), 
                                                                                    'text': Value(dtype='string'), 'index': Value(dtype='int64'), 'author_node_idx': Value(dtype='int64')}))
                
                def tokenize_function(example):
                    return tokenizer(example["text"], truncation=True)

                logger.info("Tokenizing the dataset")
                tokenized_dataset = ds.map(tokenize_function, batched=True)
                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
                
                tokenized_dataset = tokenized_dataset.remove_columns(["text"])
                tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
                tokenized_dataset.set_format("torch")
                
                def tokenize_function_anchors(example):
                    tokenized = tokenizer([item for sublist in example["anchor_inputs"] for item in sublist], truncation=True)
                    l = len(example['anchor_inputs'][0])
                    k = len(tokenized['input_ids'])
                    n = k//l
                    anchor_inputs_input_ids = [list(tokenized['input_ids'])[i*l:i*l+l] for i in range(n)]
                    anchor_inputs_attention_mask = [list(tokenized['attention_mask'])[i*l:i*l+l] for i in range(n)]
                    
                    res = {
                        'input_ids': anchor_inputs_input_ids,
                        'attention_mask': anchor_inputs_attention_mask
                    }
                    return res
                
                if num_anchors > 0:
                    anchor_ds = DatasetDict()
                    for split, d in raw_dataset_anchor_data.items():
                        anchor_ds[split] = Dataset.from_dict(mapping=d)
                        
                    tokenized_dataset_anchor = anchor_ds.map(tokenize_function_anchors, batched=True)
                    tokenized_dataset_anchor = tokenized_dataset_anchor.remove_columns(["anchor_inputs"])

                    tokenized_dataset_anchor_list = list(tokenized_dataset_anchor['train']['index'])
                    
                    anchor_weights_all = tokenized_dataset_anchor['train']['weights']
                    anchor_weights_all = [item for sublist in anchor_weights_all for item in sublist]
                    anchor_weights_all_tensor = torch.tensor(anchor_weights_all)

                    is_anchor_list_all = tokenized_dataset_anchor['train']['is_anchor_list']
                    is_anchor_list_all_tensor = torch.tensor(is_anchor_list_all)
                    
                    anchor_inputs_input_ids_all = tokenized_dataset_anchor['train']['input_ids']
                    anchor_inputs_input_ids_all = [item for sublist in anchor_inputs_input_ids_all for item in sublist]
                    anchor_inputs_input_ids_all_tensor_list = [torch.tensor(input_ids) for input_ids in anchor_inputs_input_ids_all]
                    
                    
                    anchor_inputs_attention_mask_all = tokenized_dataset_anchor['train']['attention_mask']
                    anchor_inputs_attention_mask_all = [item for sublist in anchor_inputs_attention_mask_all for item in sublist]
                    anchor_inputs_attention_mask_tensor_list = [torch.tensor(input_ids) for input_ids in anchor_inputs_attention_mask_all]
                    ############
                
                batch_size = args.batch_size
                
                train_dataloader = DataLoader(
                    tokenized_dataset["train"], batch_size=batch_size, collate_fn=data_collator, shuffle = True
                )
                eval_dataloader = DataLoader(
                    tokenized_dataset["val"], batch_size=batch_size, collate_fn=data_collator
                )
                
                test_dataloader = DataLoader(
                    tokenized_dataset["test"], batch_size=batch_size, collate_fn=data_collator
                )

                optimizer = AdamW(model.parameters(), lr=learning_rate)
                
                num_epochs = args.num_epochs
                num_training_steps = num_epochs * len(train_dataloader)
                samples_per_class_train = get_samples_per_class(tokenized_dataset["train"]['labels'])

                lr_scheduler = get_scheduler(
                    "linear",
                    optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=num_training_steps,
                )
                logger.info("Number of training steps {}".format(num_training_steps))
                loss_type=args.loss_type
                progress_bar = tqdm(range(num_training_steps))
                best_accuracy = 0
                best_f1 = 0
                val_metrics = []
                # train_loss = []
                
                all_historys = dict()
                
                # if use_early_stopping:
                #     early_stopper = EarlyStopper()
                # else:
                #     early_stopper = False
                # epoch_state_dicts = []
    
                for epoch in range(num_epochs):
                    train_loss = []
                    
                    start_time_train = time.perf_counter()
                    
                    model.train()
                    
                    for batch in train_dataloader:

                        verdicts_index = batch.pop("index")
                        author_node_idx = batch.pop("author_node_idx")
                        batch = {k: v.to(DEVICE) for k, v in batch.items()}
                        labels = batch.pop("labels")
                        
                        if num_anchors > 0:
                            
                            idxs_for_anchor_ds = [tokenized_dataset_anchor_list.index(i) for i in verdicts_index]
                            idxs_for_anchor_ds_flattened_of_stacked = [i for x in idxs_for_anchor_ds for i in range(x*num_anchors_neighbors, x*num_anchors_neighbors + num_anchors_neighbors)]
                
                            is_anchor_tensor = is_anchor_list_all_tensor[idxs_for_anchor_ds].clone().detach().to(DEVICE)
                            
                            anchor_inputs_input_ids = [anchor_inputs_input_ids_all_tensor_list[i].to(DEVICE) for i in idxs_for_anchor_ds_flattened_of_stacked]
                            anchor_inputs_attention_mask = [anchor_inputs_attention_mask_tensor_list[i].to(DEVICE) for i in idxs_for_anchor_ds_flattened_of_stacked]
                            
                            anchor_batch = {
                                'input_ids': anchor_inputs_input_ids,
                                'attention_mask': anchor_inputs_attention_mask
                            } 

                            anchor_batch = tokenizer.pad(anchor_batch, padding=True)
                            
                            anchor_batch = {k: v.to(DEVICE) for k, v in anchor_batch.items()}
                    
                            len_batch = len(verdicts_index)
                            len_tokenized = len(anchor_batch['input_ids'][0])
                            
                            anchor_weights = anchor_weights_all_tensor[idxs_for_anchor_ds_flattened_of_stacked].clone().detach().to(DEVICE)
                            anchor_weights = anchor_weights.reshape(len(anchor_weights),1)
                            
                            if anchor_version == 'v4':
                                anchors_inputs = model(anchor_batch) # these are logits
                            elif anchor_version == 'v2':  
                                anchors_inputs = model(anchor_batch, only_anchor=True) # these are downsized bert results
                            else:
                                raise Exception('wrong anchor version')
                                
                            anchors_inputs = torch.mul(anchors_inputs, anchor_weights)
                            
                            # since we have multiple anchor logits for each non_anchor (number_of_neighbors): reshape and average
                            anchors_inputs = anchors_inputs.reshape(len_batch, num_anchors_neighbors, anchors_inputs.shape[-1])
                            anchors_inputs = anchors_inputs.sum(axis=1)
                            
                            assert not anchors_inputs.isnan().any()
                            
                        if USE_AUTHORS and  (author_encoder == 'average' or author_encoder == 'attribution'):
                            authors_embeddings = torch.stack([embedder.embed_author(dataset.verdictToAuthor[dataset.idToVerdict[index.item()]]) for index in verdicts_index]).to(DEVICE)
                            output = model(batch, authors_embeddings)
                        else:
                            if num_anchors > 0:
                                # anchors_logits is the average weighted anchor logits
                                # is_anchor_tensor is used to determine if a user is achor: then dont use the anchor weights.
                                output = model(batch, anchor_input=anchors_inputs, is_anchor_tensor=is_anchor_tensor)
                            else:
                                output = model(batch)
                                
                        assert not output.isnan().any()
                        
                        loss = loss_fn(output, labels, samples_per_class_train, loss_type=loss_type)
                        train_loss.append(loss.item())
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        
                    end_time_train = time.perf_counter()
                    run_time_train = (end_time_train - start_time_train) * 1000 # milliseconds
                    
                    graph_model = None
                    data = None
                    
                    val_metric = evaluate(eval_dataloader, model, graph_model, data, embedder, USE_AUTHORS, dataset, author_encoder, samples_per_class_train, loss_type)
                    val_metrics.append(val_metric)
                    
                    train_loss_mean = float(torch.tensor(train_loss).mean())
                    
                    history = {
                        'train_loss_means': train_loss_mean,
                        'val_loss_mean': val_metric['loss_mean'],
                        'val_f1_majority_bin': val_metric['binary_majority'],
                        'val_f1_majority_macro': val_metric['macro_majority'],
                        'val_acc_majority': val_metric['accuracy_majority'],
                        'val_f1_individual_binary': val_metric['binary'],
                        'val_f1_individual_macro': val_metric['macro'],
                        'val_acc_individual': val_metric['accuracy'],
                        'train_time_millis': run_time_train,
                        'eval_time_millis': val_metric['run_time_millis']
                    }
                    
                    all_historys[f"Epoch{epoch}"] = history
                    
                    logger.info("Epoch {} **** Loss {} **** Metrics validation: {}".format(epoch, loss, val_metric))
                    if val_metric['f1_weighted'] > best_f1:
                        best_f1 = val_metric['f1_weighted']
                        torch.save(model.state_dict(), checkpoint_dir)
                        if USE_AUTHORS and author_encoder == 'graph':
                            torch.save(graph_model.state_dict(), graph_checkpoint_dir)        
                    
                    # if early_stopper and not early_stopper.loss_is_decreasing(val_metric['loss_mean']):
                    #     best_epoch = early_stopper.return_best_epoch()
                    #     model.load_state_dict(epoch_state_dicts[best_epoch])
                    #     break
                        

                logger.info("Evaluating")
                model.load_state_dict(torch.load(checkpoint_dir))
                model.to(DEVICE)
                if USE_AUTHORS and author_encoder == 'graph':
                    graph_model.load_state_dict(torch.load(graph_checkpoint_dir))
                    graph_model.to(DEVICE)
                
                test_metrics = evaluate(eval_dataloader, model, graph_model, data, embedder, USE_AUTHORS, dataset, author_encoder, samples_per_class_train, loss_type, return_predictions=True)
                results = test_metrics.pop('results')
                logger.info(test_metrics)
                
        
                test_res = {
                        'test_f1_majority_binary': test_metrics['binary_majority'],
                        'test_f1_majority_macro': test_metrics['macro_majority'],
                        'test_acc_majority': test_metrics['accuracy_majority'],
                        'test_f1_individual_binary': test_metrics['binary'],
                        'test_f1_individual_macro': test_metrics['macro'],
                        'test_acc_individual': test_metrics['accuracy'],
                        'test_time_millis': test_metrics['run_time_millis'],
                        'dataset_size': dataset.number_of_posts,
                        'number_of_annotations': dataset.number_of_annotations,
                        'train_size': len(train_verdicts),
                        'val_size': len(val_verdicts),
                        'test_size': len(test_verdicts)
                    }
                all_historys[f"Test_results"] = test_res
                
                # DELETE checkpoint dir:
                os.remove(checkpoint_dir)
                
                res_file_all_history = os.path.join(path_to_save, f"model_results" + ".json")
                    
                with open(res_file_all_history, 'w') as file:
                    json.dump(all_historys, file)
                    
                if save_indi_preds:
                    with open(os.path.join(path_to_save, f"all_preds_and_labels" + ".json"), 'w') as file:
                        json.dump(results, file)
                
                hyps = {
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'annotator_ids': annotator_ids
                }
                hyps['seed'] = seed
                hyps['type'] = f'NO VERDICTS TEXT + SITUATION {args.situation}'
                hyps['sbert_model'] = args.sbert_model
                hyps['model_name'] = args.model_name
                hyps['use_authors_embeddings'] = USE_AUTHORS
                hyps['authors_embedding_path'] = authors_embedding_path
                hyps['author_encoder'] = author_encoder
                hyps['split_type'] = split_type
                hyps['train_stats'] = train_size_stats
                hyps['val_stats'] = val_size_stats
                hyps['test_stats'] = test_size_stats
                hyps['epochs'] = num_epochs
                hyps['optimizer'] = optimizer.defaults
                hyps["loss_type"] = loss_type
                hyps['test_metrics'] = test_metrics
                hyps['checkpoint_dir'] = checkpoint_dir
                hyps['val_metrics'] = val_metrics
                hyps['results'] = results
                
                if comment_scalability:
                    hyps['comments_percentage'] = comments_percentage
                    hyps['num_comments_per_anno_train'] = num_comments_per_anno_train
                
                with open(path_to_save+'hyper_parameters.json', 'w') as file:
                    json.dump(hyps, file)

                

                    
                
                

                