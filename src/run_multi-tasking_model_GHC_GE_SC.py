from datetime import datetime

from joan_utils.utils import str2bool
from joan_utils.train_utils import AuthorsEmbedder
from utils.constants import *
from utils.run_training import train_model
import os
from argparse import ArgumentParser

from datetime import datetime

parser = ArgumentParser()
parser.add_argument("--dataset", dest="dataset", required=True, type=str)
parser.add_argument("--seed_i", dest="seed_i", default=0, type=int)
parser.add_argument("--task", dest="task", required=True, type=str)
parser.add_argument("--use_full", dest="use_full", type=str2bool, default=False)
parser.add_argument("--mt_base_model", dest="mt_base_model", type=str, default='bert')
parser.add_argument("--comment_scalability", dest="comment_scalability", type=str2bool, default=False)
parser.add_argument("--num_anchors", dest="num_anchors", type=int, default=0)
parser.add_argument("--num_anchors_neighbors", dest="num_anchors_neighbors", type=int, default=0)
parser.add_argument("--anchor_version", dest="anchor_version", type=str, default='v2')
parser.add_argument("--num_annos", dest="num_annos", type=int, default=0)
parser.add_argument("--overwrite_existing_results", dest="overwrite_existing_results", type=str2bool, default=False)
parser.add_argument("--save_indi_preds", dest="save_indi_preds", type=str2bool, default=False)

args = parser.parse_args()

assert DEVICE == torch.device('cuda')

# Parameters
#dataset_name = 'GHC'  # 'GHC' or 'GE'
#emotion = 'hate' #'anger', 'disgust', 'joy', 'fear', 'sadness', 'surprise | for GHC: 'hate'
#task = 'multi'  # 'single' or 'multi'
use_ds_reduced_to_sc = True
batch_size = 16
num_epochs = 10
num_splits = 5
val_ratio = 0.25
print_interval = 100/batch_size
#learning_rate = 5e-6 # 1e-5 for GHC, 5e-6 for GE
stop_after_fold = 2
max_length = 64
min_number_of_annotations = 1
# If false, stratification is done by all combinations of annotators (possibly 2^n_annotators)
stratify_by_majority_vote = True

comment_scalability = args.comment_scalability
task = args.task
mt_base_model = args.mt_base_model
use_full = args.use_full
num_anchors = args.num_anchors
num_anchors_neighbors = args.num_anchors_neighbors
overwrite_existing_results = args.overwrite_existing_results
anchor_version = args.anchor_version

save_indi_preds = args.save_indi_preds

dt_string = datetime.now().strftime("%m-%d:%H")

### only important for GE / GHC with full subsets and not reduced to match the subsets of SC
full_str = ''
full_path_str = ''
if use_full:
    full_str = '*full'
    full_path_str = 'ge_ghc_full_ds_scalability/'
assert (not use_full) or ('GE' in args.dataset or 'GHC' in args.dataset), 'SC is full ds each time anyway'

full_str = ''
comment_str=''
if comment_scalability:
    full_str = '*full'
    comment_str = 'comment_'

# IGNORE as well
mt_base_str = ''
if mt_base_model == 'sbert':
    mt_base_str = '-sbert'
assert not(mt_base_model=='sbert' and task=='single')

# IGNORE as well
version_str = ""
if num_anchors > 0:
    version_str = f'-aa-anchor{version_str}-{num_anchors}-{num_anchors_neighbors}'

for i in [args.seed_i]:
    seed = SEEDS[i]

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
    
    if 'GE' in args.dataset:
        num_annos_list = num_annos_GE
        if comment_scalability:
            comments_percentage_list = GE_comments_percentage_list
            if args.num_annos == 0:
                num_annos_list = comment_num_annos_list
        learning_rate = 5e-6
        if not "'" in args.dataset:
            raise Exception('missing Emotion')
        emotion = args.dataset.split("'")[-1]
        dataset_name = 'GE'
        
    elif args.dataset == 'GHC':
        num_annos_list = num_annos_GHC
        if comment_scalability:
            comments_percentage_list = GHC_comments_percentage_list  
            if args.num_annos == 0:
                num_annos_list = comment_num_annos_list[:1]
        learning_rate = 1e-5 
        emotion = 'not_needed'
        dataset_name = 'GHC'
        
    elif args.dataset == 'SC':
        num_annos_list = num_annos_SC
        if comment_scalability:
            comments_percentage_list = SC_comments_percentage_list 
            if args.num_annos == 0:
                num_annos_list = comment_num_annos_list
        learning_rate = 2e-5 
        emotion = 'not_needed'
        dataset_name = 'SC'
        
    elif args.dataset == 'MD':
        num_annos_list = num_annos_MD
        if comment_scalability:
            comments_percentage_list = funkes_ds_comments_percentage_list 
            if args.num_annos == 0:
                num_annos_list = comment_num_annos_list
        learning_rate = 3e-5 
        emotion = 'not_needed'
        dataset_name = 'MD'
        
    elif args.dataset == 'HSBrexit':
        num_annos_list = num_annos_HSBrexit
        if comment_scalability:
            comments_percentage_list = funkes_ds_comments_percentage_list 
        learning_rate = 5e-5 
        emotion = 'not_needed'
        dataset_name = 'HSBrexit'
        
    elif args.dataset == 'ConvAbuse':
        num_annos_list = num_annos_ConvAbuse
        if comment_scalability:
            comments_percentage_list = funkes_ds_comments_percentage_list 
        learning_rate = 5e-5 
        emotion = 'not_needed'
        dataset_name = 'ConvAbuse'
        
    elif args.dataset == 'ArMIS':
        num_annos_list = num_annos_ArMIS
        if comment_scalability:
            comments_percentage_list = funkes_ds_comments_percentage_list 
        learning_rate = 1e-5 
        emotion = 'not_needed'
        dataset_name = 'ArMIS'
    else:
        raise Exception('wrong ds name: ', args.dataset)
        
    # zip_list = zip(['GE','GHC','SC'], [5e-6, 1e-5, 2e-5], ['anger','hate','SC'], [num_annos_GE, num_annos_GHC, num_annos_SC])
    # if args.dataset != 'False':
    zip_list = zip([dataset_name], [learning_rate], [emotion], [num_annos_list])
        
    for dataset_name, learning_rate, emotion, num_anno_list in zip_list:
    
        if not use_ds_reduced_to_sc and not dataset_name=='sc':
            dataset_name = f"{dataset_name}*full"
        
        for num_annos in num_anno_list:
            for comments_percentage in comments_percentage_list:
                
                comment_per_str=''
                if comments_percentage:
                    comment_per_str = f'{int(comments_percentage*100)}_percentage_'
                
                if dataset_name =='SC':
                    path_to_dataset = f'../data/sc_filtered_from_bela/{dataset_name.lower()}_{num_annos}_annos_filtered.csv'   
                elif dataset_name in ['HSBrexit','MD','ConvAbuse','ArMIS']:
                    path_to_dataset = f'../data/matching_sizes_data/{args.dataset}_{num_annos}_annos_filtered.csv'   
                else:
                    path_to_dataset = f'../data/matching_sizes_data/{args.dataset}{full_str}_{num_annos}_annos_filtered.csv'   
                    
                if task == 'multi':
                    path_to_save = f'../results/{full_path_str}multi_{comment_str}scalability_{i}/{args.dataset}-multi-tasking{mt_base_str}{version_str}_{comment_per_str}{num_annos}_annos/'
                elif task == 'single':
                    path_to_save = f'../results/{full_path_str}multi_{comment_str}scalability_{i}/{args.dataset}-bertbase_{comment_per_str}{num_annos}_annos/'
                
                parent_foler_path = os.path.join(*path_to_save.strip('/').split('/')[:-1])
                if not os.path.isdir(parent_foler_path):
                    os.mkdir(parent_foler_path)    
                
                # IGNORE as well    
                if num_anchors>0:
                    if comment_scalability:
                        authors_embedding_path = f'../data/embeddings/emnlp/attribution/sit_mlp_prediction_{14}_annos_{int(comments_percentage*100)}_percentage.pkl'
                    elif num_anchors in [14,50]:
                        authors_embedding_path = f'../data/embeddings/emnlp/attribution/sit_mlp_prediction_{num_anchors}_annos_{100}_percentage.pkl'
                    else:
                        authors_embedding_path = f'../data/embeddings/emnlp/attribution/sit_mlp_prediction_{num_anchors}_annos.pkl'

                    user_dim = num_anchors
                    embedder = AuthorsEmbedder(embeddings_path=authors_embedding_path, dim=user_dim)
                else:
                    embedder = None  
                
                try:  
                    train_model(dataset_name,
                                emotion,
                                task,
                                path_to_dataset,
                                path_to_save, 
                                seed,
                                batch_size,
                                num_epochs,
                                num_splits,
                                val_ratio,
                                learning_rate,
                                stop_after_fold,
                                max_length,
                                min_number_of_annotations,
                                stratify_by_majority_vote,
                                comments_percentage=comments_percentage,
                                mt_base_model=mt_base_model,
                                embedder=embedder,
                                num_anchors=num_anchors,
                                num_anchors_neighbors=num_anchors_neighbors,
                                overwrite_existing_results=overwrite_existing_results,
                                anchor_version=anchor_version,
                                save_indi_preds=save_indi_preds
                                )
                    
                except FileExistsError as e:
                    print(e)  