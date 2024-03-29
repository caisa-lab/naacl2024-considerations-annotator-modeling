#!/bin/bash

dataset=$1
scal_com_scal=$2
emotion=$3
task=$4
seed_i=$5
num_annos=$6
model_short=$7
model_class=$8
run_time_hours_minutes=$9
num_anchors=${10}
num_anchors_neighbors=${11}
overwrite_existing_results=${12}
anchor_version=${13}
save_indi_preds=${14}
use_full=${15}

#      sbatch run-any-scalability_any.slurm SC scalability aita single 0 2500 base multi 06:00:00 0 0 True NoAnchor True
#      sbatch run-any-scalability_any.slurm SC scalability aita single 0 2500 base perso 06:00:00 0 0 True NoAnchor True
#      sbatch run-any-scalability_any.slurm SC scalability aita multi 0 6 mt multi 06:00:00 0 0 False NoAnchor
#      sbatch run-any-scalability_any.slurm SC scalability aita multi 0 6 uid perso 00:30:00 0 0 True NoAnchor
##     sbatch run-any-scalability_any.slurm SC scalability aita multi 0 16 uid perso 02:00:00 6 3 True v4
##     sbatch run-any-scalability_any.slurm SC scalability aita multi 0 58 uid perso 00:30:00 38 5 True v4
##     sbatch run-any-scalability_any.slurm SC scalability aita multi 2 58 mt multi 00:30:00 38 5 True v3 True
##     sbatch run-any-scalability_any.slurm MD scalability MD_emotion single 0 6 mt multi 00:30:00 0 0 True noAnchor True
##     sbatch run-any-scalability_any.slurm MD comment_scalability MD_emotion multi 1 50 mt multi 00:45:00 0 0 True noAnchor True

echo $dataset $scal_com_scal $emotion $task $seed_i $model_short $model_class >> logfile.log
     # SC      scalability    aita     multi  0       False     aa           perso         bert
if [ "$scal_com_scal" = "comment_scalability" ]; then
    s_short="com_s"
    comment_scal_bool=True
elif [ "$scal_com_scal" = "scalability" ]; then
    s_short="an_s"
    comment_scal_bool=False
else
    echo "Error! wrong comment_scal_string" >> logfile.log
    exit 125
fi

if [ $model_short = "aa" ]; then
    use_authors="True"
    author_encoder='attribution'
    authors_embedding_path='../data/embeddings/emnlp/attribution/sit_mlp_prediction.pkl'
elif [ $model_short = "ae" ]; then
    use_authors="True"
    author_encoder='average'
    authors_embedding_path='../data/embeddings/emnlp/sbert_authorAMIT.pkl'
elif [ $model_short = "uid" ]; then
    use_authors="False"
    author_encoder='user_id'
    authors_embedding_path='none'
elif [ $model_short = "base" ]; then
    use_authors="False"
    author_encoder='none'
    authors_embedding_path='none'
elif [ $model_short = "composite" ]; then
    use_authors="False"
    author_encoder='composite'
    authors_embedding_path='none'
elif [ $model_short = "compositeUid" ]; then
    use_authors="False"
    author_encoder='compositeUid'
    authors_embedding_path='none'
elif [ $model_short = "mt" ]; then
    echo true
else
    echo "Error! wrong model_name" >> logfile.log
    exit 125
fi

if [ "$dataset" = "GE" ]; then
    ds_emo_name="$dataset'$emotion"
else
    ds_emo_name="$dataset"
fi

use_full_str=''
if [ $use_full = 'True' ]; then
    use_full_str='-full'
fi

sbatch <<EOT
#!/bin/bash

#SBATCH --partition=short_gpu
#SBATCH --partition=owner_fb12
##SBATCH --gpus=a100:1
#SBATCH --gres=gpu:1
#SBATCH --job-name=$s_short-$dataset$use_full_str-$emotion-$num_annos-$model_class-$model_short-$task-$seed_i-$anchor_version-$num_anchors-$num_anchors_neighbors
#SBATCH --output=/home/neuendo4/slurm/output/$s_short-$dataset$use_full_str-$emotion-$num_annos-$model_class-$model_short-$task-$seed_i-$anchor_version-$num_anchors-$num_anchors_neighbors.out
#SBATCH --error=/home/neuendo4/slurm/output/$s_short-$dataset$use_full_str-$emotion-$num_annos-$model_class-$model_short-$task-$seed_i-$anchor_version-$num_anchors-$num_anchors_neighbors.err
#SBATCH --mail-user=neuendob@students.uni-marburg.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=$run_time_hours_minutes
##SBATCH --time=0-1

module load cuda/11.7


# cd src
cd Master_Thesis/src

if [ "$model_class" = "perso" ]; then
    python ft_bert_scalability.py --use_authors=$use_authors --author_encoder=$author_encoder --use_full=$use_full --loss_type='focal' --num_epochs=10 --bert_tok='sentence-transformers/all-distilroberta-v1' --sbert_dim=768 --model_name='sbert' --split_type='sit'  --authors_embedding_path=$authors_embedding_path --dataset_name="$ds_emo_name" --situation='text' --comment_scalability=$comment_scal_bool --seed_i=$seed_i --num_anchors=$num_anchors --num_anchors_neighbors=$num_anchors_neighbors --overwrite_existing_results=$overwrite_existing_results --anchor_version=$anchor_version  --num_annos=$num_annos --save_indi_preds=$save_indi_preds
elif [ "$model_class" = "multi" ]; then
    python run_multi-tasking_model_GHC_GE_SC.py --dataset="$ds_emo_name" --task=$task --seed_i=$seed_i --comment_scalability=$comment_scal_bool --use_full=$use_full  --num_anchors=$num_anchors --num_anchors_neighbors=$num_anchors_neighbors --overwrite_existing_results=$overwrite_existing_results --anchor_version=$anchor_version --num_annos=$num_annos --save_indi_preds=$save_indi_preds
else
    echo "Error! wrong model_class" >> logfile.log
    exit 125
fi