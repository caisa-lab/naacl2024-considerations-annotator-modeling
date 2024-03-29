# #!/bin/bash

# dataset=$1
# scal_com_scal=$2
# emotion=$3
# task=$4
# seed_i=$5
# num_annos=$6
# model_short=$7
# model_class=$8
# run_time_hours_minutes=$9
# num_anchors=${10}
# num_anchors_neighbors=${11}
# overwrite_existing_results=${12}
# anchor_version=${13}
# save_indi_preds=${14}

num_anchors=0
num_anchors_neighbors=0
anchor_version='None'

Emotions="hate anger fear joy disgust sadness surprise"
# Emotions="hate" #"hate aita MD_emotion ArMIS_emotion HSBrexit_emotion ConvAbuse_emotion"

Tasks="multi single" #multi single
# Tasks="multi" #multi single

# Scal_com_scal="scalability comment_scalability" #"scalability comment_scalability"
Scal_com_scal="comment_scalability" #"scalability comment_scalability"

# Model_classes="multi perso" #Model_classes="multi perso"
Model_classes="perso" #Model_classes="multi perso"

######### ! only for GE
use_full_for_GE_GHC=False

overwrite_existing_results=True
save_indi_preds=False

# uncomment code!
use_custom_num_annos_list=False
custom_num_annos_list="8"

use_custom_model_shorts=True
custom_model_shorts="composite compositeUid"

seeds="0 1 2 3 4"


cd /home/neuendo4/slurm/slurm_scripts

for scal_com_scal in $Scal_com_scal;
do
    for emotion in $Emotions;
        do
            use_full=False

            if [ "$emotion" = "hate" ]; then
                use_full=$use_full_for_GE_GHC
                dataset="GHC"
                if [ "$scal_com_scal" = "comment_scalability" ]; then
                    num_annos_list="14"
                else   
                    num_annos_list="6 8 10 12 14 16 18"
                fi

            elif [ "$emotion" = "aita" ]; then
                dataset="SC"
                if [ "$scal_com_scal" = "comment_scalability" ]; then
                    num_annos_list="14 50"
                else   
                    num_annos_list="6 8 10 12 14 16 18 22 26 30 34 38 42 46 50 54 58 62 66 70 74 78 82 100 200 300 400 500 600 700 800 900 1000 1300 1600 1900 2200 2500"
                fi

            elif [ "$emotion" = "MD_emotion" ]; then
                dataset="MD"
                if [ "$scal_com_scal" = "comment_scalability" ]; then
                    num_annos_list="14 50"
                else   
                    num_annos_list="6 8 10 12 14 16 18 22 26 30 34 38 42 46 50 54 58 62 66 70 74 78 82 100 200 300 400 500 600 700 751"
                fi

            elif [ "$emotion" = "HSBrexit_emotion" ]; then
                dataset="HSBrexit"
                if [ "$scal_com_scal" = "comment_scalability" ]; then
                    num_annos_list="6"
                else   
                    num_annos_list="1 2 3 4 5 6"
                fi

            elif [ "$emotion" = "ConvAbuse_emotion" ]; then
                dataset="ConvAbuse"
                if [ "$scal_com_scal" = "comment_scalability" ]; then
                    num_annos_list="7"
                else   
                    num_annos_list="1 2 3 4 5 6 7 8"
                fi

            elif [ "$emotion" = "ArMIS_emotion" ]; then
                dataset="ArMIS"
                if [ "$scal_com_scal" = "comment_scalability" ]; then
                    num_annos_list="3"
                else   
                    num_annos_list="1 2 3"
                fi

            else
                use_full=$use_full_for_GE_GHC
                dataset="GE"
                if [ "$scal_com_scal" = "comment_scalability" ]; then
                    num_annos_list="14 50"
                else   
                    num_annos_list="6 8 10 12 14 16 18 22 26 30 34 38 42 46 50 54 58 62 66 70 74 78 82"
                fi
            fi

            # if [ $use_full = False ]; then
            #     y='this string doesnt matter, just not empty'
            # fi
            # trap echo $y ERR
            

            if [ $use_custom_num_annos_list = 'True' ]; then
                num_annos_list=$custom_num_annos_list
            fi

            perso_single_model_shorts="base"
            mt_multi_model_shorts="mt"
            mt_single_model_shorts="base"

            if [ "$emotion" = "aita" ]; then
                perso_multi_model_shorts="aa ae uid composite compositeUid"
            else
                perso_multi_model_shorts="uid composite compositeUid"
            fi



        for num_annos in $num_annos_list; do

            if [ "$model_class" = "multi" ]; then
                if [[ "$num_annos" -le 50 ]]; then
                    run_time_hours_minutes="00:45:00"
                elif [[ "$num_annos" -le 82 ]]; then
                    run_time_hours_minutes="01:30:00"
                elif [[ "$num_annos" -le 300 ]]; then
                    run_time_hours_minutes="03:00:00"
                elif [[ "$num_annos" -le 600 ]]; then
                    run_time_hours_minutes="05:00:00"
                elif [[ "$num_annos" -le 1000 ]]; then
                    run_time_hours_minutes="08:00:00"
                elif [[ "$num_annos" -le 1600 ]]; then
                    run_time_hours_minutes="10:00:00"
                else
                    run_time_hours_minutes="12:00:00"
                fi
            else
                if [[ "$num_annos" -le 74 ]]; then
                    run_time_hours_minutes="00:45:00"
                elif [[ "$num_annos" -le 300 ]]; then
                    run_time_hours_minutes="01:30:00"
                elif [[ "$num_annos" -le 600 ]]; then
                    run_time_hours_minutes="02:00:00"
                elif [[ "$num_annos" -le 1000 ]]; then
                    run_time_hours_minutes="04:00:00"
                elif [[ "$num_annos" -le 1600 ]]; then
                    run_time_hours_minutes="06:00:00"
                else
                    run_time_hours_minutes="07:00:00"
                fi
            fi

            if [ "$scal_com_scal" = "comment_scalability" ]; then
                run_time_hours_minutes="09:00:00"
            fi

            for task in $Tasks;
            do

                for model_class in $Model_classes;
                do
                    if [ "$model_class" = "perso" ]; then
                        if [ "$task" = "multi" ]; then
                            model_short_list="$perso_multi_model_shorts"
                        else # "$task" = "single"
                            model_short_list="$perso_single_model_shorts"
                        fi
                    else #"$model_class" = "multi"
                        if [ "$task" = "multi" ]; then
                            model_short_list="$mt_multi_model_shorts"
                        else # "$task" = "single"
                            model_short_list="$mt_single_model_shorts"
                        fi
                    fi

                    if [ $use_custom_model_shorts = 'True' ]; then
                        model_short_list=$custom_model_shorts
                    fi

                    for seed_i in $seeds;
                    do

                        for model_short in $model_short_list;
                        do
                            echo $dataset $scal_com_scal $emotion $task $seed_i $use_full $model_short $model_class $run_time_hours_minutes $num_anchors >> logfile.log
                            sbatch run-any-scalability_any.slurm $dataset $scal_com_scal $emotion $task $seed_i $num_annos $model_short $model_class $run_time_hours_minutes $num_anchors $num_anchors_neighbors $overwrite_existing_results $anchor_version $save_indi_preds $use_full
                        done
        #sbatch slurm/slurm_scripts/run-any-scalability_any.slurm ConvAbuse comment_scalability no_emo single 0 1 composite perso 00:30:00 0 0 True no_anchor False    
                    done 
                done
            done
        done
    done
done