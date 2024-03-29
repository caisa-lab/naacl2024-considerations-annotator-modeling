import os



#dataset=$1
#scal_com_scal=$2
#emotion=$3
#task=$4
#seed_i=$5
#num_annos=$6
#model_short=$7
#model_class=$8
#run_time_hours=$9
#num_anchors=${10}
#num_anchors_neighbors=${11}
#overwrite_existing_results=${12}
#anchor_v3=${13}


if __name__ == "__main__":
    
    check_existing_results = True

    emotions = ['aita'] # lists of datasets to use, each emotion corresponds to a dataset.
    # hate aita anger fear joy disgust sadness surprise MD_emotion ArMIS_emotion HSBrexit_emotion ConvAbuse_emotion
    
    tasks = ['multi'] #['multi', 'single'] 
    scal_com_scal = ['scalability'] # ['scalability', 'comment_scalability']
    model_classes = ['multi']# ['multi', 'perso'] # multi is multi-tasking model from Davani et al, perso is the model from Plepi et al
    
    sc_perso_multi_model_shorts = ['uid'] #['aa', 'ae', 'uid'] # if we use the models from Plepi et al, we can choose: aa = Authorship attribution, ae = average embedding, uid = user ID
 
    use_full = False # for GE and GHC scalability
    overwrite_existing_results = True
    
    save_indi_preds = True

    anchor_version = 'NoAnchor' #one of v2, v3, v4, or just sth like NoAnchor

    num_anchors = 0
    num_anchors_neighbors = 0

    seeds = [0,1,2,3,4]
    # num_annos_list = [6, 8, 10, 12, 14, 16, 18] + list(range(22, 83, 4)) + list(range(100, 1000, 100)) + list(range(1000, 2510, 300))
    num_annos_list =  [2500] # the number of annotators to use, given that the a corresponding subset of the dataset exists
    
    
    anchor_v3 = anchor_version == 'v3' # flag for run time limit: v3 on multi-tasking
    uid = model_classes == 'perso' # True # flag for run time limit
    
    assert num_anchors== 0 or anchor_version in ['v2','v3','v4']
    
    for num_annos in num_annos_list:
        if anchor_v3: # multi-tasking and v3
            if num_annos <= 30:
                run_time_hours_minutes="00:20:00"
            elif num_annos <= 54:
                run_time_hours_minutes="01:00:00"
            elif num_annos <= 82:
                run_time_hours_minutes="02:00:00"
            elif num_annos <= 300:
                run_time_hours_minutes="05:00:00"
            elif num_annos <= 600:
                run_time_hours_minutes="07:00:00"
            elif num_annos <= 1000:
                run_time_hours_minutes="09:00:00"
            elif num_annos <= 1900:
                run_time_hours_minutes="18:00:00"
            else:
                run_time_hours_minutes="24:00:00"
        elif uid: # uid with or without anchor
            if anchor_version == 'v4':
                if num_annos <= 22:
                    run_time_hours_minutes="0-1"
                elif num_annos <= 74:
                    run_time_hours_minutes="02:00:00"
                elif num_annos <= 200:
                    run_time_hours_minutes="05:00:00"
                elif num_annos <= 300:
                    run_time_hours_minutes="07:00:00"
                elif num_annos <= 400:
                    run_time_hours_minutes="09:00:00"
                elif num_annos <= 500:
                    run_time_hours_minutes="11:00:00"
                elif num_annos <= 600:
                    run_time_hours_minutes="12:00:00"
                elif num_annos <= 700:
                    run_time_hours_minutes="13:00:00"
                elif num_annos <= 800:
                    run_time_hours_minutes="15:00:00"
                elif num_annos <= 900:
                    run_time_hours_minutes="17:00:00"
                elif num_annos <= 1000:
                    run_time_hours_minutes="18:00:00"
                elif num_annos <= 1300:
                    run_time_hours_minutes="23:00:00"
                elif num_annos <= 1600:
                    run_time_hours_minutes="27:00:00"
                elif num_annos <= 1900:
                    run_time_hours_minutes="33:00:00"
                elif num_annos <= 2200:
                    run_time_hours_minutes="37:00:00"
                else:
                    run_time_hours_minutes="42:00:00"
            else:
                if num_annos <= 100:
                    run_time_hours_minutes="0-1"
                elif num_annos <= 1300:
                    run_time_hours_minutes="02:00:00"
                else:
                    run_time_hours_minutes="03:30:00"
        else: # multi-tasking with other anchor versions
            if num_annos <= 50:
                run_time_hours_minutes="00:15:00"
            elif num_annos <= 82:
                run_time_hours_minutes="00:30:00"
            elif num_annos <= 300:
                run_time_hours_minutes="01:00:00"
            elif num_annos <= 600:
                run_time_hours_minutes="02:00:00"
            elif num_annos <= 1000:
                run_time_hours_minutes="04:00:00"
            elif num_annos <= 1600:
                run_time_hours_minutes="06:00:00"
            else:
                run_time_hours_minutes="08:00:00"
        
        for emotion in emotions:
            if emotion == "hate":
                dataset = "GHC"
                perso_multi_model_shorts = ["uid"]
                perso_single_model_shorts = ["base"]
                mt_multi_model_shorts = ["mt"]
                mt_single_model_shorts = ["base"]
            elif emotion == "aita":
                dataset = "SC"
                perso_multi_model_shorts = sc_perso_multi_model_shorts
                perso_single_model_shorts = ["base"]
                mt_multi_model_shorts = ["mt"]
                mt_single_model_shorts = ["base"]
            else:
                dataset = "GE"
                perso_multi_model_shorts = ["uid"]
                perso_single_model_shorts = ["base"]
                mt_multi_model_shorts = ["mt"]
                mt_single_model_shorts = ["base"]

            for task in tasks:
                
                for model_class in model_classes:

                    if model_class == "perso":
                        if task == "multi":
                            model_short_list = perso_multi_model_shorts
                        else: # task == "single"
                            model_short_list = perso_single_model_shorts
                    else: # model_class == "multi"
                        if task == "multi":
                            model_short_list = mt_multi_model_shorts
                        else: # task == "single"
                            model_short_list = mt_single_model_shorts

                    for seed_i in seeds:
                        
                            for scal_com in scal_com_scal:
                                
                                for model_short in model_short_list:
                                    
                                                    #    SC     scalability      aita   multi    0        6          uid          perso            00:30:00                6             3                        True                       v4           save_indi_preds
                                    command_args = f"{dataset} {scal_com} {emotion} {task} {seed_i} {num_annos} {model_short} {model_class} {run_time_hours_minutes} {num_anchors} {num_anchors_neighbors} {overwrite_existing_results} {anchor_version} {save_indi_preds}"
                                    print(command_args)
                                    
                                    if scal_com == 'scalability' and check_existing_results:
                                        emotion_for_dir = ""
                                        if emotion not in ['aita', 'hate']:
                                            emotion_for_dir = "'emotion"

                                        model_short_for_dir = model_short
                                        if model_short_for_dir == 'base':
                                            if model_classes == 'multi':
                                                model_short_for_dir = 'bertbase'
                                            else:
                                                model_short_for_dir = 'sbertbase'
                                                
                                        path_to_res = f"results/{model_classes}_{scal_com}_{seed_i}/{dataset}{emotion_for_dir}-{model_short_for_dir}_{num_annos}_annos/"
                                        
                                        if not os.path.exists(os.path.join(path_to_res,'model_results.json')):
                                            os.system(f"sbatch run-any-scalability_any.slurm {command_args}")
                                        else:
                                            print(f" results exist: {path_to_res}")
                                    else: 
                                        os.system(f"sbatch run-any-scalability_any.slurm {command_args}")
                                