# Corpus Considerations for Annotator Modeling and Scaling

This repository contains the code for our NAACL 2024 publication. If you use this repository, please cite our paper.

```
@inproceedings{sarumi-2024-corpus-considerations,
  title = {Corpus Considerations for Annotator Modeling and Scaling},
  author = {Sarumi, Olufunke and Neuendorf, Béla and Plepi, Joan and Flek, Lucie and Schlötterer, Jörg and Welch, Charles},
  booktitle = {Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics},
  year = {2024},
}
```

## Running the Models

- **run-any-scalability_any.slurm** runs a SLURM job for training and evaluating one model run, given a number of annotators, which model to use, which dataset to use it then calls with the given parameters one of the following two scripts:

  - **ft_bert_scalability.py** runs the training and evaluation of the models from [Dealing with Disagreements: Looking Beyond the Majority Vote in Subjective Annotations](https://arxiv.org/abs/2110.05719)

  - **run_multi-tasking_model_GHC_GE_SC.py** runs the training and evaluation of the models from [Unifying Data Perspectivism and Personalization: An Application to Social Norms](https://aclanthology.org/2022.emnlp-main.500/)

- to run many experiments at one, use the **run-any-model-any-ds.sh** or **run-multiple-experiments.py** script

## Notebooks

- **dataset_exploration_and_create_subsets.ipynb** provides code to create subsets from the datasets

- **check_results_for_paper.ipynb** use this notebook to view the results


## Results

As described in **notebooks/check_results_for_paper.ipynb**, the csv **notebooks/dataframes_for_plots/all_scal_annos_res.csv** contains all results of scaling the number of annotators, as loading all results takes some time. But there is also a function provided loading the results of the different (sub)directories.

The results dataframes from our experiments can as in the notebook, using the syntax listed here:

```df0 = pd.read_csv('notebooks/dataframes_for_plots/all_scal_annos_res.csv')
df1 = pd.read_csv('notebooks/dataframes_for_plots/all_scal_comments_14_res.csv')
df2 = pd.read_csv('notebooks/dataframes_for_plots/all_scal_comments_50_res.csv')
HSB_explo_res_14, missing = get_all_res_as_df(6, comments='comment_', only_df='HSBrexit')
ArMIS_explo_res_14, missing = get_all_res_as_df(3, comments='comment_', only_df='ArMIS')
CA_explo_res_14, missing = get_all_res_as_df(7, comments='comment_', only_df='ConvAbuse')
```


The columns and relevant fields of the result dataframes are listed here:

Dataset short names:
- ArMIS
- ConvAbuse
- HSBrexit
- GE (used in combination with one of the emotions as GE'emotion with emotion in [anger, disgust, fear, joy, sadness, surprise])
- GHC
- MD
- SC

Model short names:
- comp: composite embedding
- compUid: composite embedding combined with UID
- sbertbase
- uid
- bertbase
- multi-tasking
- ae: average embedding (only for sc)
- aa: authorship attribution (only for sc)

Columns of results dataframe "dataframes_for_plots/all_scal_annos_res.csv" (as well as the other CSVs):
- dataset_name: the name of the dataset and the model applied to it.
- ds_name: name of the dataset
- model
- dataset_size: number of samples
- number_annotations: total number of annotations (verdicts) for the samples
- num_annos: number of annotators
- f1_bin_maj
- f1_macro_maj
- f1_bin_indi
- f1_macro_indi
- test_acc_maj
- test_acc_indi
- mean_train_millis: per epoch
- mean_train_millis_per_annotation: per epoch per annotation
- mean_train_millis_per_sample: per epoch per sample
- mean_eval_millis_per_annotation: to be ignored, may be inconsistent, since these values were not used, they were not stored properly in the process
- mean_test_millis_per_annotation: to be ignored, may be inconsistent, since these values were not used, they were not stored properly in the process
- sum_all_millis: to be ignored, may be inconsistent, since these values were not used, they were not stored properly in the process
- annotations_per_anno: only for scaling annotations per annotator
- comment_percentage: only for scaling annotations per annotator
- learning_rate: to be ignored, may be inconsistent, since these values were not used, they were not stored properly in the process
- fold
  
