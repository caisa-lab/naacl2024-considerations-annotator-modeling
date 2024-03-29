# Corpus Considerations for Annotator Modeling and Scaling

This repository contains the code for our NAACL 2024 publication. If you use this repository, please cite our paper.

```
@inproceedings{sarumi-2024-corpus-considerations,
  title = {Corpus Considerations for Annotator Modeling and Scaling},
  author = {Sarumi, Olufunke O. and Neuendorf, Béla and Plepi, Joan and Flek, Lucie and Schlötterer, Jörg and Welch, Charles},
  booktitle = {Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics},
  year = {2024},
}
```

## Running the Models

- **run-any-scalability_any.slurm** runs a SLURM job for training and evaluating one model run, given a number of annotators, which model to use, which dataset to use it then calls with the given parameters one of the following two scripts:

  - **ft_bert_scalability.py** runs the training and evaluation of the models from [Dealing with Disagreements: Looking Beyond the Majority Vote in Subjective Annotations](https://arxiv.org/abs/2110.05719)

  - **run_multi-tasking_model_GHC_GE_SC.py** runs the training and evaluation of the models from [Unifying Data Perspectivism and Personalization: An Application to Social Norms](https://aclanthology.org/2022.emnlp-main.500/)

- to run many experiments at once, use the **run-any-model-any-ds.sh** or **run-multiple-experiments.py** script

## Result Dataframe

-- TODO Note about all_scal_annos_res -- or other CSVs that contain full experiment results
-- 18k experiments across the CSVs in dataframes_for_plots?

## Notebooks

- **dataset_exploration_and_create_subsets.ipynb** provides code to create subsets from the datasets

- **check_results_for_paper.ipynb** use this notebook to view the results

