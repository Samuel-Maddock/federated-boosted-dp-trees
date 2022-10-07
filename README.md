# Code for "Federated Boosted Decision Trees with Differential Privacy"

This repository contains code for the ACM CCS'22 paper ["Federated Boosted Decision Trees with Differential Privacy"](https://arxiv.org/abs/2210.02910)
## Reference

If the code and/or paper contained in this repository were useful to you please consider citing this [work](https://arxiv.org/abs/2210.02910):
```
@article{maddock2022federated,
  title={Federated Boosted Decision Trees with Differential Privacy},
  author={Maddock, Samuel and Cormode, Graham and Wang, Tianhao and Maple, Carsten and Jha, Somesh},
  journal={arXiv preprint arXiv:2210.02910},
  year={2022}
}
```

## Outline

The code is split into two components
* `experiments` - Main code for running and plotting experiments
* `federated_gbdt` - Main code for the private GBDT model

In order to run any of the code, datasets need to be downloaded and placed in the `data` directory. We use the following datasets in our experiments:
[Credit 1](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset), [Credit 2](https://www.kaggle.com/competitions/GiveMeSomeCredit/data?select=cs-test.csv), [Adult](https://archive.ics.uci.edu/ml/datasets/adult), [Bank](https://archive.ics.uci.edu/ml/datasets/bank+marketing), [Nomao](https://archive.ics.uci.edu/ml/datasets/Nomao), [Higgs](https://archive.ics.uci.edu/ml/datasets/HIGGS) 

<!--- For ease of replication you can download the full data folder from [here](todo) (approx 300mb) and place it in the root directory of the repo. The Higgs dataset has been subsampled to n=200,000. --->

In order to generate plots and tables as in the paper see "Paper Experiments, Plots and Tables"

In order to replicate the main figures in the paper from scratch see "Replication Instructions"

## Framework

The code structure of `federated_gbdt` is as follows
* `core`
  * `binning`- Contains quantile sketching code from the FEVERLESS implementation
  * `dp_multiq` - Central DP quantiles (not used in the paper)
  * `moments_accountant` - TensorFlow Privacy Moments Accountant (RDP)
  * `pure_ldp` - LDP protocols (not used in paper)
  * `baseline_constants.py` - Contains constants for the FEVERLESS implementation of quantile sketching
  * `loss_functions.py` - Contains loss functions used in the GBDT algorithm
  * `plotting.py` - Debugging code
* `models`
  * `base`
    * `jit_functions.py` - Numba functions for computing GBDT quantities needed for training (split scores and weights)
    * `tree_node.py` - Contains the `DecisionNode` class reworked from the FEVERLESS implementation
    * `tree_base.py` - Base tree implementation
  * `gbdt`
    * `components` 
      * `index_sampler.py`- Contains the `IndexSampler` class for managing which features/observations a tree uses during training
      * `privacy_accountant.py` - Contains the `PrivacyAccountant` class for managing DP during training of a `PrivateGBDT`
      * `split_candidate_manager.py` - Manages the various methods used to propose split candidates
      * `train_monitor.py` - Monitors various training statistics of a `PrivateGBDT` model
    * `private_gbdt.py` - Contains the main model `PrivateGBDT` class 

## Paper Experiments, Plots and Table

All experiments were run with 15 iterations in total (3 iterations over 5 different train-test sets). Code for running experiments is in `experiments/paper_experiments/paper_experiments.py` and plotting in `experiments/paper_experiments/paper_plotter.py`

The following methods in `paper_experiments.py` corresponds to the following figures/tables in the paper:
* `dp_split_methods_with_update_methods` - Corresponds to Figure 1 (a,b,c), Table 2 in main text, Figures 7-10 and Table 7-10 in the Appendix
* `dp_split_candidate_methods` - Corresponds to Figure 2 (a,b,c) and Table 3 in the main text, Figure 11 and 12 in the Appendix
* `feature_interaction_experiments` - Corresponds to Figure 3
* `dp_ebm_experiment` - Corresponds to Figure 4
* `batched_boosting` -  Corresponds to Figure 5, Table 4 in the main text, Figure 13 in the Appendix
* `comparisons_experiment` - Corresponds to Figure 6 in the main text and Figures 14-18 in the Appendix

The associated figures are already generated and present in `experiments/paper_experiments/paper_plots`. To recreate them the following methods in `experiment_plotter.py` can be used:
* `plot_split_methods_with_update` - Figure 1(a,b,c)
* `plot_split_candidates` - Figure 2(a,b,c)
* `plot_k_way` - Figure 3
* `plot_ebm_comparisons` - Figure 4
* `plot_low_eps_bb` - Figure 5
* `plot_comparisons` - Figure 6
* `table_split_methods_with_update` - Table 2
* `table_split_candidate` - Table 3
* `table_low_eps_bb` - Table 4

Plots and tables for the Appendix can be recreated via the following (although they are already present in `paper_plots`):
* `appendix_E1` - Figures 7,8,9,10
* `appendix_E1_table` - Tables 7,8,9,10
* `appendix_E2` - Figure 12
* `appendix_E3` - Not used
* `appendix_E4` - Figure 13
* `appendix_E5`- Figure 14, 15, 16, 17, 18

## Replication Instructions

As mentioned above all plots and figures from the main paper and appendix can be replicated from the .csv files by running the appropriate methods in `paper_plotter.py`

As all experiments in the paper are repeated over 15 iterations they are usually too slow to replicate within a reasonable amount of time. 
Instead, to roughly replicate an experiment from scratch additional code is provided in `experiments/replication_experiments` and in particular the `experiment_replicator.py` file

This contains an `ExperimentReplicator` class with a `replicate(figure_num)` method. Most replication experiments have been designed to run on the Credit 1 dataset in ~30 minutes depending on the device. Most run on a single test-train seed over 3 iterations.

The `experiments/replication_experiments` folder already contains data and replication figures for all 6 figures presented in the main paper.
You can also generate appendix figures by changing the dataset that is passed to `ExperimentReplicator.replicate`


Benchmark replication times performed on a Macbook Air M1:
* Fig 1(a,b,c): ~30 mins
* Fig 2(a,b,c): ~45 mins
* Fig 3: ~20 mins
* Fig 4: ~10 mins
* Fig 5: ~15 mins
* Fig 6: ~25 mins

## Acknowledgements

* Part of the tree structure implementation is based on the public implementation of the FEVERLESS [paper](https://paperswithcode.com/paper/feverless-fast-and-secure-vertical-federated) with code repo [here](https://github.com/feverless111/vfl/blob/0c0bae50c37c193938e59a95c67fa62b43e43e8e/FEVERLESS/models/vertical/tree/xgboost/centralized_xgboost.py)
* We make extensive use of the [autodp](https://github.com/yuxiangw/autodp) library by Yu-Xiang Wang to verify privacy accounting 
* Part of our privacy accountant uses the RDP moments accountant implemented in [TensorFlow Privacy](https://github.com/tensorflow/privacy)
* Although not used in our paper, the code supports using datasets from the [Penn Machine Learning Benchmarks (PMLB)](https://epistasislab.github.io/pmlb/) 
