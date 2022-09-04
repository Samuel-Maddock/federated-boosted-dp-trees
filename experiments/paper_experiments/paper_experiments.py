import math
import numpy as np

from experiments.experiment_helpers.data_loader import DataLoader
from experiments.experiment_helpers.experiment_runner import ExperimentRunner

from dev.communication_framework import CommunicationsFramework

global_seeds = [1, 4, 100, 333, 1002]
data_loader = DataLoader([1, 4, 100, 333, 1002])
replication_path = "../replication_experiments/replication_data/"


# =================== Paper Experiments ===================


# =================== E1 + E2 - Split Methods + Weight Updates ===================

# Exp 1 - Split methods + Weight updates
    # Corresponds to Figure 1 (a,b,c), Table 2 in main text, Figures 7-10 and Table 7-10 in the Appendix
def dp_split_methods_with_update_methods(save_data=False, filename="dp_split_methods_with_update1", replication=False, iters=3, datasets=None, seeds=None):
    if not replication:
        data_loader = DataLoader(global_seeds)
        datasets = data_loader.load_datasets(["Credit 1", "Credit 2", "adult", "Bank", "nomao"], remove_missing=True, return_dict=True, verbose=True).items()
        iters = 3

        num_trees = [5, 10, 25, 50, 100, 150, 200, 250, 300]
        hist_sizes=[32]
        depths = [2,3,4,5,6]
        epsilons = [0.1, 0.25, 0.5, 0.75, 1]
        update_methods = ["rf", "xgboost", "gbm"]
    else:
        iters = iters
        data_loader = DataLoader(seeds)
        datasets = data_loader.load_datasets(datasets, remove_missing=True, return_dict=True, verbose=True).items()
        num_trees = [5, 10, 25, 50, 100, 150, 200, 250, 300]
        hist_sizes=[32]
        depths = [2,3,4,5,6]
        epsilons = [0.1, 0.25, 0.5, 0.75, 1]
        update_methods = ["xgboost"]

    arg_options = [{"dp_method": "gaussian_cdp", "split_method": "totally_random", "gradient_budgets": "vector_mechanism"},
                   {"dp_method": "gaussian_cdp", "split_method": "partially_random", "gradient_budgets": "vector_mechanism"},
                   {"dp_method": "gaussian_cdp", "split_method": "hist_based", "gradient_budgets": "vector_mechanism"},]

    new_arg_options = []
    for d in depths:
        for T in num_trees:
            for update_method in update_methods:
                for args in arg_options:
                    new_args = args.copy()
                    new_args["max_depth"] = d
                    new_args["num_trees"] = T
                    new_args["track_budget"] = False
                    new_args["verbose"] = False
                    if update_method == "rf":
                        new_args["training_method"] = "rf"
                    else:
                        new_args["weight_update_method"] = update_method
                    new_arg_options.append(new_args)

    if replication:
        exp = ExperimentRunner(performance=True, data_path=replication_path)
        exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("epsilon", epsilons))
    else:
        exp = ExperimentRunner(performance=True)
        exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("epsilon", epsilons))

# Exp 2 - Non-DP Split methods + Weight updates
    # Not used in the paper
def non_dp_split_methods_with_update_methods():
    exp = ExperimentRunner()
    datasets = data_loader.load_datasets(["Credit 1", "Credit 2", "adult", "Bank", "nomao"], remove_missing=True, return_dict=True, verbose=True).items()
    iters = 3

    num_trees = [5, 10, 25, 50, 100, 150, 200, 250, 300]
    hist_sizes=[32]
    depths = [2,3,4,5,6]
    epsilons = [0]

    # Testing args
    # depths=[2]
    # epsilons=[0.5]
    # num_trees=[50]
    # iters=1
    # datasets = data_loader.load_datasets(["Credit 1"], remove_missing=True, return_dict=True, verbose=True).items()

    update_methods = ["rf", "xgboost", "gbm"]

    arg_options = [{"dp_method": "", "split_method": "totally_random",},
                   {"dp_method": "", "split_method": "partially_random",},
                   {"dp_method": "", "split_method": "hist_based",},]

    new_arg_options = []
    for d in depths:
        for T in num_trees:
            for update_method in update_methods:
                for args in arg_options:
                    new_args = args.copy()
                    new_args["max_depth"] = d
                    new_args["num_trees"] = T
                    new_args["track_budget"] = False
                    new_args["verbose"] = False
                    if update_method == "rf":
                        new_args["training_method"] = "rf"
                    else:
                        new_args["weight_update_method"] = update_method
                    new_arg_options.append(new_args)

    exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=False, filename="non_dp_split_methods_with_update1", iterated_param=("epsilon", epsilons))

# Exp 3 - Gradient Budget Allocation
    # Not used in the paper
def gradient_budget_alloc():
    exp = ExperimentRunner()
    datasets = data_loader.load_datasets(["Credit 1", "Credit 2", "adult", "Bank", "nomao"], remove_missing=True, return_dict=True, verbose=True).items()
    iters = 3

    num_trees = [100, 200, 300]
    hist_sizes=[32]
    depths = [2,4,6]
    epsilons = [0.1, 0.25, 0.5, 0.75, 1]

    # Testing args
    # depths=[2]
    # epsilons=[0.5]
    # num_trees=[50]
    # iters=1
    # datasets = data_loader.load_datasets(["Credit 1"], remove_missing=True, return_dict=True, verbose=True).items()

    update_methods = ["rf", "xgboost", "gbm"]

    arg_options = [{"dp_method": "gaussian_cdp", "split_method": "hist_based", "gradient_budgets": "vector_mechanism"},
                   {"dp_method": "gaussian_cdp", "split_method": "hist_based",  "gradient_budgets": [0.9, 0.1]},
                   {"dp_method": "gaussian_cdp", "split_method": "totally_random", "gradient_budgets": "vector_mechanism"},
                   {"dp_method": "gaussian_cdp", "split_method": "totally_random", "gradient_budgets": [0.9, 0.1]},]

    new_arg_options = []
    for d in depths:
        for T in num_trees:
            for update_method in update_methods:
                for args in arg_options:
                    new_args = args.copy()
                    new_args["max_depth"] = d

                    if new_args["split_method"] == "hist_based":
                        new_args["num_trees"] = T//10
                    else:
                        new_args["num_trees"] = T

                    new_args["track_budget"] = False
                    new_args["verbose"] = False
                    if update_method == "rf":
                        new_args["training_method"] = "rf"
                    else:
                        new_args["weight_update_method"] = update_method
                    new_arg_options.append(new_args)

    exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=False, filename="gradient_budget_alloc1", iterated_param=("epsilon", epsilons))


# =================== E3 - Split Candidate Methods ===================

# Exp 4 - Split candidate methods
    # Corresponds to Figure 2 (a,b,c) and Table 3 in the main text, Figure 11 and 12 in the Appendix
def dp_split_candidate_methods(save_data=False, filename="split_candidate_methods1", replication=False, iters=3, datasets=None, seeds=None):

    if not replication:
        data_loader = DataLoader(global_seeds)
        datasets = data_loader.load_datasets(["Credit 1", "Credit 2", "adult", "nomao", "Bank", "higgs_sampled"], remove_missing=True, return_dict=True, verbose=True).items()
        iters = 3
        num_trees = [100, 200, 300]
        depths = [4]
        hist_sizes = [2, 4, 8, 16, 32, 64, 128]
        epsilons = [0.1, 0.25, 0.5, 0.75, 1]
        sketch_rounds = [5, 10, 20, 30, float("inf")]
        hist_methods = ["uniform", "log", "exact_quantiles", "feverless", "adaptive_hessian"]
    else:
        data_loader = DataLoader(seeds)
        iters = iters

        datasets = data_loader.load_datasets(datasets, remove_missing=True, return_dict=True, verbose=True).items()
        num_trees = [100, 200, 300]
        depths = [4]
        hist_sizes = [2, 4, 8, 16, 32, 64, 128]
        epsilons = [0.1, 0.25, 0.5, 0.75, 1]
        sketch_rounds = [5, 10, 20, 30, float("inf")]
        hist_methods = ["uniform", "log", "exact_quantiles", "feverless", "adaptive_hessian"]

    arg_options = [{"dp_method": "gaussian_cdp", "split_method": "totally_random",},]

    new_arg_options = []
    for d in depths:
        for T in num_trees:
                for h in hist_methods:
                    for args in arg_options:
                        new_args = args.copy()
                        new_args["max_depth"] = d
                        if new_args["split_method"] == "hist_based" and new_args.get("feature_interaction_method") == None:
                            new_args["num_trees"] = T//10
                        else:
                            new_args["num_trees"] = T

                        new_args["verbose"] = False
                        new_args["track_budget"] = False
                        new_args["sketch_type"] = h

                        if (new_args["split_method"] == "hist_based" or new_args["split_method"] == "totally_random") and new_args.get("feature_interaction_method") == None and h=="adaptive_hessian":
                            for s in sketch_rounds:
                                new_new_args = new_args.copy()
                                new_new_args["sketch_rounds"] = s
                                new_arg_options.append(new_new_args)
                        else:
                            new_arg_options.append(new_args)

    if replication:
        exp = ExperimentRunner(performance=True, data_path=replication_path)
        exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("epsilon", epsilons))
    else:
        exp = ExperimentRunner(performance=False)
        exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("epsilon", epsilons))

# Not used in the paper
def non_dp_split_candidate_methods():
    exp = ExperimentRunner()
    datasets = data_loader.load_datasets(["Credit 1", "Credit 2", "adult", "nomao", "Bank", "higgs_sampled"], remove_missing=True, return_dict=True, verbose=True).items()

    iters = 1
    num_trees = [100,200,300]
    depths = [3,4]
    hist_sizes = [2, 4, 8, 16, 32, 64, 128]
    epsilons = [0]
    sketch_rounds = [5, 10, 20, 30, float("inf")]

    hist_methods = ["uniform", "log", "exact_quantiles", "feverless", "adaptive_hessian"]

    arg_options = [{"dp_method": "gaussian_cdp", "split_method": "totally_random",},]

    # arg_options = [{"dp_method": "", "split_method": "totally_random",},
    #                {"dp_method": "", "split_method": "totally_random",
    #                 "feature_interaction_method": "cyclical", "feature_interaction_k": 1},
    #                {"dp_method": "", "split_method": "hist_based",},
    #                {"dp_method": "", "split_method": "hist_based",
    #                 "feature_interaction_method": "cyclical", "feature_interaction_k": 1},]

    new_arg_options = []
    for d in depths:
        for T in num_trees:
                for h in hist_methods:
                    for args in arg_options:
                        new_args = args.copy()
                        new_args["max_depth"] = d
                        if new_args["split_method"] == "hist_based" and new_args.get("feature_interaction_method") == None:
                            new_args["num_trees"] = T//10
                        else:
                            new_args["num_trees"] = T
                        new_args["track_budget"] = False
                        new_args["sketch_type"] = h

                        if (new_args["split_method"] == "hist_based" or new_args["split_method"] == "totally_random") and new_args.get("feature_interaction_method") == None and h=="adaptive_hessian":
                            for s in sketch_rounds:
                                new_new_args = new_args.copy()
                                new_new_args["sketch_rounds"] = s
                                new_arg_options.append(new_new_args)
                        else:
                            new_arg_options.append(new_args)

    exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=False, filename="non_dp_split_candidate_methods1", iterated_param=("epsilon", epsilons))


# =================== E4 - Feature Interactions ===================

# Exp 5 - k-way methods
    # Corresponds to Figure 3
def feature_interaction_experiments(save_data=False, filename="k_way", replication=False, iters=3, datasets=None, seeds=None):
    if not replication:
        data_loader = DataLoader(global_seeds)
        datasets = data_loader.load_datasets(["Credit 1", "Credit 2", "adult", "Bank", "nomao"], remove_missing=True, return_dict=True, verbose=True).items()
        iters = 3
        num_trees = [10, 25, 50, 75, 100, 200, 300]
        depths = [2,3,4]
        hist_sizes=[32]
        epsilons = [0.01, 0.05, 0.1, 0.5, 1]
        feature_k =[1,2,3,4,5, None]
        feature_interaction_methods = ["cyclical", "random"]
    else:
        data_loader = DataLoader(seeds)
        datasets = data_loader.load_datasets(datasets, remove_missing=True, return_dict=True, verbose=True).items()
        iters = iters
        num_trees = [10, 25, 50, 75, 100, 200, 300]
        depths = [4]
        hist_sizes=[32]
        epsilons = [1]
        feature_k =[1,2,3,4,5, None]
        feature_interaction_methods = ["cyclical", "random"]


    arg_options = [# Standard TR-XGBoost method with vec mech
                   {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "gradient_budgets": "vector_mechanism",
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "xgboost"},

                    # TR-XGBoost with GBM update
                   {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "gradient_budgets": [0.9, 0.1],
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "gbm"},

                       {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "gradient_budgets": "vector_mechanism",
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "gbm"},]


                   #  # Hist based w/ vec mech - xgboost update and gbm update
                   # {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "hist_based", "gradient_budgets": "vector_mechanism",
                   #   "sketch_type": "uniform", "track_budget": False, "weight_update_method": "xgboost"},
                   #
                   # {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "hist_based", "gradient_budgets": "vector_mechanism",
                   #   "sketch_type": "uniform", "track_budget": False, "weight_update_method": "gbm"},]

    arg_options = arg_options
    epsilons = [1]

    new_arg_options = []
    for d in depths:
        for eps in epsilons:
            for args in arg_options:
                new_args = args.copy()
                new_args["max_depth"] = d
                new_args["epsilon"] = eps
                new_args["track_budget"] = False
                new_args["verbose"] = False

                for method in feature_interaction_methods:
                    for k in feature_k:
                        new_new_args = new_args.copy()
                        new_new_args["feature_interaction_method"] = method
                        new_new_args["feature_interaction_k"] = k

                        new_arg_options.append(new_new_args)

    if replication:
        exp = ExperimentRunner(performance=True, data_path=replication_path)
        exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("num_trees", num_trees))
    else:
        exp = ExperimentRunner(performance=False)
        exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("num_trees", num_trees))

# Exp 4 - EBM comparisons
    # Corresponds to Figure 4
def dp_ebm_experiment(save_data=False, filename="dp_ebm_experiment", replication=False, iters=3, datasets=None, seeds=None):
    if not replication:
        data_loader = DataLoader(global_seeds)
        datasets = data_loader.load_datasets(["Credit 1", "Credit 2", "adult", "Bank", "nomao"], remove_missing=True, return_dict=True, verbose=True).items()
        iters = 3
        num_trees = [10, 25, 50, 75, 100, 200, 300]
        depths = [2,3,4]
        hist_sizes=[32]
        epsilons = [0.01, 0.05, 0.1, 0.5, 1]
    else:
        data_loader = DataLoader(seeds)
        datasets = data_loader.load_datasets(datasets, remove_missing=True, return_dict=True, verbose=True).items()
        iters = iters
        num_trees = [10, 25, 50, 75, 100, 200, 300]
        depths = [4]
        hist_sizes=[32]
        epsilons = [0.01, 0.05, 0.1, 0.5, 1]

    arg_options = [
                   # Standard TR-XGBoost method with vec mech
                   {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "gradient_budgets": "vector_mechanism",
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "xgboost"},

                    # TR-XGBoost with GBM update
                   {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "gradient_budgets": [0.9, 0.1],
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "gbm"},
                       {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "gradient_budgets": "vector_mechanism",
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "gbm"},

                    # EBM and EBM XGBoost
                   {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "gradient_budgets": [0.9, 0.1],
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "gbm", "feature_interaction_method": "cyclical", "feature_interaction_k": 1},
                   {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "gradient_budgets": "vector_mechanism",
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "xgboost", "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # Hist based w/ vec mech - xgboost update and gbm update
                   {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "hist_based", "gradient_budgets": "vector_mechanism",
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "xgboost", "feature_interaction_method": "cyclical", "feature_interaction_k": 1},
                   {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "hist_based", "gradient_budgets": "vector_mechanism",
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "gbm", "feature_interaction_method": "cyclical", "feature_interaction_k": 1},]

    arg_options = arg_options
    epsilons = [1]

    new_arg_options = []
    for d in depths:
        for eps in epsilons:
            for args in arg_options:
                new_args = args.copy()
                new_args["max_depth"] = d
                new_args["epsilon"] = eps

                new_args["track_budget"] = False
                new_args["verbose"] = False

                new_arg_options.append(new_args)

    if replication:
        exp = ExperimentRunner(performance=True, data_path=replication_path)
        exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("num_trees", num_trees))
    else:
        exp = ExperimentRunner(performance=False)
        exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("num_trees", num_trees))

# Not used in the paper
def non_dp_ebm_experiment():
    exp = ExperimentRunner()
    datasets = data_loader.load_datasets(["Credit 1", "Credit 2", "adult", "nomao", "Bank"], remove_missing=True, return_dict=True, verbose=True).items()

    iters = 3
    num_trees = [10, 25, 50, 75, 100, 200, 300]
    depths = [2,3,4]
    hist_sizes=[32]
    epsilons = [0.01, 0.05, 0.1, 0.5, 1]

    # Test params
    # depths = [6]
    # epsilons = [0.01]
    # num_trees = [10]
    # iters = 1

    non_dp_arg_options = [{"dp_method": "", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "xgboost",},
                   {"dp_method": "", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "gbm",},
                   {"dp_method": "", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "gbm", "feature_interaction_method": "cyclical", "feature_interaction_k": 1},
                   {"dp_method": "", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "xgboost", "feature_interaction_method": "cyclical", "feature_interaction_k": 1},]

    arg_options = non_dp_arg_options
    epsilons = [1]

    new_arg_options = []
    for d in depths:
        for eps in epsilons:
            for args in arg_options:
                new_args = args.copy()
                new_args["max_depth"] = d
                new_args["epsilon"] = eps
                new_arg_options.append(new_args)


    exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=False, filename="non_dp_ebm_experiment", iterated_param=("num_trees", num_trees))


# =================== E5 - Batched Updates ===================

# Exp 6 - Batched Updates
    # Corresponds to Figure 5, Table 4 in the main text, Figure 13 in the Appendix
def batched_boosting(save_data=False, filename="rf_boosting1", replication=False, iters=3, datasets=None, seeds=None):

    if not replication:
        data_loader = DataLoader(global_seeds)
        datasets = data_loader.load_datasets(["Credit 1", "Credit 2", "adult", "Bank", "nomao"], remove_missing=True, return_dict=True, verbose=True).items()
        iters = 3
        num_trees = [100, 200, 300]
        depths = [2, 4]
        hist_sizes = [32]
        epsilons = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        rf_sizes = [0.05, 0.1, 0.25, 0.34, 0.5, 0.75, 1]
    else:
        data_loader = DataLoader(seeds)
        datasets = data_loader.load_datasets(datasets, remove_missing=True, return_dict=True, verbose=True).items()
        iters = iters
        num_trees = [100, 200, 300]
        depths = [4]
        hist_sizes = [32]
        epsilons = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        rf_sizes = [0.05, 0.1, 0.25, 0.34, 0.5, 0.75, 1]

    arg_options = [
                    # EBM TR-GBM and EBM TR-XGBoost
                   {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "gbm", "feature_interaction_method": "cyclical", "feature_interaction_k": 1,
                    "training_method": "boosting"},

                   {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "xgboost", "feature_interaction_method": "cyclical", "feature_interaction_k": 1,
                    "training_method": "boosting"},

                    # DP-TR-XGBoost + GBM without EBM
                   {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "xgboost",
                    "training_method": "boosting"},

                   {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "sketch_type": "uniform", "track_budget": False, "weight_update_method": "gbm",
                    "training_method": "boosting"},

                   #  # Vector mech EBM + non EBM TR-XGBoost
                   # {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                   #   "sketch_type": "uniform", "track_budget": False, "weight_update_method": "xgboost", "training_method": "boosting",
                   #   "gradient_budgets":"vector_mechanism"},
                   # {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                   #   "sketch_type": "uniform", "track_budget": False, "weight_update_method": "xgboost", "feature_interaction_method": "cyclical", "feature_interaction_k": 1,
                   #  "training_method": "boosting", "gradient_budgets":"vector_mechanism"},

                    # Normal and EBM RF
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    "sketch_type": "uniform", "track_budget": False, "verbose": False, "training_method": "rf"},
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    "sketch_type": "uniform", "track_budget": False, "verbose": False, "training_method": "rf" , "feature_interaction_method": "cyclical", "feature_interaction_k": 1,},

                    # RF boosting
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    "sketch_type": "uniform", "track_budget": False, "verbose": False, "training_method": "batched_boosting", "batched_update_size": 30},
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    "sketch_type": "uniform", "track_budget": False, "verbose": False, "training_method": "batched_boosting", "batched_update_size": 30, "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    # "sketch_type": "uniform", "track_budget": False, "verbose": False, "training_method": "batched_boosting", "batched_update_size": 30,
                    # "gradient_budgets": "vector_mechanism"},
                    # {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    # "sketch_type": "uniform", "track_budget": False, "verbose": False, "training_method": "batched_boosting", "batched_update_size": 30, "feature_interaction_method": "cyclical", "feature_interaction_k": 1,
                    # "gradient_budgets": "vector_mechanism"},
                    ]

    new_arg_options = []
    for d in depths:
        for T in num_trees:
            for args in arg_options:
                new_args = args.copy()
                new_args["max_depth"] = d
                new_args["num_trees"] = T

                new_args["verbose"] = False
                new_args["track_budget"] = False

                new_args["gradient_budgets"] = "vector_mechanism" ###

                if new_args["training_method"] == "batched_boosting":
                    for rf_size in rf_sizes:
                        copy_args = new_args.copy()
                        copy_args["batched_update_size"] = math.ceil(rf_size*copy_args["num_trees"])
                        new_arg_options.append(copy_args)
                else:
                    new_arg_options.append(new_args)

    if replication:
        exp = ExperimentRunner(performance=True, data_path=replication_path)
        exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("epsilon", epsilons))
    else:
        exp = ExperimentRunner(performance=False)
        exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("epsilon", epsilons))


# =================== E6 - Comparisons ===================

# Exp - Comparisons

# Non-private XGBoost line is ued in Figure 6 and Figures 14-18
def non_dp_comparisons_experiment():
    exp = ExperimentRunner()
    datasets = data_loader.load_datasets(["Credit 1", "Credit 2", "adult", "nomao", "Bank"], remove_missing=True, return_dict=True, verbose=True).items()
    # datasets = data_loader.load_datasets(["Credit 1", "Credit 2", "adult"], remove_missing=True, return_dict=True, verbose=True).items()
    # datasets = data_loader.load_datasets(["higgs_sampled", "nomao", "Bank"], remove_missing=True, return_dict=True, verbose=True).items()

    iters = 3

    num_trees = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    depths = [2, 3, 4]
    hist_sizes = [32]
    # epsilons = [0.1, 0.5, 1]
    epsilons = [0]

    # rf_sizes = [0.05, 0.1, 0.25, 0.5, 1]
    rf_sizes = [0.25, 1]

    # Test params
    # depths = [6]
    # epsilons = [1]
    # num_trees = [100]
    # iters = 1

    arg_options = [
                    # FEVERLESS (uniform)
                    {"dp_method": "", "split_method": "hist_based", "weight_update_method": "xgboost",
                     "sketch_type": "uniform"},

                    # FEVERLESS (quantiles)
                    {"dp_method": "", "split_method": "hist_based", "weight_update_method": "xgboost",
                     "sketch_type": "exact_quantiles"},

                    # DP-RF
                    {"dp_method": "",  "split_method": "totally_random", "training_method": "rf"},

                    # DP-EBM (full cycle, GBM)
                    {"dp_method": "", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "gbm",
                     "full_ebm": True, "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-EBM XGBoost (no full cycle)
                    {"dp_method": "", "split_method": "totally_random", "weight_update_method": "xgboost",
                     "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-TR XGBoost
                    {"dp_method": "", "split_method": "totally_random", "weight_update_method": "xgboost",},]

    new_arg_options = []
    for d in depths:
        for T in num_trees:
            for args in arg_options:
                new_args = args.copy()
                new_args["max_depth"] = d
                new_args["num_trees"] = T

                new_args["track_budget"] = False
                new_args["verbose"] = False

                if new_args.get("training_method") == "batched_boosting":
                    for rf_size in rf_sizes:
                        copy_args = new_args.copy()
                        copy_args["batched_update_size"] = math.ceil(rf_size*copy_args["num_trees"])
                        new_arg_options.append(copy_args)
                else:
                    new_arg_options.append(new_args)

    exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=False, filename="non_dp_comparisons", iterated_param=("epsilon", epsilons))

# Corresponds to Figure 6 in the main text and Figures 14-18 in the Appendix
def comparisons_experiment(save_data=False, filename="comparisons1", replication=False, iters=3, datasets=None, seeds=None):
    if not replication:
        data_loader = DataLoader(global_seeds)
        datasets = data_loader.load_datasets(["Credit 1", "Credit 2", "adult", "nomao", "Bank", "higgs_sampled"], remove_missing=True, return_dict=True, verbose=True).items()
        iters = 3
        num_trees = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
        depths = [4]
        hist_sizes = [32]
        epsilons = [0.1, 0.5, 1]
        rf_sizes = [0.25, 1]
    else:
        data_loader = DataLoader(seeds)
        datasets = data_loader.load_datasets(datasets, remove_missing=True, return_dict=True, verbose=True).items()
        iters = iters
        num_trees = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
        depths = [4]
        hist_sizes = [32]
        epsilons = [0.1, 0.5, 1]
        rf_sizes = [0.25, 1]

    arg_options = [
                    # FEVERLESS (private)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "hist_based", "weight_update_method": "xgboost",
                     "sketch_type": "uniform"},

                    # DP-GBM
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "hist_based", "weight_update_method": "gbm"},

                    # DP-RF
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "training_method": "rf"},

                    # DP-EBM (full cycle, GBM)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "gbm",
                     "full_ebm": True, "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-EBM XGBoost (no full cycle)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",
                     "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-TR XGBoost
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",},

                    # DP-TR XGBoost w/ IH
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",
                     "sketch_type": "adaptive_hessian", "sketch_rounds": 5},

                    # DP-TR XGBoost EBM w/IH
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    "sketch_type": "uniform", "weight_update_method": "xgboost", "sketch_type": "adaptive_hessian", "sketch_rounds": 5,
                    "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-TR Batched boosting
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    "sketch_type": "uniform", "training_method": "batched_boosting", "batched_update_size": 30},

                    # DP-TR Batched Boosting + IH (s=5)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "training_method": "batched_boosting", "batched_update_size": 30, "sketch_type": "adaptive_hessian", "sketch_rounds": 5},

                    # DP-TR Batched Boosting + IH (s=5) + EBM
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "training_method": "batched_boosting", "batched_update_size": 30, "sketch_type": "adaptive_hessian", "sketch_rounds": 5,
                     "feature_interaction_method": "cyclical", "feature_interaction_k": 1},
                    ]


    new_arg_options = []
    for d in depths:
        for T in num_trees:
            for args in arg_options:
                new_args = args.copy()
                new_args["max_depth"] = d
                new_args["num_trees"] = T
                new_args["verbose"] = False
                new_args["track_budget"] = False

                new_args["gradient_budgets"] = "vector_mechanism" ###

                if new_args.get("training_method") == "batched_boosting":
                    for rf_size in rf_sizes:
                        copy_args = new_args.copy()
                        copy_args["batched_update_size"] = math.ceil(rf_size*copy_args["num_trees"])
                        new_arg_options.append(copy_args)
                else:
                    new_arg_options.append(new_args)

    if replication:
        exp = ExperimentRunner(performance=True, data_path=replication_path)
        exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("epsilon", epsilons))
    else:
        exp = ExperimentRunner(performance=False)
        exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("epsilon", epsilons))

# Not used in the paper
def full_comparisons_experiment():
    exp = ExperimentRunner()
    # datasets = data_loader.load_datasets(["Credit 1", "Credit 2", "adult", "nomao", "Bank"], remove_missing=True, return_dict=True, verbose=True).items()
    datasets = data_loader.load_datasets(["Credit 1", "Credit 2", "adult"], remove_missing=True, return_dict=True, verbose=True).items()

    iters = 1

    num_trees = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    depths = [2, 3, 4]
    hist_sizes = [32]
    epsilons = [0.1, 0.5, 1]

    rf_sizes = [0.05, 0.1, 0.25, 0.5, 1]

    # Test params
    # depths = [6]
    # epsilons = [1]
    # num_trees = [100]
    # iters = 1

    arg_options = [ # FEVERLESS (non-private)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "hist_based", "weight_update_method": "xgboost",
                     "sketch_type": "feverless"},

                    # FEVERLESS (private)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "hist_based", "weight_update_method": "xgboost",
                     "sketch_type": "uniform"},

                    # DP-GBM
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "hist_based", "weight_update_method": "gbm"},

                    # DP-RF
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "training_method": "rf"},

                    # DP-EBM (full cycle, GBM)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "gbm",
                     "full_ebm": True, "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-EBM XGBoost (no full cycle)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",
                     "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-XGBoost 2-way
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",
                     "feature_interaction_method": "cyclical", "feature_interaction_k": 2},

                    # DP-XGBoost 2-way random
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",
                     "feature_interaction_method": "random", "feature_interaction_k": 2},

                    # DP-XGBoost 5-way
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",
                     "feature_interaction_method": "random", "feature_interaction_k": 5},

                    # DP-TR XGBoost
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",},

                    # DP-TR XGBoost w/ IH
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",
                     "sketch_type": "adaptive_hessian", "sketch_rounds": 5},

                    # DP-TR Batched boosting
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    "sketch_type": "uniform", "track_budget": False, "verbose": False, "training_method": "batched_boosting", "batched_update_size": 30},

                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    "sketch_type": "uniform", "track_budget": False, "verbose": False, "training_method": "batched_boosting", "batched_update_size": 30, "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-TR Batched Boosting + IH (s=5)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    "sketch_type": "uniform", "track_budget": False, "verbose": False,
                     "training_method": "batched_boosting", "batched_update_size": 30, "sketch_type": "adaptive_hessian", "sketch_rounds": 5},
                    ]

    new_arg_options = []
    for d in depths:
        for T in num_trees:
            for args in arg_options:
                new_args = args.copy()
                new_args["max_depth"] = d
                new_args["num_trees"] = T

                if new_args.get("training_method") == "batched_boosting":
                    for rf_size in rf_sizes:
                        copy_args = new_args.copy()
                        copy_args["batched_update_size"] = math.ceil(rf_size*copy_args["num_trees"])
                        new_arg_options.append(copy_args)
                else:
                    new_arg_options.append(new_args)

    exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=False, filename="comparisons", iterated_param=("epsilon", epsilons))

# =================== Revisions ===================

# LDP Baselines
def ldp_comparisons(save_data=False, filename="ldp", iters=3, datasets=None, seeds=None):
    data_loader = DataLoader(global_seeds)
    datasets = data_loader.load_datasets(["Credit 1", "Credit 2", "adult", "nomao", "Bank", "higgs_sampled"], remove_missing=True, return_dict=True, verbose=True).items()
    num_trees = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    epsilons = [0.1, 0.5, 1]
    depths = [4]
    hist_sizes = [32]
    rf_sizes = []

    arg_options = [
                    # LDP Hist based
                    {"dp_method": "gaussian_ldp", "accounting_method": "rdp_scaled_improved", "split_method": "hist_based", "weight_update_method": "xgboost",
                     "sketch_type": "uniform"},

                    # LDP TR
                    {"dp_method": "gaussian_ldp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",
                     "sketch_type": "uniform"}, ]

    # arg_options = [
    #                 # Testing TR
    #                 # DP-TR Batched boosting
    #                 {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
    #                 "sketch_type": "uniform", "weight_update_method": "xgboost", "sketch_type": "adaptive_hessian", "sketch_rounds": 5,
    #                 "feature_interaction_method": "cyclical", "feature_interaction_k": 1}]

    new_arg_options = []

    for d in depths:
        for T in num_trees:
            for args in arg_options:
                new_args = args.copy()
                new_args["max_depth"] = d
                new_args["num_trees"] = T
                new_args["verbose"] = False
                new_args["track_budget"] = False

                new_args["gradient_budgets"] = "vector_mechanism" ###

                if new_args.get("training_method") == "batched_boosting":
                    for rf_size in rf_sizes:
                        copy_args = new_args.copy()
                        copy_args["batched_update_size"] = math.ceil(rf_size*copy_args["num_trees"])
                        new_arg_options.append(copy_args)
                else:
                    new_arg_options.append(new_args)

    exp = ExperimentRunner(performance=True, output_train_monitor=False)
    exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("epsilon", epsilons), n_jobs=4)

# Synthetic exp test
def synthetic_test_exp(save_data=False, filename="synthetic_test", iters=3, datasets=None, seeds=None):
    data_loader = DataLoader([global_seeds[0]])

    num_features = [30]
    num_samples = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    num_informative = [26]

    datasets = {}
    for n in num_samples:
        for m in num_features:
            for informative in num_informative:
                datasets = datasets | data_loader.load_datasets(["synthetic_n=" + str(n) + "_m=" + str(m) + "_informative=" + str(informative)], remove_missing=True, return_dict=True, verbose=True)

    datasets = datasets.items()

    num_trees = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    depths = [4]
    hist_sizes = [32]
    epsilons = [1]
    rf_sizes = []

    arg_options = [
                    # FEVERLESS (private)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "hist_based", "weight_update_method": "xgboost",
                     "sketch_type": "uniform"},

                    # DP-GBM
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "hist_based", "weight_update_method": "gbm"},

                    # DP-RF
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "training_method": "rf"},

                    # DP-EBM (full cycle, GBM)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "gbm",
                     "full_ebm": True, "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-EBM XGBoost (no full cycle)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",
                     "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # LDP-TR XGBoost
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",},
                    ]

    new_arg_options = []

    for d in depths:
        for T in num_trees:
            for args in arg_options:
                new_args = args.copy()
                new_args["max_depth"] = d
                new_args["num_trees"] = T
                new_args["verbose"] = False
                new_args["track_budget"] = False

                new_args["gradient_budgets"] = "vector_mechanism" ###

                if new_args.get("training_method") == "batched_boosting":
                    for rf_size in rf_sizes:
                        copy_args = new_args.copy()
                        copy_args["batched_update_size"] = math.ceil(rf_size*copy_args["num_trees"])
                        new_arg_options.append(copy_args)
                else:
                    new_arg_options.append(new_args)

    exp = ExperimentRunner(performance=True)
    exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("epsilon", epsilons))

# Synthetic communication
def synthetic_communication(save_data=True, filename="synthetic_comm"):
    data_loader = DataLoader([global_seeds[0]])
    # num_features = [30]
    num_samples = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

    num_samples = [2500]
    num_features = [(10,6), (20, 12), (30, 20), (40, 30), (50, 40)]
    datasets = {}
    for n in num_samples:
        for item in num_features:
            m, informative = item
            datasets = datasets | data_loader.load_datasets(["synthetic_n=" + str(n) + "_m=" + str(m) + "_informative=" + str(informative)], remove_missing=True, return_dict=True, verbose=True)

    datasets = datasets.items()

    iters = 3
    num_trees = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    num_trees = np.arange(100, 1000, 100)
    depths = [3,4,5,6]
    # depths = [4]
    hist_sizes = [4,8,16,32,64,128]
    rf_sizes = [0.25, 1]
    epsilons = [1]

    arg_options = [
                    # FEVERLESS (private)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "hist_based", "weight_update_method": "xgboost",
                     "sketch_type": "uniform"},

                    # DP-TR XGBoost w/ IH
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",
                     "sketch_type": "adaptive_hessian", "sketch_rounds": 5},

                    # DP-TR XGBoost EBM w/IH
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    "sketch_type": "uniform", "weight_update_method": "xgboost", "sketch_type": "adaptive_hessian", "sketch_rounds": 5,
                    "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-TR Batched boosting
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    "sketch_type": "uniform", "training_method": "batched_boosting", "batched_update_size": 30},

                    # DP-TR Batched Boosting + IH (s=5)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "training_method": "batched_boosting", "batched_update_size": 30, "sketch_type": "adaptive_hessian", "sketch_rounds": 5},

                    # DP-TR Batched Boosting + IH (s=5) + EBM
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "training_method": "batched_boosting", "batched_update_size": 30, "sketch_type": "adaptive_hessian", "sketch_rounds": 5,
                     "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-TR XGBoost
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",},
                    ]

    new_arg_options = []

    for d in depths:
        for T in num_trees:
            for q in hist_sizes:
                for args in arg_options:
                    new_args = args.copy()
                    new_args["max_depth"] = d
                    new_args["num_trees"] = T
                    new_args["verbose"] = False
                    new_args["track_budget"] = False
                    # new_args["q"] = q

                    new_args["gradient_budgets"] = "vector_mechanism" ###

                    if new_args.get("training_method") == "batched_boosting":
                        for rf_size in rf_sizes:
                            copy_args = new_args.copy()
                            copy_args["batched_update_size"] = math.ceil(rf_size*copy_args["num_trees"])
                            new_arg_options.append(copy_args)
                    else:
                        new_arg_options.append(new_args)

    exp = ExperimentRunner(performance=True)
    exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("epsilon", epsilons), n_jobs=4)

# Clients
def vary_clients(save_data=True, filename="vary_clients"):
    data_loader = DataLoader([global_seeds[0]])
    # num_features = [30]
    num_samples = [100, 500, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

    # num_samples = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    num_features = [(30, 26)]
    datasets = {}
    for n in num_samples:
        for item in num_features:
            m, informative = item
            datasets = datasets | data_loader.load_datasets(["synthetic_n=" + str(n) + "_m=" + str(m) + "_informative=" + str(informative)], remove_missing=True, return_dict=True, verbose=True)

    datasets = datasets.items()

    iters = 3
    num_trees = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    num_trees = [100]
    depths = [4]
    hist_sizes = [2,4,8,16,32,64,128,256,512]
    rf_sizes = [0.25, 1]
    epsilons = [1]

    arg_options = [
                    # FEVERLESS (private)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "hist_based", "weight_update_method": "xgboost",
                     "sketch_type": "uniform"},

                    # DP-TR XGBoost w/ IH
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",
                     "sketch_type": "adaptive_hessian", "sketch_rounds": 5},

                    # DP-TR XGBoost EBM w/IH
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    "sketch_type": "uniform", "weight_update_method": "xgboost", "sketch_type": "adaptive_hessian", "sketch_rounds": 5,
                    "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-TR Batched boosting
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    "sketch_type": "uniform", "training_method": "batched_boosting", "batched_update_size": 30},

                    # DP-TR Batched Boosting + IH (s=5)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "training_method": "batched_boosting", "batched_update_size": 30, "sketch_type": "adaptive_hessian", "sketch_rounds": 5},

                    # DP-TR Batched Boosting + IH (s=5) + EBM
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "training_method": "batched_boosting", "batched_update_size": 30, "sketch_type": "adaptive_hessian", "sketch_rounds": 5,
                     "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-TR XGBoost
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",},
                    ]


    arg_options = [
                    # DP-TR XGBoost w/ IH
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",
                     "sketch_type": "adaptive_hessian", "sketch_rounds": 5},

                    # DP-TR XGBoost
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",},

                    # XGBoost
                    {"dp_method": "", "split_method": "hist_based", "weight_update_method": "xgboost",
                     "sketch_type": "exact_quantiles"}
                    ]

    new_arg_options = []

    for d in depths:
        for T in num_trees:
            # for q in hist_sizes:
            for args in arg_options:
                new_args = args.copy()
                new_args["max_depth"] = d
                new_args["num_trees"] = T
                new_args["verbose"] = False
                new_args["track_budget"] = False

                new_args["gradient_budgets"] = "vector_mechanism" ###

                if new_args.get("training_method") == "batched_boosting":
                    for rf_size in rf_sizes:
                        copy_args = new_args.copy()
                        copy_args["batched_update_size"] = math.ceil(rf_size*copy_args["num_trees"])
                        new_arg_options.append(copy_args)
                else:
                    new_arg_options.append(new_args)

    exp = ExperimentRunner(performance=True)
    exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("epsilon", epsilons), n_jobs=4)

# Computation benchmarks
def computation_benchmark(save_data=True, filename="computation_benchmark"):
    data_loader = DataLoader([global_seeds[0]])
    # num_features = [30]
    num_samples = [10000]
    num_features = [(10,6), (20, 12), (30, 20), (40, 30), (50, 40)]
    datasets = {}
    for n in num_samples:
        for item in num_features:
            m, informative = item
            datasets = datasets | data_loader.load_datasets(["synthetic_n=" + str(n) + "_m=" + str(m) + "_informative=" + str(informative)], remove_missing=True, return_dict=True, verbose=True)

    datasets = datasets.items()

    iters = 3
    num_trees = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    # num_trees = np.arange(100, 1000, 100)
    depths = [3,4,5]
    hist_sizes = [16,32,64]
    rf_sizes = [0.25, 1]
    epsilons = [1]

    # num_trees = [50]
    # depths = [4]
    # hist_sizes = [32]

    arg_options = [
                    # FEVERLESS (private)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "hist_based", "weight_update_method": "xgboost",
                     "sketch_type": "uniform"},

                    # DP-TR XGBoost w/ IH
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",
                     "sketch_type": "adaptive_hessian", "sketch_rounds": 5},

                    # DP-TR XGBoost EBM w/IH
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    "sketch_type": "uniform", "weight_update_method": "xgboost", "sketch_type": "adaptive_hessian", "sketch_rounds": 5,
                    "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-TR Batched boosting
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                    "sketch_type": "uniform", "training_method": "batched_boosting", "batched_update_size": 30},

                    # DP-TR Batched Boosting + IH (s=5)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "training_method": "batched_boosting", "batched_update_size": 30, "sketch_type": "adaptive_hessian", "sketch_rounds": 5},

                    # DP-TR Batched Boosting + IH (s=5) + EBM
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random",
                     "training_method": "batched_boosting", "batched_update_size": 30, "sketch_type": "adaptive_hessian", "sketch_rounds": 5,
                     "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-EBM XGBoost (no full cycle)
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",
                     "feature_interaction_method": "cyclical", "feature_interaction_k": 1},

                    # DP-TR XGBoost
                    {"dp_method": "gaussian_cdp", "accounting_method": "rdp_scaled_improved", "split_method": "totally_random", "weight_update_method": "xgboost",},
                    ]

    new_arg_options = []

    for d in depths:
        for T in num_trees:
            for args in arg_options:
                new_args = args.copy()
                new_args["max_depth"] = d
                new_args["num_trees"] = T
                new_args["verbose"] = False
                new_args["track_budget"] = False

                new_args["gradient_budgets"] = "vector_mechanism" ###

                if new_args.get("training_method") == "batched_boosting":
                    for rf_size in rf_sizes:
                        copy_args = new_args.copy()
                        copy_args["batched_update_size"] = math.ceil(rf_size*copy_args["num_trees"])
                        new_arg_options.append(copy_args)
                else:
                    new_arg_options.append(new_args)

    exp = ExperimentRunner(performance=True, output_train_monitor=False)
    exp.run(datasets, iters, new_arg_options, hist_sizes, save_data=save_data, filename=filename, iterated_param=("epsilon", epsilons))

# =================== Run Experiments ===================

# dp_split_methods_with_update_methods()
# non_dp_split_methods_with_update_methods()
# gradient_budget_alloc()

# dp_split_candidate_methods()

# non_dp_split_candidate_methods()

# non_dp_ebm_experiment()
# dp_ebm_experiment()
# feature_interaction_experiments()

# batched_boosting()

# feature_interaction_experiments()

# batched_boosting()

# non_dp_comparisons_experiment()
# comparisons_experiment()
# full_comparisons_experiment()

# ldp_comparisons(True)
# ldp_comparisons(True, filename="ldp_tr")

# synthetic_test_exp(True)

# analytical_stats()

# synthetic_communication()

# vary_clients()

# computation_benchmark()

# quantize_test()