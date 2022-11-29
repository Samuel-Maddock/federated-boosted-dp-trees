import math
import random
import sys

import pandas as pd
import numpy as np

from collections import Counter, defaultdict
from copy import copy

from fast_histogram import histogram1d

from federated_gbdt.models.base.tree_base import TreeBase
from federated_gbdt.models.base.tree_node import DecisionNode
from federated_gbdt.models.base.jit_functions import _calculate_gain, _calculate_weight, _L1_clip  # numba funcs

from federated_gbdt.models.gbdt.components.split_candidate_manager import SplitCandidateManager
from federated_gbdt.models.gbdt.components.index_sampler import IndexSampler
from federated_gbdt.models.gbdt.components.privacy_accountant import PrivacyAccountant
from federated_gbdt.models.gbdt.components.train_monitor import TrainMonitor

from federated_gbdt.core.pure_ldp.frequency_oracles.hybrid_mechanism.hybrid_mech_client import HMClient
from federated_gbdt.core.loss_functions import SigmoidBinaryCrossEntropyLoss, BinaryRFLoss, SoftmaxCrossEntropyLoss

from sklearn.preprocessing import LabelBinarizer

class PrivateGBDT(TreeBase):

    def __init__(self, num_trees=2, max_depth=6, # Default tree params
                 task_type="classification", loss=SigmoidBinaryCrossEntropyLoss(),
                 reg_lambda=1, reg_alpha=0, reg_gamma=1e-7, reg_eta=0.3, reg_delta=2,  # Regularisation params
                 min_samples_split=2, min_child_weight=0,  # Regularisation params
                 subsample=1, row_sample_method=None, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1,  # Sampling params
                 sketch_type="uniform", sketch_eps=0.001, sketch_rounds=float("inf"), bin_type="all", range_multiplier=1, hist_bin=32, categorical_map=None,  # Sketch params
                 dp_method="", accounting_method="rdp_scaled_improved", epsilon=1, quantile_epsilon=0, gradient_clipping=False,  # DP params
                 tree_budgets=None, level_budgets=None, gradient_budgets=None, # DP params
                 ignore_split_constraints=False, grad_clip_const=None, # DP params
                 split_method="hist_based", weight_update_method="xgboost", training_method="boosting", batched_update_size=1, # training method params
                 feature_interaction_method="", feature_interaction_k=None, full_ebm=False, # feature interaction params
                 early_stopping=None, es_metric=None, es_threshold=-5, es_window=3, # early stopping
                 track_budget=True, split_method_per_level=None, hist_estimator_method=None, sigma=None, verbose=False, output_train_monitor=False):

        super(PrivateGBDT, self).__init__(min_samples_split=min_samples_split, max_depth=max_depth, task_type=task_type)
        self.output_train_monitor = output_train_monitor

        # Training type
        self.training_method = training_method
        self.loss = loss
        if self.training_method == "rf":
            self.loss = BinaryRFLoss()

        self.batched_update_size = batched_update_size
        self.weight_update_method = weight_update_method # xgboost vs gbm updates
        self.split_method = split_method # Determines how splits are chosen - hist_based, partially_random, totally_random, hybrid_random
        self.split_method_per_level = split_method_per_level

        self.feature_interaction_method = feature_interaction_method
        self.feature_interaction_k = feature_interaction_k

        self.full_ebm = full_ebm

        if self.split_method in ["hist_based", "partially_random", "totally_random", "node_based"]:
            self.split_method_per_level = [self.split_method]*self.max_depth

        if self.split_method == "hybrid_random" and self.split_method_per_level is None:
                self.split_method_per_level = ["totally_random"] * self.max_depth # By default just do totally random

        self.hist_estimator_method = hist_estimator_method # one_sided, two_sided, two_sided_averaging

        # Base Parameters
        self.num_trees = num_trees
        self.feature_list = None
        self.num_features = None
        self.X = None
        self.ignore_split_constraints = ignore_split_constraints
        self.feature_bin = []
        self.gradient_histogram, self.hessian_histogram, self.root_hessian_histogram = [], [], []

        # Tracking vars
        self.train_monitor = TrainMonitor(0)

        # Regularisation Parameters
        self.reg_lambda = reg_lambda  # L2 regularisation on weights
        self.reg_alpha = reg_alpha  # L1 regularisation on gradients
        self.reg_gamma = reg_gamma  # Equivalent to the min impurity score needed to split a node further (or just leave it as a leaf)

        self.min_child_weight = min_child_weight  # Minimum sum of instance weight (hessian) needed in a child, if the sum of hessians less than this then the node is not split further
        self.min_samples_split = min_samples_split

        self.reg_eta = reg_eta  # Learning rate - multiplicative factor on weights
        self.reg_delta = reg_delta  # Clipping on the weights -> Useful in imablanced scenarios where it's possible for the hess to be 0 and thus the weights arbitraily large


        # Random Sampling Parameters
        self.index_sampler = IndexSampler(subsample, row_sample_method, colsample_bytree, colsample_bylevel, colsample_bynode)

        # Binning / Sketching Parameters for Feature Splits
        self.split_candidate_manager = SplitCandidateManager(hist_bin, self.num_trees, quantile_epsilon,
                                                             sketch_type, sketch_rounds, categorical_map,
                                                             sketch_eps, bin_type, range_multiplier)

        # Privacy (DP) Parameters
        self.dp_method = dp_method
        self.track_budget = track_budget
        self.verbose = verbose
        # The delta value of 1e-5 is a placeholder that is updated to 1/n when the dataset is being trained
        self.privacy_accountant = PrivacyAccountant(accounting_method, epsilon, 1e-5, quantile_epsilon, dp_method,
                                                    self.num_trees, self.max_depth, self.split_method, self.training_method, self.weight_update_method,
                                                    split_method_per_level=self.split_method_per_level,
                                                    tree_budgets=tree_budgets, gradient_budgets=gradient_budgets, level_budgets=level_budgets,
                                                    feature_interaction_method=self.feature_interaction_method, feature_interaction_k=self.feature_interaction_k,
                                                    sample_method=self.index_sampler.row_sample_method, subsample=self.index_sampler.subsample,
                                                    sketch_type=self.split_candidate_manager.sketch_type, sketch_rounds=self.split_candidate_manager.sketch_rounds,
                                                    task_type=self.task_type, sigma=sigma, grad_clip_const=grad_clip_const, gradient_clipping=gradient_clipping,
                                                    verbose=self.verbose)

        # Early stopping (not used)
        self.early_stopping = early_stopping
        self.es_metric = "root_hess" if es_metric is None else es_metric
        self.es_window = es_window
        self.es_threshold = es_threshold

    def _reset_accountant(self):
        self.privacy_accountant = PrivacyAccountant(self.privacy_accountant.accounting_method, self.privacy_accountant.epsilon, 1e-5,
                                                    self.privacy_accountant.quantile_epsilon, self.dp_method,
                                                    self.num_trees, self.max_depth, self.split_method, self.training_method, self.weight_update_method,
                                                    split_method_per_level=self.split_method_per_level,
                                                    tree_budgets=self.privacy_accountant.tree_budgets, gradient_budgets=self.privacy_accountant.gradient_budgets, level_budgets=self.privacy_accountant.level_budgets,
                                                    feature_interaction_method=self.feature_interaction_method, feature_interaction_k=self.feature_interaction_k,
                                                    sample_method = self.index_sampler.row_sample_method, subsample=self.index_sampler.subsample,
                                                    sketch_type=self.split_candidate_manager.sketch_type, sketch_rounds=self.split_candidate_manager.sketch_rounds,
                                                    task_type=self.task_type, sigma=self.privacy_accountant.sigma,
                                                    grad_clip_const=self.privacy_accountant.grad_clip_const, gradient_clipping=self.privacy_accountant.gradient_clipping,
                                                    verbose=self.verbose,)

    def _reset_tracking_attributes(self, checkpoint):
        self.X, self.y = None, None

        # These dont need to be removed but save space...
        if not checkpoint:
            self.train_monitor.current_tree_weights = []
            self.train_monitor.previous_tree_weights = []
            self.train_monitor.y_weights = []
            self.train_monitor.leaf_gradient_tracker = [[], []]
            self.train_monitor.root_gradient_tracker = [[], []]
            self.train_monitor.gradient_info = []

            self.privacy_accountant = None
            self.gradient_histogram = []
            self.feature_bin = []

    # Gradient/Hessian Calculations
    # ---------------------------------------------------------------------------------------------------

    def _compute_grad_hessian_with_samples(self, y, y_pred):
        """
        Called at the start of every tree, computes gradients and hessians for every observation from the previous predictions of the ensemble

        If using a LDP method, the perturbation is done here and the tree is formed as a post-processing step on the LDP perturbed gradients

        Otherwise, the raw gradients are passed to the model from fit() to  _build_tree() and they are perturbed later on in _add_dp_noise()

        :param y: True labels
        :param y_pred: Predicted labels
        :return: List of gradients and hessians
        """
        if self.task_type == 'classification' or self.task_type == "regression":
            grads = self.loss.compute_grad(y, y_pred)

            if self.task_type == "regression":
                grads = np.clip(grads, self.privacy_accountant.min_gradient, self.privacy_accountant.max_gradient)

            if self.weight_update_method == "xgboost":
                hess = self.loss.compute_hess(y, y_pred)
            else:
                hess = np.ones(len(y))

            if self.dp_method == "mean_mech_ldp":
                # Use mean mechanism perturbation
                hess_hm_client = HMClient(self.privacy_accountant.tree_budgets[len(self.trees)]*self.epsilon, self.privacy_accountant.max_hess, self.privacy_accountant.min_hess)  # Hess perturber
                grad_hm_client = HMClient(self.privacy_accountant.tree_budgets[len(self.trees)]*self.epsilon, self.privacy_accountant.max_gradient, self.privacy_accountant.min_gradient)  # Grad perturber
                grads = np.array([grad_hm_client.privatise(g) for g in grads])
                hess = np.array([hess_hm_client.privatise(h) for h in hess])
            elif self.dp_method == "gaussian_ldp":
                # Gaussian LDP
                grad_sigma = self.privacy_accountant.gaussian_var(gradient_type="gradient", depth=self.max_depth-1)
                hess_sigma = self.privacy_accountant.gaussian_var(gradient_type="hessian", depth=self.max_depth-1)
                if self.split_method == "hist_based":
                    grad_sigma /= math.sqrt(self.num_features * self.max_depth)
                    hess_sigma /= math.sqrt(self.num_features * self.max_depth)
                gradient_noise = np.random.normal(0, grad_sigma, size=(len(grads)))
                hess_noise = np.random.normal(0, grad_sigma, size=(len(hess)))

                grads = grads + gradient_noise
                hess = hess + hess_noise
        else:
            raise TypeError('%s task is not included in our XGboost algorithm !' % self.task_type)
        return grads, hess

    # Following methods assume that the total grads/hess that are passed have already been perturbed under some DP scheme
    def _L1_clip(self, total_grads):
        """
        L1 regularisation on the gradients, controlled by self.reg_alpha

        :param total_grads:
        :return:
        """
        return _L1_clip(total_grads, self.reg_alpha)
        # if total_grads > self.reg_alpha:
        #     return total_grads - self.reg_alpha
        # elif total_grads < -1 * self.reg_alpha:
        #     return total_grads + self.reg_alpha
        # else:
        #     return 0

    def _calculate_weight(self, total_grads, total_hess):
        """
        Calculates weight for leaf nodes

        :param total_grads: Total sum of gradients
        :param total_hess:  Total sum of hessians
        :return: Weight for leaf node
        """
        # if total_hess < self.min_hess:
        #     total_hess = 0
        return _calculate_weight(total_grads, total_hess, self.reg_alpha, self.reg_delta, self.reg_lambda)
        # weight = -1 * (self._L1_clip(total_grads) / (total_hess + self.reg_lambda))
        # if self.reg_delta != 0 and abs(weight) > self.reg_delta:
        #     return math.copysign(self.reg_delta, weight)  # Delta clipping
        # else:
        #     return weight

    def _calculate_gain(self, total_grads, total_hess):
        """
        Calculates gain from sum of gradients and sum of hessians

        :param total_grads: Sum of gradients
        :param total_hess: Sum of hessians
        :return: Gain score
        """
        return _calculate_gain(total_grads, total_hess, self.reg_alpha, self.reg_delta, self.reg_lambda)
        # weight = self._calculate_weight(total_grads, total_hess)
        # if self.reg_delta == 0:
        #     return -0.5 * weight * self._L1_clip(total_grads)  # G^2/H + lambda, with possible L1 regularisation and delta clipping on G
        # else:  # If delta-clipping is enabled the gain calculation is a little more complicated, following the implementation in the original XGBoost: https://github.com/dmlc/xgboost/blob/d7d1b6e3a6e2aa8fcb1857bf5e3188302a03b399/src/tree/param.h
        #     return -(2 * total_grads * weight + (total_hess + self.reg_lambda) * weight ** 2) + self.reg_alpha * abs(weight)  # This is an L1-regularised clipped gain calculation

    def _calculate_split_score(self, left_gain, right_gain, total_gain):
        return 0.5 * (left_gain + right_gain - total_gain)

    def _calculate_leaf_weight(self, total_grads, total_hess):
        """
        Calculates weight for leaf nodes, with optional learning rate specified by self.reg_eta

        :param total_grads: Sum of gradients
        :param total_hess: Sum of hessians
        :return: Leaf weight
        """
        if self.reg_alpha == 0:
            reg_alpha = float("inf")
        else:
            reg_alpha = self.reg_alpha

        if self.num_classes > 2:
            total_hess = np.clip(total_hess, 0, float("inf"))
            weight = -1 * (np.clip(total_grads, -reg_alpha, reg_alpha) / (total_hess + self.reg_lambda))
            if self.reg_delta != 0:
                clip_idx = np.abs(weight) > self.reg_delta
                weight[clip_idx] = np.copysign(self.reg_delta, weight[clip_idx])
            return weight
        else:
            return self._calculate_weight(total_grads, total_hess) * self.reg_eta  # Multiply the weight by the learning rate for leaf values

    # Main training logic
    # ---------------------------------------------------------------------------------------------------

    # Public method to train the model
    def fit(self, X, y):
        """
        Main training loop

        :param X: Training data as a pandas dataframe/ numpy array
        :param y: Training labels
        :return: self (trained GBDT model)
        """

        X = self._convert_df(X)
        self.num_features = X.shape[1]
        self.feature_list = range(0, self.num_features)  
        self.train_monitor.update_num_clients(X.shape[0])

        # Calculate split candidates
        self.train_monitor.start_timing_event("server", "initial split candidates")
        self.split_candidate_manager.find_split_candidates(X, 0, None, features_considering=self.feature_list)  
        self.train_monitor.end_timing_event("server", "initial split candidates")

        # TODO: Track comm (split candidates)
        self.train_monitor.update_received(range(0, X.shape[0]), 8*len(self.split_candidate_manager.feature_split_candidates)*len(self.split_candidate_manager.feature_split_candidates[0]))

        self.privacy_accountant.update_feature_candidate_size(self.split_candidate_manager.feature_split_candidates)
        self.X = X
        self.train_monitor.batched_weights = np.zeros(self.X.shape[0])

        if "cyclical" in self.feature_interaction_method and (self.split_candidate_manager.sketch_type == "adaptive_hessian" or self.full_ebm):
            if self.full_ebm:
                self.num_trees = self.num_trees * X.shape[1]
            self.split_candidate_manager.sketch_rounds = min(self.num_trees, self.split_candidate_manager.sketch_rounds*self.num_features)

            # recompute budget allocation
            self.train_monitor.start_timing_event("server", "privacy_accountant initialisation")
            self.privacy_accountant.__init__(self.privacy_accountant.accounting_method, epsilon=self.privacy_accountant.epsilon, delta=self.privacy_accountant.delta,
                                                    quantile_epsilon=self.privacy_accountant.quantile_epsilon, dp_method=self.dp_method,
                                                    num_trees=self.num_trees, max_depth=self.max_depth, split_method=self.split_method, training_method=self.training_method, weight_update_method=self.weight_update_method,
                                                    split_method_per_level=self.split_method_per_level,
                                                    feature_interaction_method=self.feature_interaction_method, feature_interaction_k=self.feature_interaction_k,
                                                    sample_method = self.index_sampler.row_sample_method, subsample=self.index_sampler.subsample,
                                                    sketch_type=self.split_candidate_manager.sketch_type, sketch_rounds=self.split_candidate_manager.sketch_rounds,
                                                    task_type=self.task_type, sigma=self.privacy_accountant.sigma,
                                                    grad_clip_const=self.privacy_accountant.grad_clip_const, gradient_clipping=self.privacy_accountant.gradient_clipping,
                                                    verbose=self.verbose,)
            self.train_monitor.end_timing_event("server", "privacy_accountant initialisation")

        if self.batched_update_size < 1:
            self.batched_update_size = int(self.batched_update_size * self.num_trees)

        y = y if not isinstance(y, pd.Series) else y.values
        self.num_classes = len(np.unique(y))
        self.train_monitor.set_num_classes(self.num_classes)

        self.train_monitor.start_timing_event("server", "initialise model weights")
        if self.num_classes > 2:
            self.loss = SoftmaxCrossEntropyLoss()
            y = LabelBinarizer().fit_transform(y)
            self.train_monitor.y_weights = np.full((X.shape[0], self.num_classes), 1/self.num_classes,)
            self.train_monitor.current_tree_weights = np.zeros((X.shape[0], self.num_classes))
        else:
            self.train_monitor.y_weights = np.zeros(X.shape[0]) # Initialise training weights to zero which is sigmoid(0) = 0.5 prob to either class
            self.train_monitor.current_tree_weights = np.zeros(X.shape[0])

        self.feature_weights = [1/self.num_features]*self.num_features

        # Initialise Gaussian DP parameters
        if "gaussian" in self.dp_method:
            self.privacy_accountant.assign_budget(self.privacy_accountant.epsilon, 1 / X.shape[0], num_rows=X.shape[0], num_features=X.shape[1]) # Update delta to 1/n

        self.train_monitor.end_timing_event("server", "initialise model weights")

        # Form histogram bin assignments for each feature - this caching saves a lot of time for histogram based gradient aggregation later on

        self.train_monitor.start_timing_event("client", "histogram building")
        for i in range(0, self.num_features):
            self.feature_bin.append(np.digitize(self.X[:, i], bins=[-np.inf] + list(np.array(self.split_candidate_manager.feature_split_candidates[i]) + 1e-11) + [np.inf]))
        self.train_monitor.end_timing_event("client", "histogram building")

        self.feature_bin = np.array(self.feature_bin)
        features = np.array(range(0, self.num_features))
        previous_rounds_features = None

        for i in range(0, self.num_trees):
            self.train_monitor.node_count = -1 # Reset node count for new trees

            if self.split_candidate_manager.sketch_each_tree:
                if self.split_candidate_manager.sketch_type == "adaptive_hessian" and len(self.trees) >= self.split_candidate_manager.sketch_rounds:
                    pass
                else:
                    features_updated = previous_rounds_features if previous_rounds_features is not None else list(range(0, self.num_features))
                    self.train_monitor.start_timing_event("server", f"split_candidates")
                    self.split_candidate_manager.find_split_candidates(X, len(self.trees), self.root_hessian_histogram, features_considering=features_updated)
                    self.train_monitor.end_timing_event("server", f"split_candidates")

                    self.train_monitor.start_timing_event("client", "histogram building")
                    for j in features_updated:
                        self.feature_bin[j] = np.digitize(self.X[:, j], bins=[-np.inf] + list(np.array(self.split_candidate_manager.feature_split_candidates[j]) + 1e-11) + [np.inf])
                    self.train_monitor.end_timing_event("client", "histogram building")

                    # TODO: Track here for communication (split candidates)
                    self.train_monitor.update_received(range(0, X.shape[0]), 8*len(features_updated)*len(self.split_candidate_manager.feature_split_candidates[0]))

            self.train_monitor.start_timing_event("server", "pre-tree ops")

            # Row and Feature Sampling if enabled
            row_sample, col_tree_sample, col_level_sample = self.index_sampler.sample(i, X.shape[0], X.shape[1], self.max_depth, feature_interaction_k=self.feature_interaction_k, feature_interaction_method=self.feature_interaction_method)
            previous_rounds_features = col_tree_sample
            split_constraints = {i : [0, len(self.split_candidate_manager.feature_split_candidates[i])+1] for i in range(0,self.num_features)}

            if i != 0:
                self.privacy_accountant.update_tree() # Increment tree count in privacy_accountant, used to index tree_budgets
            self.train_monitor.end_timing_event("server", "pre-tree ops")

            if i==0 or self.training_method == "boosting" or (self.training_method == "batched_boosting" and (i % self.batched_update_size == 0)):
                if self.training_method == "batched_boosting":
                    self.train_monitor.y_weights += self.train_monitor.batched_weights / self.batched_update_size
                    self.train_monitor.batched_weights = np.zeros(self.X.shape[0])
                    # TODO: Track communication (batched updates)
                    if i != 0:
                        self.train_monitor.update_sent(range(0, self.X.shape[0]), 8*2*self.batched_update_size*self.train_monitor.leaf_count_tracker[-1])

                self.train_monitor.start_timing_event("client", f"computing gradients")
                grads, hess = self._compute_grad_hessian_with_samples(y, self.loss.predict(self.train_monitor.y_weights)) # Compute raw grads,hess
                self.train_monitor.end_timing_event("client", f"computing gradients")
                self.train_monitor.gradient_info = [(grads, hess)] # Append to gradient_info, at each node this is retrieved and privatised with DP to calculate feature scores etc

            tree = self._build_tree(features, row_sample, None, None,
                                    split_constraints=split_constraints, col_tree_sample=col_tree_sample, col_level_sample=col_level_sample, row_ids=np.arange(0,X.shape[0]))
            self.trees.append(tree) # Build and add tree to ensemble

            self.train_monitor.start_timing_event("server", "post-tree ops")

            if self.training_method == "batched_boosting":
                self.train_monitor.batched_weights += self.train_monitor.current_tree_weights

                if i==self.num_trees-1 and (i+1) % self.batched_update_size != 0:
                    # TODO: Track communication (batched updates)
                    self.train_monitor.update_sent(range(0, self.X.shape[0]), 8*2*self.batched_update_size*self.train_monitor.leaf_count_tracker[-1])
                    self.train_monitor.y_weights += self.train_monitor.batched_weights / ((i+1) % self.batched_update_size)
                elif i==self.num_trees-1:
                    # TODO: Track communication (batched updates)
                    self.train_monitor.update_sent(range(0, self.X.shape[0]), 8*2*self.batched_update_size*self.train_monitor.leaf_count_tracker[-1])
            else:
                self.train_monitor.y_weights += self.train_monitor.current_tree_weights # Update weights

            self.train_monitor.leaf_gradient_tracker[0].append(self.train_monitor.gradient_total[0])
            self.train_monitor.leaf_gradient_tracker[1].append(self.train_monitor.gradient_total[1])

            threshold_change = self.es_threshold
            window = self.es_window
            if len(self.trees) >= 2*self.es_window and self.early_stopping:
                if self.es_metric == "leaf_hess":
                    es_metric = self.train_monitor.leaf_gradient_tracker[1]
                elif self.es_metric == "leaf_grad":
                    es_metric = self.train_monitor.leaf_gradient_tracker[0]
                elif self.es_metric == "root_grad":
                    es_metric = self.train_monitor.root_gradient_tracker[0]
                else:
                    es_metric = self.train_monitor.root_gradient_tracker[1]

                current_window_hess = abs(np.mean(es_metric[-window:])) if "grad" in self.es_metric else np.mean(es_metric[-window:])
                previous_window_hess = abs(np.mean(es_metric[-2*window:-window])) if "grad" in self.es_metric else np.mean(es_metric[-2*window:-window])
                per_change = (previous_window_hess/current_window_hess-1)*100

                if ("standard" in self.early_stopping or self.early_stopping == "rollback" or "average" in self.early_stopping) and per_change < threshold_change:
                    print("Early Stopping at round", i+1)
                    if self.early_stopping == "rollback":
                        self.trees = self.trees[:-1]
                        self.train_monitor.y_weights -= self.train_monitor.current_tree_weights
                    break
                elif self.early_stopping == "retry" and per_change < threshold_change: #es_metric[-2] - es_metric[-1] < 0
                    prune_step = -2 if "root" in self.es_metric else -1

                    self.trees = self.trees[:prune_step]
                    self.train_monitor.gradient_info = self.train_monitor.gradient_info[:prune_step]
                    self.train_monitor.root_gradient_tracker[0] = self.train_monitor.root_gradient_tracker[0][:prune_step]
                    self.train_monitor.root_gradient_tracker[1] = self.train_monitor.root_gradient_tracker[1][:prune_step]
                    self.train_monitor.leaf_gradient_tracker[0] = self.train_monitor.leaf_gradient_tracker[0][:prune_step]
                    self.train_monitor.leaf_gradient_tracker[1] = self.train_monitor.leaf_gradient_tracker[1][:prune_step]

                    self.train_monitor.y_weights -= self.train_monitor.current_tree_weights + (prune_step+1)*-1*self.train_monitor.previous_tree_weights # If root then remove 2 trees, if leaf remove 1

            self.train_monitor.end_timing_event("server", "post-tree ops")

            # Reset tracking vars
            self.train_monitor._update_comm_stats(self.split_method, self.training_method)
            self.train_monitor.reset()

        if self.early_stopping and "retrain" in self.early_stopping and len(self.trees) < self.num_trees:
            # Calculate new budget
            old_eps = self.epsilon
            self.epsilon = self.epsilon - self.privacy_accountant._autodp_check(self.privacy_accountant.sigma_arr, len(self.trees))

            if "majority" in self.early_stopping and self.epsilon >= old_eps:
                print("Budget leftover is not enough to train model... exiting with early stopped model")
                return self
            else:
                # Reset parameters
                old_trees = []
                if "average" in self.early_stopping:
                    old_trees = self.trees
                es_method = self.early_stopping

                self._reset_tracking_attributes(False)
                self._reset_accountant()
                self.num_trees = len(self.trees)
                self.trees = []
                self.privacy_accountant.assign_budget(self.epsilon, 1/X.shape[0], X.shape[0], X.shape[1])
                self.early_stopping = None
                self.fit(X,y) # Retrain model

                self.trees = old_trees + self.trees
                self.early_stopping = es_method

                return self

        self.root = self.trees[0]

        if self.training_method == "rf":
            self.train_monitor.y_weights /= len(self.trees)

        if self.verbose:
            print("Number of trees trained", len(self.trees))
            if self.dp_method != "":
                # Track budget spent by participants - for debugging
                scale_factor = 2 if self.privacy_accountant.gradient_method == "vector_mechanism" else 1
                print("\n[Ledger] Average number of queries done by a participant :", np.mean(self.privacy_accountant.ledger)/scale_factor)
                print("[Ledger] Minimum queries done by a participant:", np.min(self.privacy_accountant.ledger)/scale_factor)
                print("[Ledger] Maximum queries done by a participant:", np.max(self.privacy_accountant.ledger)/scale_factor, "\n")

        self.y_weights = self.train_monitor.y_weights
        if self.output_train_monitor:
            print(f"Size of dataset n,m={self.X.shape}")
            self.train_monitor.output_summary()
        return self

    def _calculate_feature_split(self, features_considering, split_index, current_depth, total_gain, total_grads, total_hess, grads, hess, feature_split_constraints):
        """

        Calculates split scores for a specific features and all their split candidate values

        :param feature_i: Feature index
        :param feature_values: Feature values
        :param current_depth: Current depth in the tree
        :param total_gain: Total gain
        :param total_grads: Total grads
        :param total_hess: Total hess
        :param grads: list of grads
        :param hess: list of hess
        :return:
        """
        # Iterate through all unique values of feature column i and calculate the impurity
        values = []
        current_max_score = -1e20
        split_method = self.split_method_per_level[current_depth]
        self.current_tree_depth = max(self.train_monitor.current_tree_depth, current_depth)

        valid_features = []
        for i in features_considering:
            if feature_split_constraints[i][0] < min(feature_split_constraints[i][1], len(self.split_candidate_manager.feature_split_candidates[i])): # Ignore features that cannot be split on any further
                valid_features.append(i)

        if len(valid_features) == 0:
            return []

        if split_method == "totally_random":
            new_features = valid_features
            weights = None
            chosen_feature = np.random.choice(new_features, p=weights)
            chosen_split = np.random.choice(range(feature_split_constraints[chosen_feature][0], min(feature_split_constraints[chosen_feature][1], len(self.split_candidate_manager.feature_split_candidates[chosen_feature])))) # Upper split constraint may be len(feature_splits)+1 so needs to be truncated back down
                                                                                                                                                    # This is due to how the hist_based splitting logic works because of the splicing
        # TODO: Track communication (internal splits)
        if split_method != "totally_random":
            for feature_i in valid_features:
                constraints = feature_split_constraints[feature_i]
                self.train_monitor.bin_tracker[current_depth] += constraints[1] - constraints[0]

        for feature_i in valid_features:
            split_constraints = feature_split_constraints[feature_i]

            if split_method == "partially_random":
                chosen_split = random.randint(split_constraints[0], min(split_constraints[1], len(self.split_candidate_manager.feature_split_candidates[feature_i])))
            elif split_method == "totally_random" and feature_i != chosen_feature:
                continue

            if split_method == "hist_based":
                cumulative_grads = np.cumsum(self.private_gradient_histogram[feature_i][split_constraints[0]: split_constraints[1]+1])
                cumulative_hess = np.cumsum(self.private_hessian_histogram[feature_i][split_constraints[0]: split_constraints[1]+1])
                total_grads_cu = cumulative_grads[-1]
                total_hess_cu = cumulative_hess[-1]

            for j, threshold in enumerate(self.split_candidate_manager.feature_split_candidates[feature_i]):
                if (split_method == "partially_random" or split_method == "totally_random") and j != chosen_split:
                    continue

                if split_constraints[0] <= j <= split_constraints[1] or self.ignore_split_constraints: # Only add split if it isn't one-sided (based on public knowledge of previous splits)
                    # Calculate impurity score of proposed split
                    if split_method == "hist_based":
                        left_grads_sum = cumulative_grads[j-(split_constraints[0])]
                        left_hess_sum = cumulative_hess[j-(split_constraints[0])]

                        if self.hist_estimator_method == "one-sided":
                            right_grads_sum = total_grads - left_grads_sum
                            right_hess_sum = total_hess - left_hess_sum
                        else:
                            right_grads_sum = total_grads_cu - left_grads_sum
                            right_hess_sum = total_hess_cu - left_hess_sum

                        if self.hist_estimator_method == "two_sided":
                            total_grads = self.private_gradient_histogram[feature_i][split_constraints[0]: split_constraints[1]].sum()
                            total_hess = self.private_hessian_histogram[feature_i][split_constraints[0]: split_constraints[1]].sum()
                            total_gain = self._calculate_gain(total_grads, total_hess)

                        split_score = self._calculate_split_score(self._calculate_gain(left_grads_sum, left_hess_sum), self._calculate_gain(right_grads_sum, right_hess_sum), total_gain)
                    elif split_method == "node_based" or split_method == "partially_random":
                        new_split_index = self.X[split_index, feature_i] <= threshold
                        left_grads_sum, left_hess_sum = self.privacy_accountant._add_dp_noise(grads[new_split_index].sum(), hess[new_split_index].sum(), current_depth, feature=feature_i, num_obs=len(new_split_index))
                        right_grads_sum = total_grads - left_grads_sum
                        right_hess_sum = total_hess - left_hess_sum
                        split_score = self._calculate_split_score(self._calculate_gain(left_grads_sum, left_hess_sum), self._calculate_gain(right_grads_sum, right_hess_sum), total_gain)
                    else: # In the case of totally random no score are computed
                        split_score = float("inf")
                        left_grads_sum, left_hess_sum, right_grads_sum, right_hess_sum = [float("inf")]*4

                    if split_score > current_max_score:
                        # Divide X and y depending on if the feature value of X at index feature_i meets the threshold
                        values = [feature_i, j, threshold, split_score, None, (left_grads_sum, left_hess_sum, right_grads_sum, right_hess_sum)]
                        current_max_score = split_score
        return values

    def _form_private_gradient_histogram(self, grads, hess, features_considering, split_index, current_depth, adaptive_hessian=False):
        self.train_monitor.start_timing_event("server", "initialise private histogram")
        self.train_monitor.start_timing_event("client", "initialise private histogram")

        if current_depth == 0:
            self.gradient_histogram = {}
            self.hessian_histogram = {}
            self.private_gradient_histogram = {i: np.zeros(len(self.split_candidate_manager.feature_split_candidates[i])+1) for i in features_considering}
            self.private_hessian_histogram = {i: np.zeros(len(self.split_candidate_manager.feature_split_candidates[i])+1) for i in features_considering}

        if current_depth == 0 and len(self.trees) == 0:
            self.root_hessian_histogram = {i: np.zeros(len(self.split_candidate_manager.feature_split_candidates[i])+1) for i in features_considering}

        self.train_monitor.end_timing_event("server", "initialise private histogram")
        self.train_monitor.end_timing_event("client", "initialise private histogram")

        for i in features_considering:
            self.train_monitor.start_timing_event("client", "forming gradient + hess histogram")
            num_bins = len(self.split_candidate_manager.feature_split_candidates[i])+1
            digitized = self.feature_bin[i][split_index]
            self.gradient_histogram[i] = np.array(histogram1d(digitized, bins=num_bins, range=[0, num_bins+0.1], weights=grads)) # Fast C histogram implementation
            self.hessian_histogram[i] = np.array(histogram1d(digitized, bins=num_bins, range=[0, num_bins+0.1], weights=hess))

            self.train_monitor.end_timing_event("client", "forming gradient + hess histogram")
            self.train_monitor.start_timing_event("server", "adding noise to gradient + hess histogram")
            if self.dp_method != "":
                if adaptive_hessian:
                    _, self.root_hessian_histogram[i] = self.privacy_accountant._add_dp_noise(self.gradient_histogram[i], self.hessian_histogram[i],
                                                                           depth=self.max_depth-1,
                                                                           feature=i, histogram_row=True, noise_size=num_bins, adaptive_hessian=True)
                else:
                    self.private_gradient_histogram[i], self.private_hessian_histogram[i] = self.privacy_accountant._add_dp_noise(self.gradient_histogram[i], self.hessian_histogram[i],
                                                                                                               depth=current_depth,
                                                                                                               feature=i, histogram_row=True, noise_size=num_bins)
            self.train_monitor.end_timing_event("server", "adding noise to gradient + hess histogram")
            # TODO: Communication
            if current_depth == 0 and adaptive_hessian:
                self.train_monitor.update_sent(range(0, self.X.shape[0]), 8*num_bins, increment_round=False)

        # TODO: Communication
        if current_depth == 0 and adaptive_hessian:
            self.train_monitor.client_rounds_sent[-1] += 1

        if self.dp_method == "":
            self.private_gradient_histogram, self.private_hessian_histogram = self.gradient_histogram, self.hessian_histogram

    def _get_node_id(self, depth, node_num):
        return str(depth) + "_" + str(node_num)

    def _build_tree(self, features, split_index, node_total_grads, node_total_hess,
                    current_depth=0, col_tree_sample=None, col_level_sample=None, row_ids=None, split_constraints=None, previous_node_num=1):
        """
        Main method for building a tree of the ensemble

        :param split_index: Boolean index of current observations in the node
        :param node_total_grads: Total gradients of the node
        :param node_total_hess: Total hessians of the node
        :param current_depth: Current depth of the node
        :param col_tree_sample: Boolean index of features to sample if self.colsample_bynode is not 1
        :param col_level_sample: Boolean index of features to sample if self.colsample_bylevel is not 1
        :return:
        """

        self.train_monitor.start_timing_event("server", "sampling features for node")

        features_considering = features
        # Perform column (feature) sampling if needed
        if col_tree_sample is not None:
            features_considering = features_considering[col_tree_sample]
        if col_level_sample is not None:
            features_considering = features_considering[col_level_sample[current_depth]]
        if self.index_sampler.colsample_bynode < 1:
            features_considering = features_considering[np.random.choice(range(0, len(features_considering)), size=math.ceil(len(features_considering) * self.index_sampler.colsample_bynode), replace=False)]

        self.train_monitor.node_count += 1
        self.privacy_accountant.current_node = self.train_monitor.node_count
        self.privacy_accountant.current_tree = len(self.trees)
        split_method = self.split_method_per_level[min(current_depth, self.max_depth-1)]
        self.train_monitor.end_timing_event("server", "sampling features for node")

        self.train_monitor.start_timing_event("client", "retrieving grads/hess for node")
        # Obtain raw gradients/hessians for the observations in the current node
        grads, hess = self.train_monitor.gradient_info[-1][0][split_index], self.train_monitor.gradient_info[-1][1][split_index]
        self.train_monitor.end_timing_event("client", "retrieving grads/hess for node")

        if self.num_classes > 2:
            grads, hess = self.train_monitor.gradient_info[-1][0][split_index], self.train_monitor.gradient_info[-1][1][split_index]
            raw_grads_sum, raw_hess_sum = np.sum(grads, axis=0), np.sum(hess, axis=0)
        else:
            raw_grads_sum, raw_hess_sum = grads.sum(), hess.sum()
            # raw_grads_sum, raw_hess_sum = self.train_monitor.gradient_info[-1][0].sum(where=split_index), self.train_monitor.gradient_info[-1][1].sum(where=split_index)

        if current_depth == 0: # Update private grads at root node
            if split_method == "node_based" or split_method == "partially_random":
                node_total_grads, node_total_hess = self.privacy_accountant._add_dp_noise(raw_grads_sum, raw_hess_sum, -1, num_obs=len(split_index)) # Depth zero
                if self.dp_method != "" and self.track_budget:
                    self.privacy_accountant.commit_budget_to_ledger(split_index)
            elif split_method == "totally_random":
                node_total_hess = float("inf")
                node_total_grads = float("inf")

                if self.split_candidate_manager.sketch_type == "adaptive_hessian" and len(self.trees) < self.split_candidate_manager.sketch_rounds:
                    self._form_private_gradient_histogram(grads, hess, features_considering, split_index, current_depth, adaptive_hessian=True) # Adaptive hess test
            else:
                self._form_private_gradient_histogram(grads, hess, features_considering, split_index, current_depth) # Form privatised grads,hess
                if self.dp_method != "" and self.track_budget:
                    self.privacy_accountant.commit_budget_to_ledger(split_index)
                node_total_grads = sum([i.sum() for i in self.private_gradient_histogram.values()])/len(self.private_gradient_histogram.values())
                node_total_hess = sum([i.sum() for i in self.private_hessian_histogram.values()])/len(self.private_hessian_histogram.values())

            self.train_monitor.root_gradient_tracker[0].append(node_total_grads)
            self.train_monitor.root_gradient_tracker[1].append(node_total_hess)

        # If the spliting conditions are satisfied then split the current node otherwise stop and make it a leaf
        if (node_total_hess >= self.min_child_weight or split_method == "totally_random") and (current_depth < self.max_depth):
            if split_method == "hist_based" and current_depth > 0:
                self._form_private_gradient_histogram(grads, hess, features_considering, split_index, current_depth) # Form privatised grads,hess
                if self.hist_estimator_method == "two_sided_averaging" or node_total_grads == float("inf"):
                    node_total_grads = sum([i.sum() for i in self.private_gradient_histogram.values()])/len(self.private_gradient_histogram.values())
                    node_total_hess = sum([i.sum() for i in self.private_hessian_histogram.values()])/len(self.private_hessian_histogram.values())

            self.train_monitor.start_timing_event("server", "calculating internal split")

            # Calculate current nodes total gain
            node_gain = self._calculate_gain(node_total_grads, node_total_hess)

            # Find best (feature, split) candidates for each feature
            split_data = self._calculate_feature_split(features_considering, split_index, current_depth, node_gain, node_total_grads, node_total_hess, grads, hess, split_constraints)

            # Commit budget spent by participants for computing split scores
            if self.dp_method != "" and self.track_budget:
                self.privacy_accountant.commit_budget_to_ledger(split_index)

            self.train_monitor.end_timing_event("server", "calculating internal split")

            if split_data:
                chosen_feature, bucket_index, chosen_threshold, largest_score, left_split_index, split_gradient_info = split_data
                if left_split_index is None:
                    left_split_index = self.X[split_index, chosen_feature] <= chosen_threshold
                left_grads_sum, left_hess_sum, right_grads_sum, right_hess_sum = split_gradient_info # Gradient information to pass to child nodes
                right_split_index = split_index[~left_split_index]
                left_split_index = split_index[left_split_index]

                if largest_score > self.reg_gamma:
                    self.train_monitor.start_timing_event("server", "updating split constraints")
                    # Update feature split constraints with valid feature split candidate index bounds - this stops the algo from picking one-sided splits later on
                    left_split_constraints = copy(split_constraints)
                    left_split_constraints[chosen_feature] = [left_split_constraints[chosen_feature][0], bucket_index-1]
                    right_split_constraints = copy(split_constraints)
                    right_split_constraints[chosen_feature] = [bucket_index+1, right_split_constraints[chosen_feature][1]]
                    self.train_monitor.end_timing_event("server", "updating split constraints")

                    # Build subtrees recursively for the right and left branches
                    self.train_monitor.last_feature = chosen_feature

                    left_num = 2*(previous_node_num)-1
                    right_num = 2*(previous_node_num)

                    left_branch = self._build_tree(features, left_split_index, left_grads_sum, left_hess_sum,
                                                   current_depth + 1, col_tree_sample, col_level_sample, split_constraints=left_split_constraints, previous_node_num=left_num)
                    right_branch = self._build_tree(features, right_split_index, right_grads_sum, right_hess_sum,
                                                    current_depth + 1, col_tree_sample, col_level_sample, split_constraints=right_split_constraints, previous_node_num=right_num)

                    self.train_monitor.internal_node_count[current_depth] += 1
                    return DecisionNode(node_id=str(current_depth) + "_" + str(previous_node_num), feature_i=chosen_feature, threshold=chosen_threshold, true_branch=left_branch, false_branch=right_branch, split_gain=largest_score, gradient_sum=node_total_grads, hessian_sum=node_total_hess, num_observations=len(split_index), depth=current_depth)

        self.train_monitor.start_timing_event("server", "leaf weight")
        # We're at leaf => determine weight
        if split_method == "totally_random":
            if self.dp_method != "" and self.dp_method != "gaussian_ldp":
                size = self.num_classes if self.num_classes > 2 else None
                raw_grads_sum = raw_grads_sum.sum()
                raw_hess_sum = raw_hess_sum.sum()

                node_total_grads = raw_grads_sum + np.random.normal(0, self.privacy_accountant.gaussian_var(gradient_type="gradient", depth=self.max_depth-1), size=size)
                node_total_hess = raw_hess_sum + np.random.normal(0, self.privacy_accountant.gaussian_var(gradient_type="hessian", depth=self.max_depth-1), size=size)
                if self.track_budget:
                    self.privacy_accountant.commit_budget_to_ledger(split_index)
            else:
                node_total_grads, node_total_hess = raw_grads_sum, raw_hess_sum

        # Calculate leaf weight based on DP grads/hess
        leaf_weight = self._calculate_model_update(node_total_grads, node_total_hess, None, None) # pass grads/hess?

        # TODO: Track comm (weight)
        if self.training_method != "batched_boosting":
            # self.train_monitor.update_sent(split_index, 8*2, increment_round=False)
            pass

        self.train_monitor.end_timing_event("server", "leaf weight")

        # Update training information...
        self.train_monitor.gradient_total[0] += node_total_grads
        self.train_monitor.gradient_total[1] += node_total_hess
        self.train_monitor.leaf_count += 1
        self.train_monitor.current_tree_weights[split_index] += leaf_weight

        # split_gain = self._calculate_gain(node_total_grads, node_total_hess)

        if self.num_classes == 2:
            leaf_weight = np.array([leaf_weight])
        return DecisionNode(node_id=str(current_depth) + "_" + str(previous_node_num), value=leaf_weight, num_observations=len(split_index), gradient_sum=node_total_grads,
                            hessian_sum=node_total_hess,
                            split_gain=0,
                            feature_i=self.train_monitor.last_feature)

    def _calculate_model_update(self,node_total_grads, node_total_hess,  grads=None, hess=None):
        if self.training_method == "rf":  # RF update
            if node_total_hess <= 0:
                leaf_weight = 0.5
            elif node_total_grads <= 0:
                leaf_weight = 0
            else:
                leaf_weight = node_total_grads/node_total_hess
                if leaf_weight > 1:
                    leaf_weight = 1
        else: # Grad or Newton update
            if self.num_classes > 2:
                leaf_weight = self._calculate_leaf_weight(node_total_grads, node_total_hess)
            else:
                if node_total_hess <= self.privacy_accountant.min_hess:
                    leaf_weight = 0
                else:
                    node_total_grads = np.clip(node_total_grads, self.privacy_accountant.min_gradient*self.X.shape[0], self.privacy_accountant.max_gradient*self.X.shape[0])
                    node_total_hess = np.clip(node_total_hess, self.privacy_accountant.min_hess*self.X.shape[0], self.privacy_accountant.max_hess*self.X.shape[0])
                    leaf_weight = self._calculate_leaf_weight(node_total_grads, node_total_hess)

        # Signed or individual updates...
        if "signed" in self.weight_update_method or "per_sample" in self.weight_update_method:
            leaf_weight = (-grads/(hess+self.reg_lambda)) if "newton" in self.weight_update_method else -grads
            # leaf_weight = np.sign(leaf_weight) # local sign - doesnt work...

            leaf_weight = leaf_weight.sum()

            if "signed" in self.weight_update_method:
                if leaf_weight < 0:
                    leaf_weight = -self.reg_delta
                elif leaf_weight > 0:
                    leaf_weight = self.reg_delta
                else:
                    leaf_weight = 0

            leaf_weight = np.clip(leaf_weight, -self.reg_delta, self.reg_delta)
            leaf_weight *= self.reg_eta
            # print(leaf_weight)

        return leaf_weight

    # Feature Importance
    # --------------------------------------------------------------------------------------

    # See https://xgboost.readthedocs.io/en/latest/R-package/discoverYourData.html#feature-importance for more details on how to compute feature importance

    def _traverse_tree(self, tree, feature_importance_map, feature_count_map, threshold_values, depth_map, method):
        """
        Recursively traverses a tree and calculates various statistics needed for feature importance

        :param tree: TreeBase object
        :param feature_importance_map: map of tree ids to feature importance values
        :param feature_count_map: map of feature values to counts of the time it's appeared in a split
        :param threshold_values: Split values chosen
        :param method: Feature importance method
        :return:
        """
        if tree is None:
            return

        if "gain" in method and tree.split_gain is not None and tree.split_gain != float("inf") and tree.feature_i != -1:
            feature_importance_map[tree.feature_i] += tree.split_gain
            threshold_values[tree.feature_i].append(tree.threshold)
            feature_count_map[tree.feature_i] += 1
            depth_map[tree.depth] += tree.split_gain
        elif "cover" in method and tree.hessian_sum is not None and tree.hessian_sum != float("inf") and tree.feature_i != -1:
            feature_importance_map[tree.feature_i] += tree.hessian_sum
            threshold_values[tree.feature_i].append(tree.threshold)
            feature_count_map[tree.feature_i] += 1
            depth_map[tree.depth] += tree.hessian_sum

        self._traverse_tree(tree.true_branch, feature_importance_map, feature_count_map, threshold_values, depth_map, method)
        self._traverse_tree(tree.false_branch, feature_importance_map, feature_count_map, threshold_values, depth_map, method)

    def feature_importance(self, method="gain", return_all=False):
        """
        Calculates feature importance from the trained model

        :param method: Feature importance method from "gain", "cover", "average_gain", "average_cover"
        :return: Map of features to their importance
        """
        if len(self.trees) == 0:
            Exception("Cannot calculate feature importance from an untrained model. Use .fit(X,y) first")

        feature_importance_map = Counter({k: 0 for k in self.feature_list})
        feature_count_map = Counter({k: 0 for k in self.feature_list})
        threshold_values = {k: [] for k in self.feature_list}
        for i, tree in enumerate(self.trees):
            depth_map = defaultdict(int)
            self._traverse_tree(tree, feature_importance_map, feature_count_map, threshold_values, depth_map, method)

        if "average" in method:  # average, by default total gain/cover will be used
            for k in feature_importance_map.keys():
                feature_importance_map[k] = feature_importance_map[k] / feature_count_map[k]

        if return_all:
            return feature_importance_map, feature_count_map, threshold_values, depth_map
        else:
            return feature_importance_map
