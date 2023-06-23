import statsmodels.distributions.empirical_distribution as edf
import pandas as pd
import numpy as np
import math

from federated_gbdt.core.pure_ldp.frequency_oracles.square_wave.sw_client import SWClient
from federated_gbdt.core.pure_ldp.frequency_oracles.square_wave.sw_server import SWServer
from federated_gbdt.core.pure_ldp.frequency_oracles.local_hashing import FastLHClient, FastLHServer

from scipy.interpolate import interp1d
from scipy.stats import skew, skewtest

from federated_gbdt.core.binning.feature_binning_param import FeatureBinningParam
from federated_gbdt.core.binning.quantile_binning import QuantileBinning, get_split_points
from federated_gbdt.core.dp_multiq.joint_exp import joint_exp # JointExp DP Quantiles

from copy import copy

class SplitCandidateManager():
    def __init__(self, num_candidates, num_trees, split_candidate_epsilon,
                 sketch_type, sketch_rounds, categorical_map, sketch_eps, bin_type, range_multiplier):

        self.num_candidates = num_candidates
        self.sketch_type = sketch_type
        self.split_candidate_epsilon = split_candidate_epsilon
        self.sketch_each_tree = False
        self.sketch_rounds = sketch_rounds

        if self.sketch_rounds == float("inf"):
            self.sketch_rounds = num_trees

        if self.sketch_type == "random_uniform":
            self.sketch_type = "random_guess"
            self.sketch_each_tree = True

        if self.sketch_type == "adaptive_hessian":
            self.sketch_each_tree = True

        self.categorical_map = categorical_map
        self.sketch_eps = sketch_eps
        self.bin_type = bin_type
        self.range_multiplier = range_multiplier # Used to test how min/max bounds on features affects accuracy
        
        self.feature_split_candidates = [] 

    def _find_quantiles(self, dp_hist, bins, interpolate=True):
        """
        Helper method used by _find_split_candidates() to find feature quantiles given a histogram over various bins

        :param dp_hist: DP histogram over the bins
        :param bins: Number of bins
        :param interpolate: Boolean - Whether or not to linearly/uniformly interpolate between bins to calculate quantiles
        :return: List of estimated quantiles
        """
        # Find quantiles
        dp_quantiles = []
        prob = 1/self.num_candidates
        for q in range(self.num_candidates):
            total_probs = 0
            i = 0
            frac = 0
            for k, val in enumerate(dp_hist):
                total_probs += val
                i += 1
                if total_probs >= (q+1) * prob:
                    if total_probs > (q+1)*prob and interpolate is True:
                        frac = (total_probs-(q+1)*prob)/prob
                    break
            if i < len(dp_hist) and interpolate:
                dp_quantiles.append(bins[i-1]+(bins[i]-bins[i-1])*frac)
            else:
                dp_quantiles.append(bins[i-1])

        return dp_quantiles

    def find_split_candidates(self, X, round_num, hessian_hist=None, features_considering=None):
        """
        Used to find split candidates (often via quantiles), method is determined by the self.sketch_type argument

        :param X: Data with features as columns
        """
        old_splits = copy(self.feature_split_candidates)
        self.feature_split_candidates = []

        if self.categorical_map is not None:
            split_candidate_epsilon = self.split_candidate_epsilon/(X.shape[1] - sum(self.categorical_map))
        else:
            split_candidate_epsilon = self.split_candidate_epsilon / X.shape[1]

        if self.sketch_type == "feverless_uniform" or "adaptive_hessian" in self.sketch_type:
            # feverless_splits = get_split_points(pd.DataFrame(X), bin_num=self.num_candidates) # By default feverless uses bin_num=32
            if round_num > 0:
                total_hess = sum([hessian_hist[i].sum() for i in hessian_hist.keys()])/len(hessian_hist)

        if self.sketch_type == "sketch":
            param_obj = FeatureBinningParam(bin_num=self.num_candidates, error=self.sketch_eps)
            quantile_sketch = QuantileBinning(params=param_obj)
            quantile_sketch.fit_split_points(pd.DataFrame(X), False)
            for x in quantile_sketch.get_split_points_result_numpy():
                self.feature_split_candidates.append(x)
        elif self.sketch_type == "feverless":
            self.feature_split_candidates = get_split_points(pd.DataFrame(X), bin_num=self.num_candidates) # By default feverless uses bin_num=32
        else:
            for j in range(0, X.shape[1]):
                max_val = np.nanmax(X[:, j]) * self.range_multiplier
                min_val = np.nanmin(X[:, j])
                if min_val <= 0:
                    min_val = min_val * self.range_multiplier
                else:
                    min_val = min_val / self.range_multiplier

                if self.sketch_type == "exact_quantiles":
                    quantiles = np.nanquantile(X[:, j], q=np.arange(0,1, 1/self.num_candidates), interpolation='linear')
                    self.feature_split_candidates.append(quantiles)
                elif self.sketch_type == "exact_icdf":
                    # Inverse CDF method
                    sample_edf = edf.ECDF(X[:, j])
                    slope_changes = sorted(set(X[:,j]))
                    sample_edf_values_at_slope_changes = [sample_edf(item) for item in slope_changes]
                    if len(slope_changes) > 2:
                        inverted_edf = interp1d(sample_edf_values_at_slope_changes, slope_changes, fill_value="extrapolate")
                        self.feature_split_candidates.append(inverted_edf(np.arange(0,1, 1/self.num_candidates)))
                    else:
                        self.feature_split_candidates.append([slope_changes[0]])
                elif self.sketch_type == "log":
                    col = X[:,j]
                    data = np.log(1+(-np.nanmin(col))+col)
                    splits = np.exp(np.linspace(np.nanmin(data),np.nanmax(data), num=self.num_candidates, endpoint=False))+np.nanmin(col)-1

                    if skew(col) < 0:
                        splits = -splits + np.nanmax(col)

                    self.feature_split_candidates.append(np.sort(splits))
                elif self.sketch_type == "skew_test_log_uniform":
                    col = X[:,j]
                    if skewtest(col).pvalue <= 0.05: # Feature is skewed
                        data = np.log(1+(-np.nanmin(col))+col)
                        splits = np.exp(np.linspace(np.nanmin(data),np.nanmax(data), num=self.num_candidates, endpoint=False))+np.nanmin(col)-1 # For positive skews

                        if skew(col) < 0:
                            splits = -splits + np.nanmax(col) # For negative skews
                        splits = np.sort(splits)
                    else: # Uniform bins
                        step = (np.max(col) - np.min(col)) / self.num_candidates
                        unique_values = np.unique(X[:,j])
                        if self.bin_type != "all":
                            if len(unique_values) <= self.num_candidates:
                                splits = list(unique_values)  # If a feature has <= values than the number of bins then we can do exact splitting
                        if step == 0:  # In the rare cases min_val=max_val
                            splits = [max_val] * self.num_candidates
                        else:
                            splits = np.linspace(min_val, max_val, num=self.num_candidates, endpoint=False)

                    self.feature_split_candidates.append(splits)
                elif self.sketch_type == "log_uniform": # Combine log and uniform splits i.e if hist_bin = 32 then 16 log-uniform and 16 uniform
                    col = X[:,j]
                    data = np.log(1+(-np.nanmin(col))+col)
                    log_splits = np.exp(np.linspace(np.nanmin(data),np.nanmax(data), num=int(self.num_candidates/2), endpoint=False))+np.nanmin(col)-1 # For positive skews

                    if skew(col) < 0:
                        log_splits = -log_splits + np.nanmax(col) # For negative skews

                    step = (np.max(col) - np.min(col)) / (self.num_candidates/2)
                    unique_values = np.unique(X[:,j])
                    if self.bin_type != "all":
                        if len(unique_values) <= self.num_candidates/2:
                            splits = list(unique_values)  # If a feature has <= values than the number of bins then we can do exact splitting
                    if step == 0:  # In the rare cases min_val=max_val
                        splits = [max_val] * (int(self.num_candidates/2))
                    else:
                        splits = np.linspace(min_val, max_val, num=int(self.num_candidates/2), endpoint=False)

                    self.feature_split_candidates.append(np.sort(np.concatenate([log_splits, splits])))
                elif self.sketch_type == "uniform":
                    step = (max_val - min_val) / self.num_candidates
                    unique_values = np.unique(X[:,j])
                    if self.bin_type != "all":
                        if len(unique_values) <= self.num_candidates:
                            splits = list(unique_values)  # If a feature has <= values than the number of bins then we can do exact splitting
                    if step == 0:  # In the rare cases min_val=max_val
                        splits = [max_val] * self.num_candidates
                    else:
                        splits = np.linspace(min_val, max_val, num=self.num_candidates, endpoint=False) # Changed 19/01/2022: Should stop larger than q=hist_bins splits being formed for some features
                    self.feature_split_candidates.append(splits)
                elif self.sketch_type == "dp_hist":
                    if self.categorical_map is None or not self.categorical_map[j]:
                        # Form DP histogram
                        hist, bins = np.histogram(X[:,j], range=(min_val, max_val), bins=1024, density=False)
                        dp_hist = (hist + np.random.laplace(0, split_candidate_epsilon, hist.shape))/X.shape[0]

                        # Simple post-processing
                        dp_hist[dp_hist < 0] = 0
                        dp_hist = dp_hist / sum(dp_hist)

                        dp_quantiles = self._find_quantiles(dp_hist, bins)

                        self.feature_split_candidates.append(list(set(dp_quantiles)))
                    else:
                        self.feature_split_candidates.append([])
                elif self.sketch_type == "dp_quantiles":
                    if self.categorical_map is None or not self.categorical_map[j]:
                        quantiles = joint_exp(np.sort(X[:,j][~np.isnan(X[:,j])]), min_val, max_val, np.arange(start=0, stop=1, step=1/self.num_candidates), split_candidate_epsilon, swap=False)
                        self.feature_split_candidates.append(quantiles)
                    else:
                        self.feature_split_candidates.append([])
                elif self.sketch_type == "ldp_hist":
                    if self.categorical_map is None or not self.categorical_map[j]:
                        # Create bins
                        hist, bins = np.histogram(X[:, j], range=(min_val, max_val), bins=1024)
                        binned_features = np.digitize(X[:,j], bins)-1

                        # Create client server objects
                        client = FastLHClient(split_candidate_epsilon, 1024, 500, use_olh=True)
                        server = FastLHServer(split_candidate_epsilon, 1024, 500, use_olh=True)

                        # Estimate frequency of domain
                        privatised = [client.privatise(item) for item in binned_features]
                        server.aggregate_all(privatised)
                        ldp_hist = server.estimate_all(range(1024))

                        # Post-processing
                        ldp_hist[ldp_hist < 0] = 0
                        ldp_hist = ldp_hist / sum(ldp_hist)

                        # Find quantiles
                        ldp_quantiles = self._find_quantiles(ldp_hist, bins)
                        self.feature_split_candidates.append(list(set(ldp_quantiles)))
                    else:
                        self.feature_split_candidates.append([])
                elif self.sketch_type == "ldp_quantiles":
                    if self.categorical_map is None or not self.categorical_map[j]:
                        client = SWClient(split_candidate_epsilon)
                        server = SWServer(split_candidate_epsilon, smooth=False)
                        perturbed_data = [client.privatise((x-min_val)/(max_val-min_val)) for x in X[:, j]]
                        server.aggregate_all(perturbed_data)
                        density = server.estimate_density()
                        bins = np.arange(0,1, 1/1024)
                        ldp_quantiles = (np.array(self._find_quantiles(density, bins)) * (max_val - min_val) + min_val)
                        self.feature_split_candidates.append(list(set(ldp_quantiles)))
                    else:
                        self.feature_split_candidates.append([])
                elif self.sketch_type == "random_guess":
                    self.feature_split_candidates.append(np.sort(np.random.uniform(np.nanmin(X[:,j]), np.nanmax(X[:,j]), self.num_candidates)))
                elif self.sketch_type == "adaptive_hessian":
                    if round_num == 0:
                        # Do uniform splitting, no hessian information yet...
                        step = (max_val - min_val) / self.num_candidates
                        unique_values = np.unique(X[:,j])
                        if self.bin_type != "all" and len(unique_values) <= self.num_candidates:
                            splits = list(unique_values)  # If a feature has <= values than the number of bins then we can do exact splitting
                        if step == 0:  # In the rare cases min_val=max_val
                            splits = [max_val] * self.num_candidates
                        else:
                            # splits = np.arange(min_val, max_val, step=step)
                            splits = np.linspace(min_val, max_val, num=self.num_candidates, endpoint=False) # Changed 19/01/2022: Should stop larger than q=hist_bins splits being formed for some features
                        # print(j , splits)
                        self.feature_split_candidates.append(splits)
                    else:
                        current_splits = old_splits[j] # Get current splits
                        if j not in features_considering:
                            self.feature_split_candidates.append(copy(current_splits))
                            continue

                        new_splits = []
                        opt_freq = total_hess / self.num_candidates # Uniform hess bins...
                        hess_hist = np.clip(hessian_hist[j], 0, None)
                        total_freq = 0
                        for i, freq in enumerate(hess_hist): # For each bin and freq in the hessiaan histogram
                            total_freq += freq
                            if total_freq > opt_freq and i < len(current_splits)-1:
                                new_splits.append((current_splits[i], freq))
                                total_freq = 0
                                if len(current_splits) < self.num_candidates and len(new_splits) < self.num_candidates:
                                    new_splits.append((((current_splits[i] + current_splits[i+1])/2), 0))
                            elif i == len(current_splits)-1:
                                new_splits.append((current_splits[i], total_freq))
                        new_splits, _ = zip(*new_splits)
                        new_splits = np.sort(np.unique(new_splits))
                        if len(new_splits) < self.num_candidates and math.ceil(self.num_candidates / len(new_splits)) >= 1:
                            num_candidates = math.ceil(self.num_candidates / len(new_splits))
                            for i in range(len(new_splits)-1):
                                new_splits = np.concatenate([new_splits, np.linspace(new_splits[i], new_splits[i+1], num=num_candidates, endpoint=False)])
                        new_splits = np.sort(np.unique(new_splits))
                        self.feature_split_candidates.append(new_splits)

        if self.categorical_map is not None:
            for j in range(len(self.feature_split_candidates)):
                if self.categorical_map[j]:
                    self.feature_split_candidates[j] = np.unique(X[:,j])
