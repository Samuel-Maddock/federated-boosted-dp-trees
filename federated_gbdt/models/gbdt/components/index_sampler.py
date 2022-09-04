import numpy as np
import math
import itertools

class IndexSampler():
    def __init__(self, subsample, row_sample_method, colsample_bytree, colsample_bylevel, colsample_bynode):
        self.subsample = subsample
        self.row_sample_method = row_sample_method
        self.colsample_bytree = colsample_bytree  # Number of features to uniformly sample per tree
        self.colsample_bylevel = colsample_bylevel  # Per level of a tree
        self.colsample_bynode = colsample_bynode  # Per node of a tree
        # Sampling is multiplicative i.e cols_sample_bytree * colsample_bylevel * colsample_bynode * d features are considered at each split

        self.feature_interaction_iter = None

    def sample(self, current_round, num_samples, num_features, max_depth, feature_interaction_method="cyclical", feature_interaction_k=""):
        """
        Helper method to perform sampling for the XGBoost model

        :param num_samples: The number of rows in the dataset
        :param num_features: The number of features
        :return: The sampled indexes for rows, features per tree and features per node according to the self.subsample,
            self.colsample_bytree and self.colsample_bylevel parameters
        """
        col_tree_sample = None
        col_level_sample = None
        row_sample = np.arange(num_samples)

        if self.subsample < 1:  # Sample rows per tree
            if self.row_sample_method == "wor":
                row_sample = np.random.choice(num_samples, size=round(num_samples * self.subsample), replace=False)
            elif self.row_sample_method == "wr":
                raise NotImplemented("With replacement sampling is not implemented")
            elif self.row_sample_method == "poisson":
                row_sample = np.where(np.random.binomial(1, self.subsample, size=num_samples)==1)[0]
            elif self.row_sample_method == "disjoint":
                subset_size = math.ceil(num_samples*self.subsample)
                start = (((current_round) % math.ceil(num_samples / subset_size))) * subset_size
                end = start + subset_size
                row_sample = self.disjoint[start:end]

        if self.colsample_bytree < 1:  # Sample columns per tree
            col_tree_sample = np.random.choice(num_features, size=math.ceil(num_features * self.colsample_bytree), replace=False)
        if self.colsample_bylevel < 1 and self.colsample_bytree < 1:  # Sample columns per level of the tree (taking into account the cols alreaady sampled for the current tree)
            col_level_sample = [np.random.choice(range(0, len(col_tree_sample)), size=math.ceil(len(col_tree_sample) * self.colsample_bylevel), replace=False) for i in range(0, self.max_depth + 2)]
        elif self.colsample_bylevel < 1:
            col_level_sample = [np.random.choice(num_features, size=math.ceil(num_features* self.colsample_bylevel), replace=False) for i in range(0, self.max_depth + 2)]

        if "cyclical" in feature_interaction_method:
            if feature_interaction_k == 1:
                col_tree_sample = [current_round % num_features]
            elif feature_interaction_k:
                if self.feature_interaction_iter is None:
                    self.feature_interaction_iter = itertools.cycle(itertools.combinations(list(range(0, num_features)), feature_interaction_k)) # precompute
                col_tree_sample = list(next(self.feature_interaction_iter))
        elif "random" in feature_interaction_method:
            if feature_interaction_k:
                col_tree_sample = np.random.choice(num_features, size=feature_interaction_k, replace=False) # Choose k features at random

        return row_sample, col_tree_sample, col_level_sample
