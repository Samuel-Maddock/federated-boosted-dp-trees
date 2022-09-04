import numpy as np
import federated_gbdt.core.baseline_constants as consts
from abc import ABC, abstractmethod
import pandas
from sklearn.metrics import roc_auc_score, accuracy_score
import _pickle as pickle

class TreeBase(ABC):
    def __init__(self, min_samples_split=2,
                 max_depth=3, task_type=consts.CLASSIFICATION,
                 num_classes=-1):
        self.root = None  # Root node in dec. tree
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.task_type = task_type
        self.num_classes = num_classes
        self.training_method = "boosting"
        self.batched_update_size = 1
        self.trees = []
        self.loss = None

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def _build_tree(self, *args, **kwargs):
        """
        Build the tree model according to impurity and leaf value
        """
        pass

    def _convert_df(self, X):
        """

        :param X: Data as a Pandas DataFrame
        :return: X as a Numpy array
        """
        if isinstance(X, pandas.DataFrame):
            X = X.to_numpy()

        return X

    @staticmethod
    def predict_value(X, tree):
        out = np.zeros(X.shape[0])

        if tree.value is None:
            # Choose the feature that we will test
            feature_value = X[:, tree.feature_i]
            left_split = feature_value <= tree.threshold
            right_split = ~left_split

            # Determine if we will follow left or right branch
            out[left_split] = TreeBase.predict_value(X[left_split], tree.true_branch)
            out[right_split] = TreeBase.predict_value(X[right_split], tree.false_branch)
        else:
            out = np.repeat(tree.value[0], X.shape[0])

        return out

    def predict_over_trees(self, X, y):
        metrics = []
        for i in range(1, len(self.trees)+1):
            trees = self.trees[:i]
            y_pred = self.loss.predict(self.predict_weight(X, trees))
            auc = roc_auc_score(y, y_pred)
            acc = accuracy_score(y, (y_pred >= 0.5).astype("int"))
            metrics.append((auc,acc))
            print("Tree", i, "AUC :", auc )
            print("Tree", i, "Acc :", acc , "\n")

        return metrics

    def predict_weight(self, X, tree=None):
        """
        Predicts a weight (i.e for classification a non-sigmoided value) for each observation passed.
        By default this is calculated from the whole ensemble or from a specific tree if passed

        :param X: Data
        :param tree: Tree to predict weight from
        :return: Model prediction as a weight
        """
        X = self._convert_df(X)
        pred = np.zeros((X.shape[0], self.num_classes)) if self.num_classes > 2 else np.zeros(X.shape[0])
        trees = tree if tree is not None else self.trees

        preds = []
        for i, tree in enumerate(trees):
            pred += self.predict_value(X, tree)
            if self.training_method == "batched_boosting":
                if (i+1) % self.batched_update_size == 0: # Average current weights and add to preds
                    pred /= self.batched_update_size
                    preds.append(pred)
                    pred = np.zeros(X.shape[0])
                elif (i+1) == len(trees):
                    pred /= (i+1) % self.batched_update_size
                    preds.append(pred)
            elif self.early_stopping == "average_retrain" and (i+1) % (len(self.trees)/2) == 0:
                preds.append(pred)

        if self.training_method == "batched_boosting":
            # print("NUM OF BATCHES", len(preds))
            pred = np.add.reduce(preds)
        elif self.training_method == "rf":
            pred /= len(trees)
        elif self.early_stopping == "average_retrain":
            pred = (preds[0] + preds[1]) / 2

        return pred

    def predict(self, X):
        """ Classify samples one by one and return the set of labels """
        X = self._convert_df(X)
        return (self.predict_prob(X) >= 0.5).astype("int")

    def predict_prob(self, X):
        """
        For binary classification will return probabilities instead of raw weights

        :param X: Rows of observations to classify
        :return: A list of probabilities for each observation
        """
        X = self._convert_df(X)
        pred = self.predict_weight(X)
        if self.task_type == consts.CLASSIFICATION:
            pred = self.loss.predict(pred)
        return pred

    def _reset_tracking_attributes(self, checkpoint):
        return

    def save(self, filename, checkpoint=False):
        self._reset_tracking_attributes(checkpoint) # Otherwise saved file will be large...
        f = open(filename+".pkl", 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)
