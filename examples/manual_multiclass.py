import sys

sys.path.append("../")

from federated_gbdt.models.gbdt.private_gbdt import PrivateGBDT
from federated_gbdt.core.loss_functions import SoftmaxCrossEntropyLoss
from experiments.experiment_helpers.data_loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from xgboost import XGBClassifier

# Load connect4 dataset
dataloader = DataLoader()
X_train, X_test, y_train, y_test = dataloader.load_datasets(
    ["connect_4"], return_dict=False
)[0]
onehot_y_test = OneHotEncoder(sparse_output=False).fit_transform(y_test.reshape(-1, 1))

# XGBoost baseline
xgb = XGBClassifier().fit(X_train, y_train)
xgb_probs = xgb.predict_proba(X_test)
xgb_pred = np.argmax(xgb_probs, axis=1)
print(f"XGBoost AUC - {roc_auc_score(onehot_y_test, xgb_probs)}")
print(f"XGBoost Accuracy - {accuracy_score(y_test, xgb_pred)}")
print("\n")

# PrivateGBDT (eps=0, non-private)
C = len(np.unique(y_train))  # C=3 classes for connect4
total_eps = 3
# scale privacy budget, here we have eps=0 (non-private) so scaling not needed
class_eps = total_eps / C
class_probs = []
for c in range(0, C):
    print(f"Training model... class {c} vs all")
    dp_method = "" if class_eps == 0 else "gaussian_cdp"
    xgb_model = PrivateGBDT(num_trees=100, epsilon=class_eps, dp_method=dp_method)
    y_train_c = (y_train == c).astype(int)  # one-vs-all for class k
    xgb_model = xgb_model.fit(X_train, y_train_c)
    class_probs.append(xgb_model.predict_proba(X_test)[:, 1])
y_probs = SoftmaxCrossEntropyLoss().predict(np.array(list(zip(*class_probs))))
y_pred = np.argmax(y_probs, axis=1)
print(
    f"PrivateGBDT (epsilon={total_eps}) AUC - {roc_auc_score(onehot_y_test, y_probs)}"
)
print(f"PrivateGBDT (epsilon={total_eps}) Accuracy - {accuracy_score(y_test, y_pred)}")
