import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import gaussian_kde

from federated_gbdt.models.gbdt.private_gbdt import PrivateGBDT

from experiments.experiment_helpers.data_loader import DataLoader

def plot_feature_importance(model, feature_names, method="gain"):
    """
    Plots feature importance

    :param feature_names: List of feature names as strings (for plotting)
    :param method: Feature importance method to be used
    """
    x, y = zip(*model.feature_importance(method).most_common())
    plt.figure(figsize=(10, 10))
    plt.bar(feature_names[list(x)], y)
    plt.xticks(rotation=90)
    plt.xlabel("Feature")
    plt.ylabel(method)
    plt.title("Feature Importance - " + method)
    plt.show()

# Pass data, types of sketches to visualise and the features to visualise - Optional is to pass different bin nums to be printed
def visualise_quantiles(model, X, sketch_types, feature_list, hist_bins=[32]):
    """
    Helper method to visualise quantiles calculated via various methods

    :param X: Data
    :param sketch_types: List of quantile methods to be computed on features in X
    :param feature_list: List of features to visualise quantiles
    :param hist_bins: List of # of histogram bins to visualise
    """
    quantile_map = {}
    for hist_bin in hist_bins:
        model.split_candidate_manager.num_candidates = hist_bin
        for sketch_type in sketch_types:
            model.split_candidate_manager.sketch_type = sketch_type
            model.split_candidate_manager.find_split_candidates(X, 0)
            quantile_map[sketch_type] = model.split_candidate_manager.feature_split_candidates

    _, axes = plt.subplots(len(feature_list), len(sketch_types), figsize=(20,30))
    axes = np.array(axes).reshape(len(feature_list), len(sketch_types))
    print(axes.shape)
    for j, feature_index in enumerate(feature_list):
        # Create subplot grid...
        print("Feature j", X[:, j])
        for i, sketch_type in enumerate(quantile_map.keys()):
            # Plot feature dist
            sns.kdeplot(x=X[:, feature_index], ax=axes[j,i])
            # sns.histplot(x=X[:, j], stat="density", kde=True, hist=False)

            # x,y = kde.get_lines()[0].get_data()
            kde = gaussian_kde(X[:,feature_index][~np.isnan(X[:,feature_index])])

            quantiles = quantile_map[sketch_type][feature_index]
            # print(sketch_type, "quantiles:", len(quantiles))
            # print(sketch_type, "unique quantiles:", len(set(quantiles)))
            # print(sketch_type, quantiles, "\n")
            axes[j,i].vlines(quantiles, 0, kde(quantiles), colors="red", linestyles="--", linewidth=0.4)
            axes[j,i].set_xlim(left=np.nanmin(X[:,j]), right=np.nanmax(X[:,j]))
            axes[j,i].set_yticklabels([])
            axes[j,i].set_xticklabels([])
            y_label = axes[j,i].get_yaxis().get_label()
            y_label.set_visible(False)
            # axes[j,i].set_title("Density of feature " + str(feature_index) + "\n Quantile Method: " + sketch_type)

        if "uniform" in quantile_map.keys():
            uniform_quantiles = quantile_map["uniform"][feature_index]

            for k in quantile_map.keys():
                ldp_quantiles = quantile_map[k][feature_index]
                ldp_quantiles = np.sort(ldp_quantiles)
                total_mse = 0
                for i, q in enumerate(ldp_quantiles):
                    total_mse += np.min((uniform_quantiles-q)**2)
                    # total_mse += (uniform_quantiles[i]-q)**2

                print("Feature", feature_index, "Method:", k, "MSE:", total_mse/len(uniform_quantiles))

    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    dataloader = DataLoader()
    data = list(dataloader.load_datasets(["Credit 1"], remove_missing=True, return_dict=True, verbose=True).items())[0]
    X, X_test, y_train, y_test = data[1]
    X = X.to_numpy()
    model = PrivateGBDT()
    visualise_quantiles(model, X, ["uniform", "log"], [2,4,5,7])
