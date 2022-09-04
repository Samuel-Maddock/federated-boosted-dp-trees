import time
import pandas as pd
import numpy as np
import os
from collections import Counter
import itertools
import pathos.multiprocessing as mp

dirname = os.path.dirname(__file__)
base_path = os.path.split(dirname)[0] + os.sep + "experiment_data" + os.sep

from federated_gbdt.models.gbdt.private_gbdt import PrivateGBDT
from sklearn.metrics import roc_auc_score, roc_curve,\
    accuracy_score, f1_score, average_precision_score, mean_squared_error, r2_score, classification_report

import matplotlib.pyplot as plt
import tqdm

class ExperimentRunner:
    def __init__(self, performance=False, output_train_monitor=False, data_path=None):
        self.performance = performance
        self.data_path = base_path if data_path is None else data_path
        self.output_train_monitor = output_train_monitor
        self.time_col_names = []

    def default_args(self, args):
        final_args = args.copy()
        final_args.setdefault("num_trees", 10)
        final_args.setdefault("max_depth", 6)
        final_args.setdefault("sketch_type", "uniform")
        # final_args.setdefault("remove_categorical", False)
        return final_args

    def plot_auc(self, y, y_pred):
        fpr, tpr, _ = roc_curve(y,  y_pred)
        print(fpr, tpr, _)
        auc = roc_auc_score(y, y_pred)
        plt.plot(fpr,tpr,label="EXP, auc="+str(auc))
        plt.legend(loc=4)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    def experiment_job(self, params):
        i = params[0]
        args = params[1]
        hist_bin = params[2]
        data = params[3]
        iterated_param_name = params[4]

        name = data[0]
        X_train, X_test, y_train, y_test = data[1]

        final_args = self.default_args(args)
        if not self.performance:
            print("Starting experiment", params[5], "iter=", i, "args=",args)

        sketch_type = "uniform"
        if hist_bin == "sketch":
            hist_bin = 64
            sketch_type = "sketch"

        start = time.time()
        model = PrivateGBDT(hist_bin=hist_bin, categorical_map=None, output_train_monitor=self.output_train_monitor, **final_args).fit(X_train, y_train)
        end = time.time()
        train_time = end-start
        if not self.performance:
            print("Training Time:", train_time)

        start = time.time()
        y_pred = model.predict_prob(X_test)
        y_train_pred = model.loss.predict(model.y_weights)

        if args.get("task_type", "classification") == "classification":
            if len(np.unique(y_train)) > 2: # multi-class
                train_acc, test_acc = 0, 0
                train_auc, test_auc = 0, 0
                train_pr, test_pr = 0, 0
                f1 = 0

                y_test_pred_classes = np.argmax(y_pred, axis=1)
                print("Predicted Class Distribution", Counter(y_test_pred_classes))
                print("Actual Class Distribution", Counter(y_test))

                print("Test Accuracy", accuracy_score(y_test, np.argmax(y_pred, axis=1)))
                print(classification_report(y_test, y_test_pred_classes))
            else:
                y_pred_class = (y_pred >= 0.5).astype("int")
                y_train_pred_class = (y_train_pred >= 0.5).astype("int")
                train_acc = accuracy_score(y_train, y_train_pred_class)
                train_auc = roc_auc_score(y_train, y_train_pred)
                train_pr = average_precision_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_pred_class)
                test_auc = roc_auc_score(y_test, y_pred)
                test_pr = average_precision_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred_class)

                if not self.performance:
                    print(classification_report(y_test, y_pred_class))
                    # self.plot_auc(y_test, y_pred)
                    print("Train accuracy", train_acc)
                    print("Train AUC", train_auc)
                    print("PR AUC Score", test_pr)
                    print("F1 score", f1)
        else: # Regression
            # TODO: Fix the metrics naming hack...
            test_acc = mean_squared_error(y_pred, y_test)
            train_acc = mean_squared_error(y_train_pred, y_train)
            test_auc = r2_score(y_pred, y_test)
            train_auc = r2_score(y_train_pred, y_train)

            if not self.performance:
                print("Train MSE", train_acc)
                print("Train RMSE", mean_squared_error(y_train_pred, y_train, squared=False))
                print("Train R2", train_auc)
                print("Test MSE", test_acc)
                print("Test RMSE", mean_squared_error(y_pred, y_test, squared=False))
                print("Test R2", test_auc)

        end = time.time()
        if not self.performance:
            print("Test Time:", end-start)

            if sketch_type == "sketch":
                hist_bin = "sketch"  # For plotting reasons

            if not self.performance:
                print(name, "iter=", i, "hist_bin=", hist_bin, "args:", args, )
                print("Test Accuracy:", test_acc, "Test AUC:", test_auc, "\n")
                print("Final number of trees trained:", len(model.trees))

        # Rounds
        client_rounds_sent = model.train_monitor.client_rounds_sent
        client_rounds_received = model.train_monitor.client_rounds_received

        avg_client_rounds_sent = np.mean(client_rounds_sent)
        std_client_rounds_sent = np.std(client_rounds_sent)
        min_client_rounds_sent = np.min(client_rounds_sent)
        max_client_rounds_sent = np.max(client_rounds_sent)
        total_client_rounds_sent = np.sum(client_rounds_sent)

        avg_client_rounds_received = np.mean(client_rounds_received)
        std_client_rounds_received = np.std(client_rounds_received)
        min_client_rounds_received = np.min(client_rounds_received)
        max_client_rounds_received = np.max(client_rounds_received)
        total_client_rounds_received = np.sum(client_rounds_received)

        # Comm Size
        client_sent = list(model.train_monitor.client_payload_sent)
        client_received = list(model.train_monitor.client_payload_received)

        avg_client_rounds_sent_payload = np.mean(client_sent)
        std_client_rounds_sent_payload = np.std(client_sent)
        min_client_rounds_sent_payload = np.min(client_sent)
        max_client_rounds_sent_payload = np.max(client_sent)
        total_client_rounds_sent_payload = np.sum(client_sent)

        avg_client_rounds_received_payload = np.mean(client_received)
        std_client_rounds_received_payload = np.std(client_received)
        min_client_rounds_received_payload = np.min(client_received)
        max_client_rounds_received_payload = np.max(client_received)
        total_client_rounds_received_payload = np.sum(client_received)

        comm_stats = [avg_client_rounds_sent,std_client_rounds_sent, min_client_rounds_sent, max_client_rounds_sent, total_client_rounds_sent,
                      avg_client_rounds_received, std_client_rounds_received, min_client_rounds_received, max_client_rounds_received, total_client_rounds_received,
                      avg_client_rounds_sent_payload, std_client_rounds_sent_payload, min_client_rounds_sent_payload, max_client_rounds_sent_payload, total_client_rounds_sent_payload,
                      avg_client_rounds_received_payload,std_client_rounds_received_payload, min_client_rounds_received_payload, max_client_rounds_received_payload, total_client_rounds_received_payload]

        client_time_dict = model.train_monitor.client_time_dict
        server_time_dict = model.train_monitor.server_time_dict

        time_vals = list(client_time_dict.values()) + list(server_time_dict.values())

        return [hist_bin, name, model.X.shape[0], model.X.shape[1], str(args), False, final_args[iterated_param_name], len(model.trees), train_auc, test_auc,
                                     train_pr, test_pr, train_acc, test_acc, f1, train_time] + comm_stats + time_vals

    def run(self, datasets, iters, arg_options, hist_sizes, iterated_param=None, categorical_map=None, filename="experiment",
            save_data=False, evaluate_all_trees=False, n_jobs=None):

        if iterated_param is not None: # An iterated param is one like a budget allocation that needs to be iterated through an added to every arg option
            new_arg_options = []
            for args in arg_options:
                for v in iterated_param[1]:
                    new_args = args.copy()
                    new_args[iterated_param[0]] = v
                    new_arg_options.append(new_args)
        else:
            new_arg_options = arg_options
            iterated_param = ("num_trees", None) # Otherwise set to some default parameter that isn't needed to be iterated over

        arg_options = new_arg_options
        total_num_experiments = len(datasets) * iters * len(arg_options) * len(hist_sizes)
        print("Beginning simulation... Total number of experiments:", total_num_experiments)
        print(f"Data Path {self.data_path}")
        global_time = time.time()

        if n_jobs is None:
            pool = mp.ProcessingPool()
        else:
            pool = mp.ProcessingPool(nodes=n_jobs)

        res = []

        for dataset_num, data in enumerate(datasets):
            param_list = list(itertools.product(range(0, iters), arg_options, hist_sizes, [data], [iterated_param[0]]))
            final_param_list = []

            for i in range(0, len(param_list)):
                final_param_list.append(param_list[i]+(len(param_list)*dataset_num + i,))

            if self.performance:
                for x in tqdm.tqdm(pool.imap(self.experiment_job, final_param_list, pool=pool), total=len(param_list)):
                    res.append(x)
            else:
                res += pool.map(self.experiment_job, final_param_list, pool=pool)

            if save_data:
                base_columns = ["hist_bin", "dataset", "n", "m", "args",  # main params
                                                        "remove_categorical", iterated_param[0], "final_trees", # other params
                                                        "train_auc", "test_auc", # metrics
                                                        "train_pr", "test_pr",
                                                        "train_acc", "test_acc",
                                                        "f1_score",
                                                        "training_time"]
                comm_stats = ["avg_client_rounds_sent", "std_client_rounds_sent", "min_client_rounds_sent", "max_client_rounds_set", "total_client_rounds_sent",
                              "avg_client_rounds_received", "std_client_rounds_received", "min_client_rounds_received", "max_client_rounds_received", "total_client_rounds_received",
                              "avg_client_rounds_sent_payload", "std_client_rounds_sent_payload", "min_client_rounds_sent_payload", "max_client_rounds_sent_payload", "total_client_rounds_sent_payload",
                              "avg_client_rounds_received_payload","std_client_rounds_received_payload", "min_client_rounds_received_payload", "max_client_rounds_received_payload", "total_client_rounds_received_payload"]

                time_stats = ["t_client_histogram_building",  "t_client_computing_gradients", 't_client_initialise_private_histogram', "t_client_forming_grad_histogram", "t_client_retrieving_grads_for_node",
                                 "t_server_initial_split_candidates", "t_server_privacy_accountant_initialisation", "t_server_init_model_weights", "t_server_split_candidates",
                                 "t_server_pre_tree_ops", "t_server_post_tree ops", "t_server_initialise_priv_hist", "t_server_adding_noise_to_hist",
                                 "t_server_sampling_features", "t_server_calculating_internal_split", "t_server_split_constraints", "t_server_leaf_weight"]

                columns = base_columns + comm_stats + time_stats
                pd.DataFrame(res, columns=columns).to_csv(self.data_path + filename + ".csv", index=False)
        pool.close()
        end_time = time.time()
        print("Experiment finished total time:", end_time-global_time)

    def _old_run(self, datasets, iters, arg_options, hist_sizes, iterated_param=None, categorical_map=None, filename="experiment",
            save_data=False, evaluate_all_trees=False):
        experiment_stats = []
        counter = 0

        if iterated_param is not None: # An iterated param is one like a budget allocation that needs to be iterated through an added to every arg option
            new_arg_options = []
            for args in arg_options:
                for v in iterated_param[1]:
                    new_args = args.copy()
                    new_args[iterated_param[0]] = v
                    new_arg_options.append(new_args)
        else:
            new_arg_options = arg_options
            iterated_param = ("num_trees", None) # Otherwise set to some default parameter that isn't needed to be iterated over

        arg_options = new_arg_options

        total_num_experiments = len(datasets) * iters * len(arg_options) * len(hist_sizes)
        print("Beginning simulation... Total number of experiments:", total_num_experiments)
        global_time = time.time()

        for data in datasets:
            name = data[0]
            X_train, X_test, y_train, y_test = data[1]

            cat_map = None
            if categorical_map is not None:
                cat_map = categorical_map[name]

            for i in range(0, iters):
                for args in arg_options:
                    final_args = self.default_args(args)
                    remove_categorical = final_args["remove_categorical"]
                    final_args.pop("remove_categorical")
                    if remove_categorical:
                        # Code removes categorical features for testing...
                        X_train = X_train.to_numpy()[:, ~np.array(categorical_map[name])]
                        X_test = X_test.to_numpy()[:, ~np.array(categorical_map[name])]

                    for hist_bin in hist_sizes:
                        counter += 1
                        print("Starting experiment", counter, " - " + str(round(((counter - 1) / total_num_experiments)*100, 2)) + "% complete")

                        sketch_type = "uniform"
                        if hist_bin == "sketch":
                            hist_bin = 64
                            sketch_type = "sketch"

                        start = time.time()
                        model = PrivateGBDT(hist_bin=hist_bin, categorical_map=cat_map, **final_args).fit(X_train, y_train)
                        end = time.time()
                        train_time = end-start
                        if not self.performance:
                            print("Training Time:", train_time)

                        start = time.time()
                        y_pred = model.predict_prob(X_test)
                        y_train_pred = model.loss.predict(model.y_weights)

                        if args.get("task_type", "classification") == "classification":
                            if len(np.unique(y_train)) > 2: # multi-class
                                train_acc, test_acc = 0, 0
                                train_auc, test_auc = 0, 0
                                train_pr, test_pr = 0, 0
                                f1 = 0

                                y_test_pred_classes = np.argmax(y_pred, axis=1)
                                print("Predicted Class Distribution", Counter(y_test_pred_classes))
                                print("Actual Class Distribution", Counter(y_test))

                                print("Test Accuracy", accuracy_score(y_test, np.argmax(y_pred, axis=1)))
                                print(classification_report(y_test, y_test_pred_classes))
                            else:
                                y_pred_class = (y_pred >= 0.5).astype("int")
                                y_train_pred_class = (y_train_pred >= 0.5).astype("int")
                                train_acc = accuracy_score(y_train, y_train_pred_class)
                                train_auc = roc_auc_score(y_train, y_train_pred)
                                train_pr = average_precision_score(y_train, y_train_pred)
                                test_acc = accuracy_score(y_test, y_pred_class)
                                test_auc = roc_auc_score(y_test, y_pred)
                                test_pr = average_precision_score(y_test, y_pred)
                                f1 = f1_score(y_test, y_pred_class)

                                if not self.performance:
                                    print(classification_report(y_test, y_pred_class))
                                    # self.plot_auc(y_test, y_pred)
                                    print("Train accuracy", train_acc)
                                    print("Train AUC", train_auc)
                                    print("PR AUC Score", test_pr)
                                    print("F1 score", f1)
                        else: # Regression
                            # TODO: Fix the metrics naming hack...
                            test_acc = mean_squared_error(y_pred, y_test)
                            train_acc = mean_squared_error(y_train_pred, y_train)
                            test_auc = r2_score(y_pred, y_test)
                            train_auc = r2_score(y_train_pred, y_train)

                            if not self.performance:
                                print("Train MSE", train_acc)
                                print("Train RMSE", mean_squared_error(y_train_pred, y_train, squared=False))
                                print("Train R2", train_auc)
                                print("Test MSE", test_acc)
                                print("Test RMSE", mean_squared_error(y_pred, y_test, squared=False))
                                print("Test R2", test_auc)

                        end = time.time()
                        if not self.performance:
                            print("Test Time:", end-start)

                        if evaluate_all_trees:
                            test_auc_list = model.predict_over_trees(X_test, y_test)
                            for i, test_auc in enumerate(test_auc_list):
                                experiment_stats.append([hist_bin, name, str(args), remove_categorical, i+1, train_auc, test_auc, train_acc, test_acc, train_time])

                        if sketch_type == "sketch":
                            hist_bin = "sketch"  # For plotting reasons

                        if not self.performance:
                            print(name, "iter=", i, "hist_bin=", hist_bin, "args:", args, )
                            print("Test Accuracy:", test_acc, "Test AUC:", test_auc, "\n")
                            print("Final number of trees trained:", len(model.trees))


                        if not evaluate_all_trees:
                            experiment_stats.append([hist_bin, name, str(args), remove_categorical, final_args[iterated_param[0]], len(model.trees),
                                                     train_auc, test_auc,
                                                     train_pr, test_pr,
                                                     train_acc, test_acc,
                                                     f1, train_time])

                        if save_data:
                            pd.DataFrame(experiment_stats, columns=["hist_bin", "dataset", "args",  # main params
                                                                    "remove_categorical", iterated_param[0], "final_trees", # other params
                                                                    "train_auc", "test_auc", # metrics
                                                                    "train_pr", "test_pr",
                                                                    "train_acc", "test_acc",
                                                                    "f1_score",
                                                                    "training_time"]).to_csv(self.data_path + filename + ".csv", index=False)

        end_time = time.time()
        print("Experiment finished total time:", end_time-global_time)
        return model  # For debugging
