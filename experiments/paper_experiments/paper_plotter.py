import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import shutil
import matplotlib.lines as mlines

from collections import defaultdict

sns.set_theme(style="whitegrid")

def set_fontsize(size=14):
    tex_fonts = {
        #"text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": size,
        "font.size": size,
        "legend.fontsize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size
    }
    plt.rcParams.update(tex_fonts)

def exp_plot():
    x = np.linspace(1, 100, num=10)
    y = np.ones(len(x))
    sns.lineplot(x=x, y=y)
    plt.savefig("./paper_plots/fig1.eps")

# Add columns from args for plotting
def process_df_for_plotting(df):
    df["args"] = df["args"].str.replace("'gaussian_cdp", "DP")

    dp_method = df["args"].str.split("'dp_method':",expand=True)[1].str.split(",", expand=True)[0].str.strip("'")
    df["dp_method"] = dp_method

    df["split_method"] = df["args"].str.split("'split_method':",expand=True)[1].str.split(",", expand=True)[0].str.strip("'")
    df["split_method"] = df["split_method"].str.replace(" ", "")

    max_depth = df["args"].str.split("'max_depth':",expand=True)[1].str.split(",", expand=True)[0].str.strip("'")
    df["max_depth"] = max_depth.str.replace(" ", "")

    if not "num_trees" in df.columns:
        df["num_trees"] = df["args"].str.split("'num_trees':",expand=True)[1].str.split(",", expand=True)[0].str.strip("'").str.strip(" ")

    if not "epsilon" in df.columns:
        df["epsilon"] = df["args"].str.split("'epsilon':",expand=True)[1].str.split(",", expand=True)[0].str.strip("'").str.strip(" ")

    df["dataset"] = df["dataset"].str.split("_", expand=True)[0]

    # Feature Interaction...
    df["ebm"] = "False"
    if df["args"].str.contains("ebm").sum() > 0 and df["args"].str.contains("full_ebm").sum() == 0:
        df["ebm"][df["args"].str.contains("ebm")] = df["args"].str.split("'ebm':", expand=True)[1].str.split(",", expand=True)[0]

    df["feature_interaction_method"] = "standard"
    if df["args"].str.contains("feature_interaction_method").sum() > 0:
        df["feature_interaction_method"][df["args"].str.contains("feature_interaction_method")] = df["args"].str.split("'feature_interaction_method':", expand=True)[1].str.split(",", expand=True)[0]

    df["feature_interaction_k"] = "d"
    if df["args"].str.contains("feature_interaction_k").sum() > 0:
        df["feature_interaction_k"][df["args"].str.contains("feature_interaction_k")] = df["args"].str.split("'feature_interaction_k':", expand=True)[1].str.split(",", expand=True)[0]

    df["weight_update"] = ""
    if df["args"].str.contains("weight_update_method").sum() > 0:
        df["weight_update"][df["args"].str.contains("weight_update_method")] = df["args"].str.split("'weight_update_method':",expand=True)[1].str.split(",", expand=True)[0].str.strip("'").str.strip(" ")
        df["weight_update"][df["args"].str.contains("rf")]  = "rf"

    df["split_method"] = df["split_method"].str.replace("'", "")
    df["weight_update"] = df["weight_update"].str.replace("'", "")

    df["gradient_budgets"] = ""
    if df["args"].str.contains("gradient_budgets").sum() > 0:
        df["gradient_budgets"][df["args"].str.contains("gradient_budgets")] = df["args"].str.split("'gradient_budgets':",expand=True)[1].str.split(",", expand=True)[0].str.strip("'").str.strip(" ")

    df["batch_size"] = df["num_trees"].astype("float32")
    if df["args"].str.contains("batched_update_size").sum() > 0:
        df["batch_size"][df["args"].str.contains("batched_update_size")] = df[df["args"].str.contains("batched_update_size")]["args"].str.split("'batched_update_size':",expand=True)[1].str.split(",", expand=True)[0].str.strip("'")
    df["batch_perc"] = df["batch_size"].astype("float32") / df["num_trees"].astype("float32")

    df["training_method"] = "boosting"
    if df["args"].str.contains("training_method").sum() > 0:
        df["training_method"][df["args"].str.contains("training_method")] = df["args"].str.split("'training_method':",expand=True)[1].str.split(",", expand=True)[0].str.strip("'")

    df["vec_mech"] = "False"
    if df["args"].str.contains("vector_mechanism").sum() > 0:
        df["vec_mech"][df["args"].str.contains("vector_mechanism")] = "True"

    df["sketch_type"] = "uniform"
    if df["args"].str.contains("sketch_type").sum() > 0:
        df["sketch_type"] = df["args"].str.split("'sketch_type':",expand=True)[1].str.split(",", expand=True)[0].str.strip("'")

    df["sketch_type"][df["sketch_type"].isna()] = "uniform"
    df["sketch_type"] = df["sketch_type"].str.replace(" '", " ")

    df["sketch_rounds"] = "inf"
    if df["args"].str.contains("sketch_rounds").sum() > 0:
        df["sketch_rounds"] = df["args"].str.split("'sketch_rounds':",expand=True)[1].str.split(",", expand=True)[0].str.strip("'")

    df["quantize_q"] = "float64"
    if df["args"].str.contains("quantize_q").sum() > 0:
        df["quantize_q"] = df["args"].str.split("'quantize_q':",expand=True)[1].str.split(",", expand=True)[0].str.strip("'")

    return df


def clear_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)

# ============================================= E1 + E2 - Split Methods =============================================

# Vary T, D, eps
def plot_split_methods_with_update(in_path="./paper_results/E1_dp_split_methods_with_update_final.csv",
                                   out_path="./paper_plots/", replication=False, dataset="Credit 1", show_dp=True, y_lims=None, legends=None):
    if y_lims is None:
        y_lims = [None]*3

    if legends is None:
        legends = [True]*3

    if not replication:
        df1 = pd.read_csv(in_path)
        df2 = pd.read_csv("./paper_results/non_dp_split_methods_with_update.csv")
        df = pd.concat([df1, df2])
        df = df.reset_index()
    else:
        df = pd.read_csv(in_path)

    df = process_df_for_plotting(df)

    df["args"] = df["dp_method"]
    df["args"] = df["args"].str.replace("'", "")
    df["args"] = df["args"].str.replace(" ", "")

    arg_map = {"DP" : "DP",}
    df = df.replace({"args":arg_map})

    # df["args"] += "-" + df["weight_update"] + " " + df["split_method"] + " " + df["gradient_budgets"]
    df["args"] += "-" + df["weight_update"] + " " + df["split_method"] + " "

    # df = df[df["args"].str.contains("DP")]

    df["args"] = df["args"].str.replace(" ", "")

    arg_map = {"DP-xgboosttotally_random" : "DP-TR Newton",
       "DP-xgboostpartially_random" : "DP-PR Newton",
       "DP-xgboosthist_based" : "DP-Hist Newton",
        "-xgboosttotally_random" : "TR Newton",
       "-xgboostpartially_random" : "PR Newton",
       "-xgboosthist_based" : "Hist Newton",}

    df = df.replace({"args":arg_map})

    # ================= Vary T ==================
    filter_df = df[df["dataset"] == dataset]

    max_depth = "4"
    filter_df = filter_df[filter_df["max_depth"] == max_depth]

    # trees = "200"
    # df = df[df["num_trees"] == trees]

    # epsilon = ""
    epsilon = 1
    # filter_df = filter_df[filter_df["epsilon"] == epsilon]
    if show_dp:
        filter_df = pd.concat([filter_df[filter_df["epsilon"] == epsilon], filter_df[filter_df["epsilon"] == 0]])
    else:
        filter_df = filter_df[filter_df["epsilon"] == epsilon]

    # cm = sns.color_palette("Blues_r", 10) + sns.color_palette("Reds", 10) + sns.color_palette("Purples",1)
    cm = None

    filter_df = filter_df[filter_df["args"].str.contains("Newton")] # xgb update

    filter_df["Split Method"] = filter_df["args"]
    filter_df["Type"] = "Test"
    filter_df["auc"] = filter_df["test_auc"]

    new_df = filter_df[["num_trees", "auc", "Split Method", "Type"]].copy()
    new_df["auc"] = filter_df["train_auc"]
    new_df["Type"] = "Train"
    filter_df = filter_df[["num_trees", "auc", "Split Method", "Type"]]

    plot_df = pd.concat([filter_df, new_df]).reset_index()

    ax = sns.lineplot(data=plot_df, x="num_trees", y="auc", hue="Split Method", style="Type", palette=cm)

    # sns.lineplot(data=filter_df, x="num_trees", y="test_auc", hue="test_args", palette=cm)
    # sns.lineplot(data=filter_df, x="num_trees", y="train_auc", hue="train_args", palette=cm, linestyle="--")
    plt.xlabel("Number of trees (T)")
    plt.ylabel("Test AUC")

    if not legends[0]:
        ax.get_legend().remove()

    plt.tight_layout()
    plt.savefig(out_path + "vary_t_" + dataset + ".pdf")
    #plt.show()
    plt.clf()

    # ================= Vary D ==================
    filter_df = df[df["dataset"] == dataset]

    # max_depth = "4"
    # filter_df = filter_df[filter_df["max_depth"] == max_depth]

    filter_df_hist = filter_df[filter_df["args"].str.contains("Hist") | filter_df["args"].str.contains("PR") ]
    filter_df_hist = filter_df_hist[filter_df_hist["num_trees"] == "25"]

    filter_df_other = filter_df[filter_df["args"].str.contains("TR")]
    filter_df_other = filter_df_other[filter_df_other["num_trees"] == "300"]

    filter_df = pd.concat([filter_df_other, filter_df_hist])

    # trees = "200"
    # df = df[df["num_trees"] == trees]

    # epsilon = ""
    epsilon = 1
    filter_df = filter_df[filter_df["epsilon"] == epsilon]
    filter_df = filter_df[filter_df["args"].str.contains("Newton")] # xgb update

    filter_df["Split Method"] = filter_df["args"]
    filter_df["Type"] = "Test"
    filter_df["auc"] = filter_df["test_auc"]

    new_df = filter_df[["max_depth", "auc", "Split Method", "Type"]].copy()
    new_df["auc"] = filter_df["train_auc"]
    new_df["Type"] = "Train"
    filter_df = filter_df[["max_depth", "auc", "Split Method", "Type"]]
    plot_df = pd.concat([filter_df, new_df]).reset_index()

    ax = sns.lineplot(data=plot_df, x="max_depth", y="auc", hue="Split Method", style="Type", palette=cm)
    leg = plt.legend( loc = 'lower right')

    if not legends[1]:
        ax.get_legend().remove()

    plt.xlabel("Maximum Depth (d)")
    plt.ylabel("Test AUC")
    plt.ylim(y_lims[1])
    plt.tight_layout()
    plt.savefig(out_path + "vary_D_" + dataset + ".pdf")
    #plt.show()
    plt.clf()

    # ================= Vary eps ==================
    filter_df = df[df["dataset"] == dataset]

    max_depth = "3"
    filter_df = filter_df[filter_df["max_depth"] == max_depth]

    filter_df = filter_df[filter_df["epsilon"] != 0] # Will block non-dp from being plotted

    filter_df_hist = filter_df[filter_df["args"].str.contains("Hist") | filter_df["args"].str.contains("PR") ]
    filter_df_hist = filter_df_hist[filter_df_hist["num_trees"] == "25"]

    filter_df_other = filter_df[filter_df["args"].str.contains("TR")]
    filter_df_other = filter_df_other[filter_df_other["num_trees"] == "300"]

    filter_df = pd.concat([filter_df_other, filter_df_hist])

    cm = None
    filter_df = filter_df[filter_df["args"].str.contains("Newton")] # xgb update

    filter_df["Split Method"] = filter_df["args"]
    filter_df["Type"] = "Test"
    filter_df["auc"] = filter_df["test_auc"]

    new_df = filter_df[["epsilon", "auc", "Split Method", "Type"]].copy()
    new_df["auc"] = filter_df["train_auc"]
    new_df["Type"] = "Train"
    filter_df = filter_df[["epsilon", "auc", "Split Method", "Type"]]
    plot_df = pd.concat([filter_df, new_df]).reset_index()
    leg = plt.legend( loc = 'lower right')

    if not legends[2]:
        ax.get_legend().remove()

    sns.lineplot(data=plot_df, x="epsilon", y="auc", hue="Split Method", style="Type", palette=cm)
    plt.xlabel("Privacy Budget ($\epsilon$)")
    plt.ylabel("Test AUC")
    plt.savefig(out_path + "vary_e_" + dataset + ".pdf")
    set_fontsize()
    #plt.show()
    plt.clf()

# Displays latex for Table 3
def table_split_methods_with_update(epsilon=0.5, max_depth="4"):
    df = pd.read_csv("./paper_results/E1_dp_split_methods_with_update_final.csv")
    # df2 = pd.read_csv("./paper_results/non_dp_split_methods_with_update.csv")
    # df = pd.concat([df1, df2])
    # df = df.reset_index()
    df = process_df_for_plotting(df)

    df["args"] = df["dp_method"]
    df["args"] = df["args"].str.replace("'", "")
    df["args"] = df["args"].str.replace(" ", "")

    arg_map = {"DP" : "DP-XGBoost",}
    df = df.replace({"args":arg_map})

    # df["args"] += df["split_method"] + " " + df["weight_update"] + " " + df["gradient_budgets"]
    df["args"] += df["split_method"] + " " + df["weight_update"]

    # Filter plot params
    # dataset = "Credit 1"
    # df = df[df["dataset"] == dataset]
    #

    df = df[df["max_depth"] == max_depth]

    # trees = "200"
    # df = df[df["num_trees"] == trees]

    # epsilon = ""
    df = df[df["epsilon"] == epsilon]

    filter_df_hist = df[df["args"].str.contains("hist_based") | df["args"].str.contains("partially_random") ]
    filter_df_hist = filter_df_hist[filter_df_hist["num_trees"] == "25"]

    filter_df_other = df[df["args"].str.contains("totally_random")]
    filter_df_other = filter_df_other[filter_df_other["num_trees"] == "200"]

    df = pd.concat([filter_df_other, filter_df_hist])

    df = df[df["args"].str.contains("DP")]
    # print(df.groupby(["weight_update"]).mean()["test_auc"].astype("str") + " +- " + df.groupby(["weight_update"]).std()["test_auc"].astype("str"))

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        means = df.groupby(["dataset", "split_method", "weight_update"]).mean()["test_auc"].round(4).astype("str")
        sds = df.groupby(["dataset", "split_method", "weight_update"]).std()["test_auc"].round(4).astype("str")
        table = means + " +- " + sds

        max_vals = df.groupby(["dataset", "split_method", "weight_update"]).mean()["test_auc"].round(4).groupby(["dataset", "split_method"]).max().astype("str")
        # print(max_vals)
        # means[means.isin(max_vals)] = "\\textbf{" + means[means.isin(max_vals)] + "}"

        for val in max_vals.values:
            table[means==val] = "\\textbf{" + table[table.str.contains(val)] + "}"

        datasets, column_index, _ = zip(*table.index)
        # print("COLUMN INDEX", _)
        column_index = [["Hist"]*3 + ["PR"]*3 + ["TR"]*3, ["Gradient", "Averaging", "Newton"]*3]

        datasets = list(dict.fromkeys(datasets))
        table = pd.DataFrame(table.values.reshape(len(datasets), -1), columns=column_index, index=datasets)
        # table = pd.DataFrame(table.values.reshape(-1, len(datasets)), columns=datasets, index=column_index)
        print(table)
        print(table.transpose().to_latex(escape=False))
        print("\n")

# Not used...
def table_low_eps_split_methods_with_update():
    df1 = pd.read_csv("./paper_results/dp_split_methods_with_update_final.csv")
    df2 = pd.read_csv("./paper_results/non_dp_split_methods_with_update.csv")

    df = pd.concat([df1, df2])
    df = df.reset_index()
    df = process_df_for_plotting(df)

    df["args"] = df["dp_method"]
    df["args"] = df["args"].str.replace("'", "")
    df["args"] = df["args"].str.replace(" ", "")

    arg_map = {"DP" : "DP-XGBoost",}
    df = df.replace({"args":arg_map})

    df["args"] += df["split_method"] + " " + df["weight_update"] + " " + df["gradient_budgets"]

    # Filter plot params
    # dataset = "Credit 1"
    # df = df[df["dataset"] == dataset]
    #

    max_depth = "4"
    df = df[df["max_depth"] == max_depth]

    # trees = "200"
    # df = df[df["num_trees"] == trees]

    # epsilon = ""
    # epsilon = 0.5
    # df = df[df["epsilon"] == epsilon]

    filter_df_hist = df[df["args"].str.contains("hist_based") | df["args"].str.contains("partially_random") ]
    filter_df_hist = filter_df_hist[filter_df_hist["num_trees"] == "25"]

    filter_df_other = df[df["args"].str.contains("totally_random")]
    filter_df_other = filter_df_other[filter_df_other["num_trees"] == "200"]

    df = pd.concat([filter_df_other, filter_df_hist])

    df = df[df["args"].str.contains("DP")]
    print(df.groupby(["weight_update"]).mean()["test_auc"].astype("str") + " +- " + df.groupby(["weight_update"]).std()["test_auc"].astype("str"))

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        means = df.groupby(["dataset", "epsilon", "split_method", "weight_update"]).mean()["test_auc"].round(4).astype("str")
        sds = df.groupby(["dataset","epsilon",  "split_method", "weight_update"]).std()["test_auc"].round(4).astype("str")
        table = means + " +- " + sds
        print(table)


    # sns.boxplot(data=df, x="epsilon", y="test_auc", hue="args")
    # plt.show()


# ============================================= E3 - Split Candidate Methods =============================================

def plot_split_candidates(in_path=None, out_path="./paper_plots/", replication=False, dataset="Credit 1", ylim=0.7):

    if not replication:
        df = pd.read_csv("./paper_results/E3_split_candidate_methods_final.csv")
        df2 = pd.read_csv("./paper_results/E3_split_candidate_methods_IH.csv")

        df = df[~df["args"].str.contains("adaptive_hessian")]
        df = pd.concat([df, df2])
    else:
        df = pd.read_csv(in_path)

    df = process_df_for_plotting(df)

    df["args"] = df["dp_method"] + df["split_method"] + df["ebm"].astype("str")
    df["args"] = df["args"].str.replace("'", "")
    df["args"] = df["args"].str.replace(" ", "")

    arg_map = {"DPtotally_randomFalse" : "DP-TR Newton",
       "DPtotally_randomTrue" : "DP-TR Newton EBM",
       "DPhist_basedTrue" : "DP-Hist Newton EBM",
       "DPhist_basedFalse" : "DP-Hist Newton",}

    df = df.replace({"args":arg_map})
    df["args"] = df["args"] + " -" + df["sketch_type"] + " rounds:" + df["sketch_rounds"].astype("str")

    # Remove quantile sketching
    df = df[~df["args"].str.contains("feverless")]

    print(df["args"])
    print(set(df["args"].values))

    # Filter plot params

    # Hist no EBM, no vary rounds
    # df = df[df["args"].str.contains("DP-XGBoost") & (~df["args"].str.contains("EBM")) &
    #         (~df["args"].str.contains("TR")) & (df["args"].str.contains("inf"))]

    # Hist no EBM, just vary rounds
    # df = df[df["args"].str.contains("DP-XGBoost") & (~df["args"].str.contains("EBM")) &
    #         (~df["args"].str.contains("TR")) & (df["args"].str.contains("adaptive_hessian"))]

    # TR, no EBM, no vary rounds
    # df = df[(df["args"].str.contains("TR")) & ~(df["args"].str.contains("EBM")) & (df["args"].str.contains("inf"))]

    # TR, no EBM, vary rounds
    # df = df[(df["args"].str.contains("TR")) & ~(df["args"].str.contains("EBM")) & (df["args"].str.contains("adaptive_hessian"))]


    # ==================== VARY EPS PLOT ====================

    # TR, no EBM, rounds=5 best one
    filter1 = df[(df["args"].str.contains("TR")) & ~(df["args"].str.contains("EBM")) & (~df["args"].str.contains("adaptive_hessian"))]
    filter2 = df[df["args"].str.contains("adaptive_hessian rounds: 5")]
    filtered_df = pd.concat([filter1,filter2])
    filtered_df = filtered_df.reset_index()
    filtered_df["args"] = filtered_df["sketch_type"]

    arg_map = {" uniform" : "Uniform",
       " log" : "Log",
       " exact_quantiles" : "Quantiles",
       " adaptive_hessian" : "Iterative Hessian (s=5)"}

    filtered_df = filtered_df.replace({"args":arg_map})

    # TR with EBM splits
    # df = df[(df["args"].str.contains("TR")) & (df["args"].str.contains("EBM"))]

    # Hist with EBM splits
    # df = df[~(df["args"].str.contains("TR")) & (df["args"].str.contains("EBM"))]

    filtered_df["num_trees"] = filtered_df["num_trees"].astype("str")
    filtered_df["max_depth"] = filtered_df["max_depth"].astype("str")

    print(set(filtered_df["max_depth"].values))
    print(set(filtered_df["num_trees"].values))
    print(set(filtered_df["dataset"].values))

    filtered_df = filtered_df[filtered_df["dataset"] == dataset]

    max_depth = "4"
    filtered_df = filtered_df[filtered_df["max_depth"] == max_depth]

    trees = "100"
    # trees = "200"
    filtered_df = filtered_df[filtered_df["num_trees"] == trees]

    filtered_df = filtered_df[filtered_df["hist_bin"] == 32]

    # epsilon = ""
    # epsilon = 1
    # df = df[df["epsilon"] == epsilon]

    # cm = sns.color_palette("Blues_r", 10) + sns.color_palette("Reds", 10) + sns.color_palette("Purples",1)
    cm = None

    # sns.barplot(data=df, x="num_trees", y="test_auc", hue="args", palette=cm)
    ax = sns.barplot(data=filtered_df, x="epsilon", y="test_auc", hue="args", palette=cm, ci="sd")
    # plt.title(dataset + "- max_depth:" + max_depth + " epsilon: " + str(epsilon))
    leg = plt.legend( loc = 'lower right')

    # ax.legend(bbox_to_anchor=(1.1, 1.05))
    # plt.tight_layout()
    plt.ylim(0.5)
    plt.xlabel("Privacy Budget ($\epsilon$)")
    plt.ylabel("Test AUC")
    plt.tight_layout()
    plt.savefig(out_path+ "split_candidates_vary_eps_" + dataset+".pdf")
    plt.clf()

    ax = sns.lineplot(data=filtered_df, x="epsilon", y="test_auc", hue="args", palette=cm, ci="sd")
    leg = plt.legend( loc = 'lower right')
    plt.ylim(0.5)
    plt.xlabel("Privacy Budget ($\epsilon$)")
    plt.ylabel("Test AUC")
    plt.tight_layout()
    plt.savefig(out_path+ "split_candidates_vary_eps_" + dataset+"_lineplot.pdf")
    plt.clf()

    # ==================== VARY T PLOT ====================
    filter1 = df[(df["args"].str.contains("TR")) & ~(df["args"].str.contains("EBM")) & (~df["args"].str.contains("adaptive_hessian"))]
    filter2 = df[df["args"].str.contains("adaptive_hessian rounds: 5")]
    filtered_df = pd.concat([filter1,filter2])
    filtered_df = filtered_df.reset_index()
    filtered_df["args"] = filtered_df["sketch_type"]
    arg_map = {" uniform" : "Uniform",
       " log" : "Log",
       " exact_quantiles" : "Quantiles",
       " adaptive_hessian" : "Iterative Hessian (s=5)"}

    filtered_df = filtered_df.replace({"args":arg_map})

    # TR with EBM splits
    # df = df[(df["args"].str.contains("TR")) & (df["args"].str.contains("EBM"))]

    # Hist with EBM splits
    # df = df[~(df["args"].str.contains("TR")) & (df["args"].str.contains("EBM"))]

    filtered_df = filtered_df[filtered_df["dataset"] == dataset]

    max_depth = "4"
    filtered_df = filtered_df[filtered_df["max_depth"] == max_depth]

    # trees = "100"
    filtered_df = filtered_df[filtered_df["num_trees"].isin(["100", "200", "300"])]

    # epsilon = ""
    epsilon = 1
    filtered_df = filtered_df[filtered_df["epsilon"] == epsilon]

    filtered_df = filtered_df[filtered_df["hist_bin"] == 32]

    # cm = sns.color_palette("Blues_r", 10) + sns.color_palette("Reds", 10) + sns.color_palette("Purples",1)
    cm = None

    # sns.barplot(data=df, x="num_trees", y="test_auc", hue="args", palette=cm)
    ax = sns.barplot(data=filtered_df, x="num_trees", y="test_auc", hue="args", palette=cm, ci="sd")
    # plt.title(dataset + "- max_depth:" + max_depth + " epsilon: " + str(epsilon))
    leg = plt.legend( loc = 'lower right')

    # ax.legend(bbox_to_anchor=(1.1, 1.05))
    # plt.tight_layout()
    plt.ylim(0.5)
    plt.xlabel("Number of trees (T)")
    plt.ylabel("Test AUC")
    plt.tight_layout()
    plt.savefig(out_path+ "split_candidates_vary_T_" + dataset+".pdf")
    plt.clf()

    ax = sns.barplot(data=filtered_df, x="num_trees", y="test_auc", hue="args", palette=cm, ci="sd")
    leg = plt.legend( loc = 'lower right')
    plt.xlabel("Number of trees (T)")
    plt.ylabel("Test AUC")
    plt.ylim(0.65)
    plt.tight_layout()
    plt.savefig(out_path+ "split_candidates_vary_T_" + dataset+"_zoom.pdf")
    plt.clf()

    ax = sns.lineplot(data=filtered_df, x="num_trees", y="test_auc", hue="args", palette=cm, ci="sd")
    leg = plt.legend( loc = 'lower right')
    plt.xlabel("Number of trees (T)")
    plt.ylabel("Test AUC")
    plt.tight_layout()
    plt.savefig(out_path+ "split_candidates_vary_T_" + dataset+"_lineplot.pdf")
    plt.clf()

    #========================= VARY Q =======================

    filter1 = df[(df["args"].str.contains("TR")) & ~(df["args"].str.contains("EBM")) & (~df["args"].str.contains("adaptive_hessian"))]
    filter2 = df[df["args"].str.contains("adaptive_hessian rounds: 5")]
    filtered_df = pd.concat([filter1,filter2])
    filtered_df = filtered_df.reset_index()
    filtered_df["args"] = filtered_df["sketch_type"]
    arg_map = {" uniform" : "Uniform",
       " log" : "Log",
       " exact_quantiles" : "Quantiles",
       " adaptive_hessian" : "Iterative Hessian (s=5)"}

    filtered_df = filtered_df.replace({"args":arg_map})

    # TR with EBM splits
    # df = df[(df["args"].str.contains("TR")) & (df["args"].str.contains("EBM"))]

    # Hist with EBM splits
    # df = df[~(df["args"].str.contains("TR")) & (df["args"].str.contains("EBM"))]

    filtered_df = filtered_df[filtered_df["dataset"] == dataset]

    max_depth = "4"
    filtered_df = filtered_df[filtered_df["max_depth"] == max_depth]

    trees = "100"
    filtered_df = filtered_df[filtered_df["num_trees"] == trees]

    # epsilon = ""
    epsilon = 1
    filtered_df = filtered_df[filtered_df["epsilon"] == epsilon]

    # cm = sns.color_palette("Blues_r", 10) + sns.color_palette("Reds", 10) + sns.color_palette("Purples",1)
    cm = None

    # sns.barplot(data=df, x="num_trees", y="test_auc", hue="args", palette=cm)
    ax = sns.lineplot(data=filtered_df, x="hist_bin", y="test_auc", hue="args", palette=cm, ci="sd", marker="o")
    # plt.title(dataset + "- max_depth:" + max_depth + " epsilon: " + str(epsilon))
    leg = plt.legend( loc = 'lower right')
    # ax.legend(bbox_to_anchor=(1.1, 1.05))
    # plt.tight_layout()
    plt.ylim(ylim)
    plt.xlabel("Number of Split Candidates (Q)")
    plt.ylabel("Test AUC")
    plt.tight_layout()
    plt.savefig(out_path + "/split_candidates_vary_Q_" + dataset + ".pdf")
    #plt.show()
    plt.clf()

    # ==================== VARY s PLOT ====================
    filtered_df = df[(df["args"].str.contains("TR")) & ~(df["args"].str.contains("EBM")) & (df["args"].str.contains("adaptive_hessian"))]

    # TR with EBM splits
    # df = df[(df["args"].str.contains("TR")) & (df["args"].str.contains("EBM"))]

    # Hist with EBM splits
    # df = df[~(df["args"].str.contains("TR")) & (df["args"].str.contains("EBM"))]

    filtered_df = filtered_df[filtered_df["dataset"] == dataset]

    max_depth = "4"
    filtered_df = filtered_df[filtered_df["max_depth"] == max_depth]

    trees = "100"
    filtered_df = filtered_df[filtered_df["num_trees"] == trees]

    filtered_df = filtered_df[filtered_df["hist_bin"] == 32]

    # epsilon = ""
    # epsilon = 1
    # filtered_df = filtered_df[filtered_df["epsilon"] == epsilon]
    filtered_df["sketch_rounds"][filtered_df["sketch_rounds"]==" inf"] = trees
    filtered_df["args"] = "IH (s=" + filtered_df["sketch_rounds"] + ")"

    filtered_df = filtered_df.replace({"args":arg_map})
    cm = sns.color_palette("Reds_r", 5) + sns.color_palette("Purples",1)

    # sns.barplot(data=df, x="num_trees", y="test_auc", hue="args", palette=cm)
    ax = sns.barplot(data=filtered_df, x="epsilon", y="test_auc", hue="args", palette=cm, ci="sd")
    # plt.title(dataset + "- max_depth:" + max_depth + " epsilon: " + str(epsilon))
    leg = plt.legend( loc = 'lower right')

    plt.xlabel("Privacy Budget ($\epsilon$)")
    plt.ylabel("Test AUC")

    # ax.legend(bbox_to_anchor=(1.1, 1.05))
    # plt.tight_layout()
    plt.ylim(0.5)
    plt.tight_layout()
    plt.savefig(out_path + "split_candidates_vary_s_" + dataset+".pdf")
    #plt.show()
    plt.clf()

    cm = sns.color_palette("Reds_r", 5)
    ax = sns.lineplot(data=filtered_df, x="epsilon", y="test_auc", hue="args", marker="o", palette=cm, ci="sd")
    leg = plt.legend( loc = 'lower right')
    plt.xlabel("Privacy Budget ($\epsilon$)")
    plt.ylabel("Test AUC")
    plt.ylim(0.7)
    plt.tight_layout()
    plt.savefig(out_path + "split_candidates_vary_s_" + dataset+"_lineplot.pdf")
    plt.clf()

def table_split_candidate():
    df = pd.read_csv("./paper_results/E3_split_candidate_methods_final.csv")
    df2 = pd.read_csv("./paper_results/E3_split_candidate_methods_IH.csv")

    df = df[~df["args"].str.contains("adaptive_hessian")]
    df = pd.concat([df, df2])

    df = process_df_for_plotting(df)

    # Needed for ebm rework dataset
    df["ebm"] = False
    df["ebm"][df["feature_interaction_method"] == "cyclical"] = True

    df["args"] = df["dp_method"] + df["split_method"] + df["ebm"].astype("str")
    df["args"] = df["args"].str.replace("'", "")
    df["args"] = df["args"].str.replace(" ", "")

    arg_map = {"DPtotally_randomFalse" : "DP-XGBoost TR",
       "DPtotally_randomTrue" : "DP-XGBoost TR EBM",
       "DPhist_basedTrue" : "DP-XGBoost EBM",
       "DPhist_basedFalse" : "DP-XGBoost",}

    df = df.replace({"args":arg_map})
    df["args"] = df["args"] + " -" + df["sketch_type"] + " rounds:" + df["sketch_rounds"].astype("str") + df["ebm"].astype("str")

    print(set(df["args"].values))

    # Filter plot params

    # TR, no EBM, rounds=5 best one
    df1 = df[(df["args"].str.contains("TR")) & ~(df["args"].str.contains("True")) & (~df["args"].str.contains("adaptive_hessian"))]
    df2 = df[df["args"].str.contains("adaptive_hessian rounds: 5") & (df["args"].str.contains("False"))]
    df = pd.concat([df1,df2])
    df = df.reset_index()

    max_depth = "4"
    df = df[df["max_depth"] == max_depth]

    trees = "100" # old dataset
    # trees = "200" # new dataset
    df = df[df["num_trees"] == trees]

    # epsilon = ""
    epsilon = 1
    df = df[df["epsilon"] == epsilon]

    df = df[df["hist_bin"] == 32]

    df = df[~df["args"].str.contains("feverless")]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        # print(df.groupby(["epsilon", "args", "num_trees", "max_depth"]).mean()["test_auc"])
        print(df.groupby(["dataset", "args", "num_trees", "epsilon", "max_depth"]).mean()["test_auc"])
        print(df.groupby(["dataset", "args", "num_trees", "epsilon", "max_depth"]).std()["test_auc"])

        means = df.groupby(["dataset", "args"]).mean()["test_auc"].round(4).astype("str")

        max_vals = df.groupby(["dataset", "args"]).mean()["test_auc"].round(4).groupby(["dataset"]).max().astype("str")

        means[means.isin(max_vals)] = "\\textbf{" + means[means.isin(max_vals)] + "}"

        sds = df.groupby(["dataset", "args"]).std()["test_auc"].round(4).astype("str")
        table = means + " (" + sds + ")"

        print(table)
        datasets, args = zip(*table.index)
        # column_index = ["Iterative Hessian (s=5)", "Quantiles", "Log", "Uniform"]
        column_index = ["IH (s=5)", "Quantiles", "Log", "Uniform"]

        datasets = list(dict.fromkeys(datasets))
        print(datasets)
        print(column_index)

        table = pd.DataFrame(table.values.reshape(len(datasets), -1), columns=column_index, index=datasets)
        # table = table.style.applymap(is_bold)

        # table = pd.DataFrame(table.values.reshape(-1, len(datasets)), columns=datasets, index=column_index)
        print(table)
        print(table.to_latex(escape=False))

# ============================================= E4 - Feature Interactions =============================================

def plot_k_way(in_path="./paper_results/E4_k_way.csv", out_path="./paper_plots/", replication=False, dataset="Credit 1"):
    df = pd.read_csv(in_path)
    df = process_df_for_plotting(df)

    df["args"] = df["split_method"] + df["sketch_type"] + \
                   + df["weight_update"] + df["training_method"]

    df["args"] = df["args"].str.replace(" ", "")
    df["args"] = df["args"].str.replace("'", "")

    arg_map = {"totally_randomuniformxgboostboosting" : "DP-TR Newton",
               "totally_randomuniformgbmboosting": "DP-TR Gradient"}

    df = df.replace({"args":arg_map})
    df["feature_interaction_k"] = df["feature_interaction_k"].str.replace("None", "m")
    df["args"] = df["args"] + " " + df["feature_interaction_method"] + " (k="+ df["feature_interaction_k"] + ")"
    df["args"] = df["args"].str.replace("'", "")

    cm = sns.color_palette("Blues_r", 6) + sns.color_palette("Reds_r", 6)
    cm = sns.color_palette("Blues_r", 6)

    df = df[df["dataset"] ==  dataset]
    df = df[df["max_depth"] == "4"]
    df = df[df["epsilon"] == "1"]
    df = df[df["args"].str.contains("cyclical")]
    df = df[df["args"].str.contains("Newton")]
    df = df.sort_values("args")

    ax = sns.lineplot(data=df, x="num_trees", y="test_auc", hue="args", palette=cm)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc="lower right")

    plt.xlabel("Number of Trees (T)")
    plt.ylabel("Test AUC")
    plt.tight_layout()
    plt.savefig(out_path + "feature_interactions_vary_k_" + dataset+".pdf")
    plt.clf()

def plot_ebm_comparisons(in_path="./paper_results/E4_k_way.csv", out_path="./paper_plots/", replication=False, dataset="Credit 1"):
    df = pd.read_csv(in_path)
    df = process_df_for_plotting(df)

    # Needed for ebm rework dataset
    df["ebm"] = True
    df["ebm"][df["feature_interaction_method"] == "standard"] = False
    df["ebm"] = df["ebm"].astype("str")

    df["args"] = df["split_method"] + df["weight_update"] + df["ebm"]
    df["args"] = df["args"].str.replace(" ", "")
    arg_map = {"totally_randomgbmFalse" : "DP-TR Gradient ($k=m$)",
               "totally_randomgbmTrue" : "DP-TR Gradient EBM ($k=1$)",
               "totally_randomxgboostFalse": "DP-TR Newton ($k=m$)",
               "totally_randomxgboostTrue": "DP-TR Newton EBM ($k=1$)",
               "hist_basedgbmTrue" : "DP-Hist Gradient EBM ($k=1$)",
               "hist_basedxgboostTrue" : "DP-Hist Newton EBM ($k=1$)"}

    df = df.replace({"args":arg_map})
    df = df[df["args"].str.contains("TR")]
    df = df.sort_values("args")

    print("Options:", set(df["dataset"].values),
          set(df["epsilon"].values),
          set(df["max_depth"].values))

    # =================== Vary T ===================
    filtered_df = df[df["dataset"] == "Credit 1"]
    filtered_df = filtered_df[filtered_df["epsilon"] == "1"]
    filtered_df = filtered_df[filtered_df["max_depth"] == "4"]

    print(filtered_df)

    # cm = sns.color_palette("Purples_r", 2) +  sns.color_palette("Greens_r", 2)
    cm = None
    ax = sns.lineplot(data=filtered_df, x="num_trees", y="test_auc", hue="args", palette=cm)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc="lower right")

    ax.set_xlabel("Number of trees (T)")
    ax.set_ylabel("Test AUC")
    plt.tight_layout()
    plt.savefig(out_path+ "dp_ebm_vary_T_" + dataset+".pdf")

    plt.clf()

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(filtered_df.groupby(["dataset", "epsilon", "args", "num_trees", "max_depth"]).mean()["test_auc"])



# ============================================= E5 - Batched Updates =============================================

def table_low_eps_bb():
    # df = pd.read_csv("./paper_results/pre_paper/rf_boosting_low_eps_pre_paper.csv")
    # df1 = pd.read_csv("./paper_results/pre_paper/rf_boosting_low_eps_vec_mech_pre_paper.csv")
    # df = pd.concat([df,df1])
    # df = df.reset_index()

    df = pd.read_csv("./paper_results/E5_rf_boosting_final.csv")

    df = process_df_for_plotting(df)

    # Needed for ebm rework dataset
    df["ebm"] = True
    df["ebm"][df["feature_interaction_method"] == "standard"] = False

    df["args"] = df["dp_method"] + df["training_method"] + df["ebm"].astype("str") + df["weight_update"] + df["vec_mech"]
    df["args"] = df["args"].str.replace("'", "")
    df["args"] = df["args"].str.replace(" ", "")

    arg_map = {
       "DPbatched_boostingTrueTrue" : "DP-TR Batch Newton EBM (Vec Mech)",
       "DPbatched_boostingFalseTrue" : "DP-TR Batch Newton (Vec Mech)",
       "DPbatched_boostingTrueFalse" : "DP-TR Batch Newton EBM",
       "DPbatched_boostingFalseFalse" : "DP-TR Batch Newton",
       "DPboostingFalsexgboostTrue" : "DP-TR Newton (Vec Mech)",
       "DPboostingFalsegbmTrue" : "DP-TR Gradient",
       "DPboostingTruegbmFalse" : "DP-TR Gradient EBM",
       "DPboostingTruexgboostTrue" : "DP-TR Newton EBM (Vec Mech)",
       "DPboostingTruexgboostFalse" : "DP-TR Newton EBM",
       "DPboostingFalsexgboostFalse" : "DP-TR Newton",
       "DPrfTruerfFalse" : "DP-RF EBM",
       "DPrfTruerfTrue" : "DP-RF EBM (Vec Mech)",
       "DPrfFalserfTrue" : "DP-RF (Vec Mech)",
       "DPrfFalserfFalse" : "DP-RF"}

    df = df.replace({"args":arg_map})
    df["args"] = df["args"] + " " + df["batch_perc"].astype("str")

    df = df.replace({"args":arg_map})

    # Filter plot params
    # dataset = "Credit 1"
    # df = df[df["dataset"] == dataset]

    max_depth = "4"
    df = df[df["max_depth"] == max_depth]

    trees = "200"
    df = df[df["num_trees"] == trees]

    # epsilon = ""
    # epsilon = 0.5
    # df = df[df["epsilon"] == epsilon]
    df = df[df["epsilon"] == 0.1]

    df = df[~df["args"].str.contains("EBM")]
    df = df[~df["args"].str.contains("gbm")]
    df = df[~df["args"].str.contains("Gradient")]
    # df = df[~df["args"].str.contains("Vec Mech")]
    df = df[~df["args"].str.contains("0.34")]
    df = df[~df["args"].str.contains("0.75")]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df.groupby(["dataset", "args", "num_trees", "epsilon", "max_depth"]).mean()["test_auc"])
        print(df.groupby(["dataset", "args", "num_trees", "epsilon", "max_depth"]).std()["test_auc"])

        means = df.groupby(["dataset", "args"]).mean()["test_auc"].round(4).astype("str")
        sds = df.groupby(["dataset", "args"]).std()["test_auc"].round(4).astype("str")

        max_vals = df.groupby(["dataset", "args"]).mean()["test_auc"].round(4).groupby(["dataset"]).max().astype("str")

        means[means.isin(max_vals)] = "\\textbf{" + means[means.isin(max_vals)] + "}"
        table = means + " (" + sds + ")"

        # for val in max_vals.values:
        #     table[table.str.contains(val)] = "\\textbf{" + table[table.str.contains(val)] + "}"

        print(table)
        datasets, args = zip(*table.index)
        # percs = [0.05, 0.1, 0.25, 0.34, 0.5, 0.75, 1]
        percs = [0.05, 0.1, 0.25, 0.5, 1]

        column_index = []
        for i in range(0, len(percs)+2):
            if i <= 4:
                column_index.append("Batch (B=" + str(int(int(trees)*percs[i])) + ")")
            elif i == 5:
                column_index.append("DP-RF (B="+trees +")")
            else:
                column_index.append("Newton (B=1)")

        datasets = list(dict.fromkeys(datasets))
        print(datasets)
        print(column_index)

        table = pd.DataFrame(table.values.reshape(len(datasets), -1), columns=column_index, index=datasets)
        # table = pd.DataFrame(table.values.reshape(-1, len(datasets)), columns=datasets, index=column_index)
        print(table.transpose(), "\n")
        print(table.transpose().to_latex(escape=False))

def plot_low_eps_bb(in_path="./paper_results/E5_rf_boosting_final.csv", out_path="./paper_plots/", replication=False, dataset="Credit 1"):
    set_fontsize(12)
    df = pd.read_csv(in_path)

    df = process_df_for_plotting(df)

    # Needed for ebm rework dataset
    df["ebm"] = True
    df["ebm"][df["feature_interaction_method"] == "standard"] = False

    df["args"] = df["dp_method"] + df["training_method"] + df["ebm"].astype("str") + df["weight_update"] + df["vec_mech"]
    df["args"] = df["args"].str.replace("'", "")
    df["args"] = df["args"].str.replace(" ", "")

    arg_map = {
       "DPbatched_boostingTrueTrue" : "DP-TR Batch Newton EBM (Vec Mech)",
       "DPbatched_boostingFalseTrue" : "DP-TR Batch Newton (Vec Mech)",
       "DPbatched_boostingTrueFalse" : "DP-TR Batch Newton EBM",
       "DPbatched_boostingFalseFalse" : "DP-TR Batch Newton",
       "DPboostingFalsexgboostTrue" : "DP-TR Newton (Vec Mech)",
       "DPboostingFalsegbmTrue" : "DP-TR Gradient",
       "DPboostingTruegbmFalse" : "DP-TR Gradient EBM",
       "DPboostingTruexgboostTrue" : "DP-TR Newton EBM (Vec Mech)",
       "DPboostingTruexgboostFalse" : "DP-TR Newton EBM",
       "DPboostingFalsexgboostFalse" : "DP-TR Newton",
       "DPrfTruerfFalse" : "DP-RF EBM",
       "DPrfTruerfTrue" : "DP-RF EBM (Vec Mech)",
       "DPrfFalserfTrue" : "DP-RF (Vec Mech)",
       "DPrfFalserfFalse" : "DP-RF"}

    df = df[~df["batch_perc"].astype("str").str.contains("0.34")]
    df = df[~df["batch_perc"].astype("str").str.contains("0.75")]
    df = df.replace({"args":arg_map})

    # Filter plot params
    df = df[df["dataset"] == dataset]

    max_depth = "4"
    df = df[df["max_depth"] == max_depth]

    trees = "200"
    df = df[df["num_trees"] == trees]

    df["batch_perc"][df["args"].str.contains("DP-TR Newton")] = 1/int(trees)

    df["args"] += " (B=" + (df["batch_perc"] * int(trees)).astype("int").astype("str") + ")"

    # epsilon = ""
    # epsilon = 0.5
    # df = df[df["epsilon"] == epsilon]
    df = df[df["epsilon"].isin([0.1, 0.5])]
    df = df[~df["args"].str.contains("EBM")]
    df = df[~df["args"].str.contains("Gradient")]
    df = df[~df["args"].str.contains("gbm")]
    # df = df[~df["args"].str.contains("Vec Mech")] # Comment this out for fixed data

    df["args"] = df["args"].str.replace("\(Vec Mech\)", "")

    # cm = sns.color_palette("Blues_r", 1) + sns.color_palette("Oranges_r", 1) + sns.color_palette("Reds", 8)
    cm = sns.color_palette("Blues_r", 1) + sns.color_palette("Greens_r", 1) + sns.color_palette("Reds", 7)

    # sns.barplot(data=df, x="num_trees", y="test_auc", hue="args", palette=cm)
    ax = sns.barplot(data=df, x="epsilon", y="test_auc", hue="args", palette=cm, ci="sd")
    # ax = sns.lineplot(data=df, x="epsilon", y="test_auc", hue="args", palette=cm, ci="sd")
    # plt.title(dataset + "- max_depth:" + max_depth + " epsilon: " + str(epsilon))
    # leg = plt.legend( loc = 'lower right')
    # plt.ylim(0.72, 0.81)
    plt.ylim(0.5)
    # ax.legend(bbox_to_anchor=(1.1, 1.05))
    # plt.tight_layout()
    plt.xlabel("Privacy Budget ($\epsilon$)")
    plt.ylabel("Test AUC")

    # remove df column name from legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path + "low_eps_bb_" + dataset + ".pdf")
    plt.clf()

    ax = sns.barplot(data=df, x="epsilon", y="test_auc", hue="args", palette=cm, ci="sd")
    # ax = sns.lineplot(data=df, x="epsilon", y="test_auc", hue="args", palette=cm, ci="sd")
    # plt.title(dataset + "- max_depth:" + max_depth + " epsilon: " + str(epsilon))
    # leg = plt.legend( loc = 'lower right')
    # plt.ylim(0.72, 0.81)
    # ax.legend(bbox_to_anchor=(1.1, 1.05))
    # plt.tight_layout()
    plt.xlabel("Privacy Budget ($\epsilon$)")
    plt.ylabel("Test AUC")

    # remove df column name from legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path + "low_eps_bb_" + dataset + "_zoom.pdf")
    plt.clf()


# ============================================= E6 - Comparisons =============================================

def plot_comparisons(in_path=None, out_path="./paper_plots/", replication=False, dataset="Credit 1",
                     reduced=False, ylim1=0.5, ylim2=0.82):
    set_fontsize(11)

    if not replication:
        df = pd.read_csv("./paper_results/E6_comparisons_fix.csv")
        df2 = pd.read_csv("./paper_results/E6_comparisons_IH.csv")
        df = df[~df["args"].str.contains("adaptive_hessian")]
        df = df[~df["args"].str.contains("batched_boosting")]
        df = pd.concat([df,df2])
        non_dp_df = pd.read_csv("./paper_results/non_dp_comparisons.csv")
    else:
        df = pd.read_csv(in_path)
        non_dp_df = pd.read_csv("../paper_experiments/paper_results/non_dp_comparisons.csv")

    # ldp_df = pd.read_csv("./paper_results/ldp.csv")
    ldp_df = pd.read_csv("./paper_results/ldp_tr.csv")

    print(ldp_df["args"].unique())

    df = pd.concat([df, ldp_df])

    df = df.reset_index()

    df = process_df_for_plotting(df)
    non_dp_df = process_df_for_plotting(non_dp_df)
    non_dp_df = non_dp_df[non_dp_df["args"].str.contains("exact_quantiles")]

    df["args"] = df["dp_method"] + df["split_method"] + df["sketch_type"] + \
                   + df["weight_update"] + df["training_method"] +\
                 df["feature_interaction_method"] + df["feature_interaction_k"]

    df["args"] = df["args"].str.replace(" ", "")
    df["args"] = df["args"].str.replace("'", "")

    non_dp_df["args"] = non_dp_df["args"].str.replace("'", "")

    arg_map = {'DPhist_basedfeverlessxgboostboostingstandardd' : "FEVERLESS (sketch)",
               'DPtotally_randomuniformbatched_boostingcyclical1' : "DP-TR Batch EBM",
               'DPtotally_randomadaptive_hessianbatched_boostingstandardd' : "DP-TR Batch IH",
               'DPtotally_randomuniformgbmboostingcyclical1' : "DP-EBM",
               'DPtotally_randomuniformxgboostboostingrandom2' : "DP-TR Newton (Random k=2)",
               'DPtotally_randomuniformxgboostboostingstandardd' : "DP-TR Newton",
               'gaussian_ldphist_baseduniformxgboostboostingstandardd' : "LDP",
               'gaussian_ldptotally_randomuniformxgboostboostingstandardd' : "LDP-TR",
               'DPtotally_randomadaptive_hessianxgboostboostingstandardd' : "DP-TR Newton IH",
               'DPtotally_randomuniformrfrfstandardd' : "DP-RF",
               'DPtotally_randomuniformbatched_boostingstandardd' : "DP-TR Batch",
               'DPtotally_randomuniformxgboostboostingrandom5' : "DP-TR Newton (Random k=5)",
               'DPhist_baseduniformgbmboostingstandardd' : "DP-GBM",
               'DPtotally_randomuniformxgboostboostingcyclical2' : "DP-TR Newton (Cyclical k=2)",
               'DPtotally_randomuniformxgboostboostingcyclical1' : "DP-EBM Newton",
               'DPhist_baseduniformxgboostboostingstandardd' : "FEVERLESS (uniform)",
               'DPtotally_randomadaptive_hessianbatched_boostingcyclical1': "DP-TR Batch Newton IH EBM",
               'DPtotally_randomadaptive_hessianxgboostboostingcyclical1': "DP-TR Newton IH EBM",
               }

    df = df.replace({"args":arg_map})
    non_dp_df = non_dp_df.replace({"args": arg_map})
    non_dp_df["args"] = "XGBoost (Non-private)"

    a = 0.05

    df["batch_perc"] = round(df["batch_perc"]/ a) * a
    df["batch_perc"] = df["batch_perc"].astype("str")

    df["batch_perc"][df["batch_perc"] == "0.3"] = "0.25" # It got rounded up...
    df["batch_perc"][df["batch_perc"] == "0.25"] = "(p=0.25)"
    df["batch_perc"][~df["args"].str.contains("Batch")] = ""
    df["batch_perc"][df["batch_perc"] == "1.0"] = "(p=1)"

    # print(df["batch_perc"])

    df["args"] += " " + df["batch_perc"].astype("str")
    df["num_trees"] = df["num_trees"].astype('int')

    print(set(df["args"].values))

    filter_list = ["FEVERLESS (uniform)",
                   "DP-GBM",
                   "DP-RF",
                   "DP-EBM",
                   "DP-EBM Newton",
                   "DP-TR Newton",
                   "DP-TR Newton IH",
                   "DP-TR Batch IH (p=0.25)",
                   "DP-TR Batch IH (p=1)",
                   "DP-TR Batch Newton IH EBM (p=1)",
                   "DP-TR Batch Newton IH EBM (p=0.25)",
                   "DP-TR Newton IH EBM"]

    filter_list = ["FEVERLESS (uniform)",
                   "DP-GBM",
                   "DP-RF",
                   "DP-EBM",
                   "DP-EBM Newton",
                   "DP-TR Newton IH",
                   "DP-TR Batch Newton IH EBM (p=1)",
                   "DP-TR Batch Newton IH EBM (p=0.25)",
                   "DP-TR Newton IH EBM",
                   "LDP",
                   "LDP-TR"]

    filter_list = ["FEVERLESS (uniform)",
                   "DP-GBM",
                   "DP-RF",
                   "DP-EBM",
                   "DP-EBM Newton",
                   "DP-TR Newton IH",
                   "DP-TR Batch Newton IH EBM (p=1)",
                   "DP-TR Batch Newton IH EBM (p=0.25)",
                   "DP-TR Newton IH EBM",
                   "LDP",]

    if reduced:
        filter_list = ["FEVERLESS (uniform)",
               "DP-RF",
               "DP-EBM",
               "DP-EBM Newton",
               "DP-TR Newton IH",
               "DP-TR BatchBoost IH 0.25",
               "DP-TR BatchBoost IH"]


    styles = {
        "FEVERLESS (uniform)": "",
       "DP-GBM": "",
       "DP-RF": "",
       "DP-EBM": (1, 1),
       "DP-EBM Newton": (1, 1),
       "DP-TR Newton IH": (4, 1.5),
       "DP-TR Batch Newton IH EBM (p=1)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Batch Newton IH EBM (p=0.25)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Newton IH EBM": (3, 1.25, 1.5, 1.25),
        "XGBoost (Non-private)": "",
        "LDP": ""
    }

    styles = {
        "FEVERLESS (uniform)": (4, 1.5),
       "DP-GBM": (4, 1.5),
       "DP-RF": (4, 1.5),
       "DP-EBM": (4, 1.5),
       "DP-EBM Newton": "",
       "DP-TR Newton IH": "",
       "DP-TR Batch Newton IH EBM (p=1)": "",
       "DP-TR Batch Newton IH EBM (p=0.25)": "",
       "DP-TR Newton IH EBM": "",
        "XGBoost (Non-private)": (4, 1.5),
        "LDP": (4, 1.5),
        "LDP-TR": (4, 1.5),
    }

    markers = {
       "FEVERLESS (uniform)": ".",
       "DP-GBM": ".",
       "DP-RF": ".",
       "DP-EBM": "v",
       "DP-EBM Newton": "v",
       "DP-TR Newton IH": "",
       "DP-TR Batch Newton IH EBM (p=1)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Batch Newton IH EBM (p=0.25)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Newton IH EBM": (3, 1.25, 1.5, 1.25),
        "XGBoost (Non-private)": ""
    }

    df["args"] = df["args"].str.strip(" ")
    df = df[df["args"].isin(filter_list)]

    # print(set(df["epsilon"].values))
    # print(set(df["max_depth"].values))
    # print(set(df["dataset"]))

    # Filter df
    df = df[df["dataset"] == dataset]
    df = df[df["max_depth"] == "4"]
    df = df[df["epsilon"] == 1]

    non_dp_df = non_dp_df[non_dp_df["dataset"] == dataset]
    non_dp_df = non_dp_df[non_dp_df["max_depth"] == "4"]

    df = pd.concat([df, non_dp_df])
    df = df.sort_values("args")

    print(set(df["args"].values))
    print(len(set(df["args"].values)))
    print(df.columns)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    # ax = sns.lineplot(data=df, x="num_trees", y="f1_score", hue="args")
    cm = None

    print(set(df["args"].values))
    print(len(filter_list)+1)

    cm = None
    cm = sns.color_palette("deep", len(filter_list)+1)
    cm[-1] = "black"
    cm[5] = "#933136"
    cm[7] = "#2C760A"
    sns.lineplot(data=df, x="num_trees", y="test_auc", hue="args", ax=ax1, palette=cm, style="args", dashes=styles)
    # sns.lineplot(data=df, x="num_trees", y="test_auc", hue="args", ax=ax1, palette=cm)

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[0:], labels=labels[0:])
    # ax.legend(handles,labels, ncol=2) # you can specify any location parameter you want here

    ax1.set_ylim(ylim1)
    ax1.set_xlabel("Number of Trees (T)")
    ax1.set_ylabel("Test AUC")
    # plt.savefig(filepath + "comparisons_" + dataset + ".pdf")

    #plt.show()
    # plt.clf()

    # cm = sns.color_palette("bright", len(filter_list)+1)
    sns.lineplot(data=df, x="num_trees", y="test_auc", hue="args", palette=cm, style="args", dashes=styles, ax=ax2)
    # sns.lineplot(data=df, x="num_trees", y="test_auc", hue="args", palette=cm, ax=ax2)

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[0:], labels=labels[0:])
    # ax.legend(handles,labels, ncol=2) # you can specify any location parameter you want here

    ax2.set_ylim(ylim2)
    ax2.set_xlabel("Number of Trees (T)")
    ax2.set_ylabel("Test AUC")
    plt.tight_layout()

    handles, labels = ax2.get_legend_handles_labels()
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    # fig.legend(handles, labels, loc='lower left', ncol=2, bbox_to_anchor=(0, 1.02, 1.02), expand=True, borderaxespad=0,)
    # fig.legend(handles, labels, loc=9, ncol=2)
    from textwrap import fill
    labels = [fill(l, 20) for l in labels]

    fig.legend(handles, labels, loc="center right", ncol=1)
    fig.subplots_adjust(right=0.75)
    # fig.subplots_adjust(top=0.75)
    plt.savefig(out_path + "comparisons_zoom_" + dataset + ".pdf", bbox_inches="tight")
    plt.clf()

def table_comparisons():
    df = pd.read_csv("./paper_results/E6_comparisons_fix.csv")
    df2 = pd.read_csv("./paper_results/E6_comparisons_IH.csv")

    df = df[~df["args"].str.contains("adaptive_hessian")]
    df = df[~df["args"].str.contains("batched_boosting")]

    df = pd.concat([df,df2])
    df = df.reset_index()

    df = process_df_for_plotting(df)

    df["args"] = df["split_method"] + df["sketch_type"] + \
                   + df["weight_update"] + df["training_method"] +\
                 df["feature_interaction_method"] + df["feature_interaction_k"]

    df["args"] = df["args"].str.replace(" ", "")
    df["args"] = df["args"].str.replace("'", "")

    arg_map = {'hist_basedfeverlessxgboostboostingstandardd' : "FEVERLESS (sketch)",
               'totally_randomuniformbatched_boostingcyclical1' : "DP-TR Batch EBM",
               'totally_randomadaptive_hessianbatched_boostingstandardd' : "DP-TR Batch IH",
               'totally_randomuniformgbmboostingcyclical1' : "DP-EBM",
               'totally_randomuniformxgboostboostingrandom2' : "DP-TR Newton (Random k=2)",
               'totally_randomuniformxgboostboostingstandardd' : "DP-TR Newton",
               'totally_randomadaptive_hessianxgboostboostingstandardd' : "DP-TR Newton IH",
               'totally_randomuniformrfrfstandardd' : "DP-RF",
               'totally_randomuniformbatched_boostingstandardd' : "DP-TR Batch",
               'totally_randomuniformxgboostboostingrandom5' : "DP-TR Newton (Random k=5)",
               'hist_baseduniformgbmboostingstandardd' : "DP-GBM",
               'totally_randomuniformxgboostboostingcyclical2' : "DP-TR Newton (Cyclical k=2)",
               'totally_randomuniformxgboostboostingcyclical1' : "DP-EBM Newton",
               'hist_baseduniformxgboostboostingstandardd' : "FEVERLESS (uniform)",
               'totally_randomadaptive_hessianbatched_boostingcyclical1': "DP-TR Batch Newton IH EBM",
               'totally_randomadaptive_hessianxgboostboostingcyclical1': "DP-TR Newton IH EBM",
               }

    df = df.replace({"args":arg_map})
    a = 0.05

    df["batch_perc"] = round(df["batch_perc"]/ a) * a
    df["batch_perc"][df["batch_perc"] == 0.3] = 0.25
    df["batch_perc"][df["batch_perc"] == 1] = ""

    print(df["batch_perc"])

    df["args"] += " " + df["batch_perc"].astype("str")
    df["num_trees"] = df["num_trees"].astype('int')

    print(set(df["args"].values))

    filter_list = set(df["args"].values)

    # filter_list = ["FEVERLESS (uniform) ",
    #                "DP-GBM ",
    #                "DP-RF ",
    #                "DP-EBM ",
    #                "DP-EBM Newton ",
    #                "DP-TR Newton ",
    #                "DP-TR Newton IH ",
    #                "DP-TR BatchBoost IH 0.25",
    #                "DP-TR BatchBoost IH "]
    #
    # df = df[df["args"].isin(filter_list)]

    # Filter df
    df = df[df["max_depth"] == "4"]
    df = df[df["epsilon"] == 1]

    datasets = set(df["dataset"])

    filtered = []
    for arg in filter_list:
        for dataset in datasets:
            new_filter = df[df["args"] == arg]
            new_filter = new_filter[new_filter["dataset"] == dataset]
            avgs = new_filter.groupby(["num_trees"])["test_auc"].mean()
            max = avgs.argmax()
            t = avgs.index[max]
            new_filter = new_filter[new_filter["num_trees"] == t]
            filtered.append(new_filter)

    df = pd.concat(filtered)
    df["args"] = df["args"].str.strip(" ")
    df = df.sort_values("args")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        # print(df.groupby(["dataset", "args", "num_trees", "epsilon", "max_depth"]).mean()["test_auc"])
        # print(df.groupby(["dataset", "args", "num_trees", "epsilon", "max_depth"]).std()["test_auc"])

        means = df.groupby(["dataset", "args"]).mean()["test_auc"].round(4).astype("str")
        sds = df.groupby(["dataset", "args"]).std()["test_auc"].round(4).astype("str")

        max_vals = df.groupby(["dataset", "args"]).mean()["test_auc"].round(4).groupby(["dataset"]).max().astype("str")

        means[means.isin(max_vals)] = "\\textbf{" + means[means.isin(max_vals)] + "}"
        table = means + " (" + sds + ")" + " T=" + df.groupby(["dataset", "args"]).mean()["num_trees"].astype(int).astype("str")

        # for val in max_vals.values:
        #     table[table.str.contains(val)] = "\\textbf{" + table[table.str.contains(val)] + "}"

        print(table)
        datasets, args = zip(*table.index)

        datasets = list(dict.fromkeys(datasets))
        print(datasets)

        unique_args = []
        for arg in args:
            if arg not in unique_args:
                unique_args.append(arg)

        table = pd.DataFrame(table.values.reshape(len(datasets), -1), columns=unique_args, index=datasets)
        # table = pd.DataFrame(table.values.reshape(-1, len(datasets)), columns=datasets, index=column_index)
        print(table.transpose(), "\n")
        print(table.transpose().to_latex(escape=False))

# ============================================= Not Used =============================================

# Not used
def plot_non_dp_ebm(filepath="./paper_plots/", dataset="Credit 1"):
    df = pd.read_csv("./paper_results/E4_non_dp_ebm.csv")
    df = process_df_for_plotting(df)

    # Needed for ebm rework dataset
    df["ebm"] = True
    df["ebm"][df["feature_interaction_method"] == "standard"] = False
    df["ebm"] = df["ebm"].astype("str")

    print("Options:", set(df["dataset"].values), set(df["epsilon"].values), set(df["max_depth"].values))

    depth = '4'
    df = df[df["dataset"] == dataset]
    df = df[df["max_depth"] == depth]

    cm = sns.color_palette("Purples_r", 2) +  sns.color_palette("Greens_r", 2)
    cm = None

    df["args"] = df["split_method"] + df["weight_update"] + df["ebm"]
    df["args"] = df["args"].str.replace(" ", "")
    arg_map = {"totally_randomgbmFalse" : "Gradient",
               "totally_randomgbmTrue" : "Gradient EBM",
               "totally_randomxgboostFalse": "Newton",
               "totally_randomxgboostTrue": "Newton EBM"}

    df = df.replace({"args":arg_map})
    df = df.sort_values("args")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df.groupby(["dataset", "args", "num_trees", "max_depth"]).mean()["test_auc"])

    ax = sns.lineplot(data=df, x="num_trees", y="test_auc", hue="args", ci="sd", palette=cm)
    cm = None
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc="lower right")
    # plt.title(dataset + " depth " + depth)
    plt.xlabel("Number of trees (T)")
    plt.ylabel("Test AUC")
    #plt.show()
    plt.tight_layout()
    plt.savefig(filepath+ "non_dp_ebm_vary_T_" + dataset+".pdf")
    plt.clf()

def plot_grad_budgets():
    df = pd.read_csv("./paper_results/gradient_budget_alloc.csv")
    df2 = pd.read_csv("./paper_results/dp_split_methods_with_update.csv")

    df = pd.concat([df, df2])
    df = df.reset_index()
    df = process_df_for_plotting(df)

    df = df[df["num_trees"] == "10"]
    df = df[df["split_method"] == "hist_based"]

    df["args"] = df["dp_method"]
    df["args"] = df["args"].str.replace("'", "")
    df["args"] = df["args"].str.replace(" ", "")

    arg_map = {"DP" : "DP-XGBoost",}
    df = df.replace({"args":arg_map})

    df["args"] += " " + df["split_method"] + " " + df["weight_update"] + " " + df["gradient_budgets"]

    # Filter plot params
    # dataset = "Credit 1"
    # df = df[df["dataset"] == dataset]
    #
    # max_depth = "2"
    # df = df[df["max_depth"] == max_depth]
    #
    # trees = "200"
    # df = df[df["num_trees"] == trees]

    # epsilon = ""
    # epsilon = 0.5
    # df = df[df["epsilon"] == epsilon]

    df = df[df["args"].str.contains("xgboost")]

    # cm = sns.color_palette("Blues_r", 10) + sns.color_palette("Reds", 10) + sns.color_palette("Purples",1)

    # sns.barplot(data=df, x="num_trees", y="test_auc", hue="args", palette=cm)
    # ax = sns.barplot(data=df, x="epsilon", y="test_auc", hue="args", ci="sd")

    # ax = sns.pointplot(data=df, x="es_window", y="test_auc", hue="es_threshold")
    # plt.title(dataset + "- max_depth:" + max_depth + " epsilon: " + str(epsilon))
    # leg = plt.legend( loc = 'upper right')
    # ax.legend(bbox_to_anchor=(1.1, 1.05))
    # plt.tight_layout()
    # plt.show()

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        # print(df.groupby(["epsilon", "args", "num_trees", "max_depth"]).mean()["test_auc"])
        print(df.groupby(["dataset", "args", "num_trees", "epsilon", "max_depth"]).mean()["test_auc"])
        print(df.groupby(["dataset", "args", "num_trees", "epsilon", "max_depth"]).std()["test_auc"])

        keys = df.groupby(["dataset", "args", "num_trees", "epsilon", "max_depth"]).mean()["test_auc"].index
        means = df.groupby(["dataset", "args", "num_trees", "epsilon", "max_depth"]).mean()["test_auc"].values
        sds = df.groupby(["dataset", "args", "num_trees", "epsilon", "max_depth"]).std()["test_auc"].values

        print(df.groupby(["gradient_budgets"]).mean()["test_auc"])

        # for i in range(len(means)):
        #     print(keys[i][1], round(means[i],4), "+-", round(sds[i],4))

def ebm_top_bottom_plot():
    df = pd.read_csv("../experiment_data/ebm_top_bottom.csv")
    df = process_df_for_plotting(df)

    print(df["dataset"].unique())

    df = df[df["dataset"] == "APS"]
    df = df[df["epsilon"] == '0.01']
    df = df[df["max_depth"] == '2']

    df["args"] = df["dp_method"] + df["split_method"] + df["weight_update"] + df["ebm"] + df["gradient_budgets"]
    df["args"] = df["args"].str.replace("'", "")
    df["args"] = df["args"].str.replace(" ", "")

    arg_map = {"DPtotally_randomxgboostTruevector_mechanism" : "DP-XGBoost EBM (Vector Mech)",
               "DPtotally_randomxgboostFalsevector_mechanism" : "DP-XGBoost (Vector Mech)",
               "DPtotally_randomgbmTrue[0.9": "DP-GBM EBM (0.9,0.1)",
               "DPtotally_randomgbmFalsevector_mechanism": "DP-GBM (Vector Mech)",
               "DPtotally_randomgbmFalse[0.9": "DP-GBM (0.9,0.1)",
               "DPhist_basedxgboostTruevector_mechanism": "DP-XGBoost EBM Hist (Vector Mech)",
               "DPhist_basedgbmTruevector_mechanism":  "DP-GBM EBM Hist (Vector Mech)"}

    df = df.replace({"args":arg_map})
    # epsilons = ['0.01', '0.1', '0.5', '1']
    # for dataset in df["dataset"].unique():
    #     for eps in epsilons:
    #         for method in arg_map.values():
    #             new_df = df[df["args"] == method]
    #             new_df = new_df[new_df["dataset"] == dataset]
    #             new_df = new_df[new_df["epsilon"] == eps]
    #             grouped_df = new_df.groupby(["num_trees", "max_depth"]).mean()["test_auc"]
    #             argmax = new_df.groupby(["num_trees", "max_depth"]).mean()["test_auc"].idxmax()
    #             print(dataset, eps, method, argmax, round(new_df.groupby(["num_trees", "max_depth"]).mean()["test_auc"].max(),4))
    #
    #         print("\n")
    # print("\n")

    # dp_ebm_xgboost = df[df["args"].str.contains("DP-XGBoost EBM \(")]
    # dp_xgboost_df = df[df["args"].str.contains("DP-XGBoost \(")]
    # dp_gbm_df = df[df["args"].str.contains("DP-GBM \(")]
    # dp_hist_df = df[df["args"].str.contains("Hist")]
    #
    # sns.barplot(data=dp_xgboost_df, x="num_trees", y="test_auc", hue="args", ci="sd")
    # sns.barplot(data=dp_hist_df, x="num_trees", y="test_auc", hue="args", ci="sd", palette="Reds")
    # sns.barplot(data=df, x="num_trees", y="test_auc", hue="args", ci="sd")
    # sns.barplot(data=df, x="num_trees", y="test_auc", hue="args", ci="sd")
    # plt.show()

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df.groupby(["dataset", "epsilon", "args", "num_trees", "max_depth"]).mean()["test_auc"])
    #
    sns.barplot(data=df, x="num_trees", y="test_auc", hue="args", ci="sd")
    #plt.show()

def boosting_rf_plot():
    df = pd.read_csv("./paper_results/batch_fix_pre_paper.csv")
    df = process_df_for_plotting(df)
    print(df["batch_perc"])

    df["args"] = df["dp_method"] + df["split_method"] + df["training_method"] + df["batch_perc"].astype("str") + df["ebm"].astype("str") + df["weight_update"]
    df["args"] = df["args"].str.replace("'", "")
    df["args"] = df["args"].str.replace(" ", "")

    # Filter plot params
    dataset = "Credit 1"
    df = df[df["dataset"] == dataset]

    max_depth = "4"
    df = df[df["max_depth"] == max_depth]

    trees = "100"
    df = df[df["num_trees"] == trees]

    # epsilon = ""
    epsilon = 0.5
    df = df[df["epsilon"] == epsilon]
    df = df[df["args"].str.contains("True")]

    arg_map = {"DPboosting1.0" : "DP-TR XGBoost",
               "DPrf1.0" : "DP-TR RF",
               "DPrf_boosting0.05": "DP-TR Boosting Averages- 5%",
               "DPrf_boosting0.1": "DP-TR Boosting Averages- 10%",
               "DPrf_boosting0.25": "DP-TR Boosting Averages- 25%",
               "DPrf_boosting0.5": "DP-TR Boosting Averages- 50%",
               "DPrf_boosting0.75": "DP-TR Boosting Averages- 75%",}

    df = df.replace({"args":arg_map})

    cm = sns.color_palette("Blues_r", 1) + sns.color_palette("Oranges_r", 1) + sns.color_palette("Reds", 8)
    cm = sns.color_palette("Blues_r", 3) + sns.color_palette("Purples_r", 2) + sns.color_palette("Greens", 7) + sns.color_palette("Reds", 7)

    # sns.barplot(data=df, x="num_trees", y="test_auc", hue="args", palette=cm)
    ax = sns.barplot(data=df, x="epsilon", y="test_auc", hue="args", palette=cm, ci="sd")
    # plt.title(dataset + "- max_depth:" + max_depth + " epsilon: " + str(epsilon))
    leg = plt.legend( loc = 'upper right')

    ax.legend(bbox_to_anchor=(1.1, 1.05))
    plt.tight_layout()
    #plt.show()

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        # print(df.groupby(["epsilon", "args", "num_trees", "max_depth"]).mean()["test_auc"])
        print(df.groupby(["dataset", "args", "num_trees", "epsilon", "max_depth"]).mean()["test_auc"])
        print(df.groupby(["dataset", "args", "num_trees", "epsilon", "max_depth"]).std()["test_auc"])


# ============================================= Revision Plots =============================================

def comparison_bubble_plot(in_path=None, out_path="./paper_plots/", replication=False,
                     reduced=False, ylim1=0.5, ylim2=0.82):
    set_fontsize(11)

    if not replication:
        df = pd.read_csv("./paper_results/E6_comparisons_fix.csv")
        df2 = pd.read_csv("./paper_results/E6_comparisons_IH.csv")
        df = df[~df["args"].str.contains("adaptive_hessian")]
        df = df[~df["args"].str.contains("batched_boosting")]
        df = pd.concat([df,df2])
        non_dp_df = pd.read_csv("./paper_results/non_dp_comparisons.csv")
    else:
        df = pd.read_csv(in_path)
        non_dp_df = pd.read_csv("../paper_experiments/paper_results/non_dp_comparisons.csv")

    df = df.reset_index()

    df = process_df_for_plotting(df)
    non_dp_df = process_df_for_plotting(non_dp_df)
    non_dp_df = non_dp_df[non_dp_df["args"].str.contains("exact_quantiles")]

    df["args"] = df["split_method"] + df["sketch_type"] + \
                   + df["weight_update"] + df["training_method"] +\
                 df["feature_interaction_method"] + df["feature_interaction_k"]

    df["args"] = df["args"].str.replace(" ", "")
    df["args"] = df["args"].str.replace("'", "")

    non_dp_df["args"] = non_dp_df["args"].str.replace("'", "")

    arg_map = {'hist_basedfeverlessxgboostboostingstandardd' : "FEVERLESS (sketch)",
               'totally_randomuniformbatched_boostingcyclical1' : "DP-TR Batch EBM",
               'totally_randomadaptive_hessianbatched_boostingstandardd' : "DP-TR Batch IH",
               'totally_randomuniformgbmboostingcyclical1' : "DP-EBM",
               'totally_randomuniformxgboostboostingrandom2' : "DP-TR Newton (Random k=2)",
               'totally_randomuniformxgboostboostingstandardd' : "DP-TR Newton",
               'totally_randomadaptive_hessianxgboostboostingstandardd' : "DP-TR Newton IH",
               'totally_randomuniformrfrfstandardd' : "DP-RF",
               'totally_randomuniformbatched_boostingstandardd' : "DP-TR Batch",
               'totally_randomuniformxgboostboostingrandom5' : "DP-TR Newton (Random k=5)",
               'hist_baseduniformgbmboostingstandardd' : "DP-GBM",
               'totally_randomuniformxgboostboostingcyclical2' : "DP-TR Newton (Cyclical k=2)",
               'totally_randomuniformxgboostboostingcyclical1' : "DP-EBM Newton",
               'hist_baseduniformxgboostboostingstandardd' : "FEVERLESS (uniform)",
               'totally_randomadaptive_hessianbatched_boostingcyclical1': "DP-TR Batch Newton IH EBM",
               'totally_randomadaptive_hessianxgboostboostingcyclical1': "DP-TR Newton IH EBM",
               }

    df = df.replace({"args":arg_map})
    non_dp_df = non_dp_df.replace({"args": arg_map})
    non_dp_df["args"] = "XGBoost (Non-private)"

    a = 0.05

    df["batch_perc"] = round(df["batch_perc"]/ a) * a
    df["batch_perc"] = df["batch_perc"].astype("str")

    df["batch_perc"][df["batch_perc"] == "0.3"] = "0.25" # It got rounded up...
    df["batch_perc"][df["batch_perc"] == "0.25"] = "(p=0.25)"
    df["batch_perc"][~df["args"].str.contains("Batch")] = ""
    df["batch_perc"][df["batch_perc"] == "1.0"] = "(p=1)"

    # print(df["batch_perc"])

    df["args"] += " " + df["batch_perc"].astype("str")
    df["num_trees"] = df["num_trees"].astype('int')

    print(set(df["args"].values))

    filter_list = ["FEVERLESS (uniform)",
                   "DP-GBM",
                   "DP-RF",
                   "DP-EBM",
                   "DP-EBM Newton",
                   "DP-TR Newton",
                   "DP-TR Newton IH",
                   "DP-TR Batch IH (p=0.25)",
                   "DP-TR Batch IH (p=1)",
                   "DP-TR Batch Newton IH EBM (p=1)",
                   "DP-TR Batch Newton IH EBM (p=0.25)",
                   "DP-TR Newton IH EBM"]

    filter_list = ["FEVERLESS (uniform)",
                   "DP-GBM",
                   "DP-RF",
                   "DP-EBM",
                   "DP-EBM Newton",
                   # "DP-TR Newton",
                   "DP-TR Newton IH",
                   # "DP-TR Batch IH (p=0.25)",
                   # "DP-TR Batch IH (p=1)",
                   "DP-TR Batch Newton IH EBM (p=1)",
                   "DP-TR Batch Newton IH EBM (p=0.25)",
                   "DP-TR Newton IH EBM"]

    if reduced:
        filter_list = ["FEVERLESS (uniform)",
               "DP-RF",
               "DP-EBM",
               "DP-EBM Newton",
               "DP-TR Newton IH",
               "DP-TR BatchBoost IH 0.25",
               "DP-TR BatchBoost IH"]


    styles = {
        "FEVERLESS (uniform)": "",
       "DP-GBM": "",
       "DP-RF": "",
       "DP-EBM": (1, 1),
       "DP-EBM Newton": (1, 1),
       "DP-TR Newton IH": (4, 1.5),
       "DP-TR Batch Newton IH EBM (p=1)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Batch Newton IH EBM (p=0.25)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Newton IH EBM": (3, 1.25, 1.5, 1.25),
        "XGBoost (Non-private)": ""
    }

    styles = {
        "FEVERLESS (uniform)": (4, 1.5),
       "DP-GBM": (4, 1.5),
       "DP-RF": (4, 1.5),
       "DP-EBM": (4, 1.5),
       "DP-EBM Newton": "",
       "DP-TR Newton IH": "",
       "DP-TR Batch Newton IH EBM (p=1)": "",
       "DP-TR Batch Newton IH EBM (p=0.25)": "",
       "DP-TR Newton IH EBM": "",
        "XGBoost (Non-private)": (4, 1.5)
    }

    markers = {
       "FEVERLESS (uniform)": "o",
       "DP-GBM": "o",
       "DP-RF": "o",
       "DP-EBM": "o",
       "DP-EBM Newton": "v",
       "DP-TR Newton IH": "v",
       "DP-TR Batch Newton IH EBM (p=1)": "v",
       "DP-TR Batch Newton IH EBM (p=0.25)": "v",
       "DP-TR Newton IH EBM": "v",
        "XGBoost (Non-private)": "*"
    }


    dataset_filter = ["adult", "Credit 1", "Credit 2", "higgs-sample"]

    print(df["dataset"].unique())
    df["args"] = df["args"].str.strip(" ")
    df = df[df["args"].isin(filter_list)]
    df = df[df["dataset"].isin(dataset_filter)]
    non_dp_df = non_dp_df[non_dp_df["dataset"].isin(dataset_filter)]

    # print(set(df["epsilon"].values))
    # print(set(df["max_depth"].values))
    # print(set(df["dataset"]))

    # Filter df
    # df = df[df["dataset"] == dataset]
    df = df[df["max_depth"] == "4"]
    df = df[df["epsilon"] == 0.5]

    # non_dp_df = non_dp_df[non_dp_df["dataset"] == dataset]
    non_dp_df = non_dp_df[non_dp_df["max_depth"] == "4"]

    df = pd.concat([df, non_dp_df])
    df = df.sort_values("args")

    print(set(df["args"].values))
    print(len(set(df["args"].values)))
    print(df.columns)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    cm = None

    print(set(df["args"].values))
    print(len(filter_list)+1)

    cm = None
    cm = sns.color_palette("deep", len(filter_list)+1)
    cm[-1] = "black"
    cm[5] = "#933136"
    cm[7] = "#2C760A"

    cm_map = {df["args"].unique()[i] : col for i, col in enumerate(cm)}

    print(cm_map)

    df['dataset'] = df['dataset'].replace(['nomao'], 'Nomao')
    df['dataset'] = df['dataset'].replace(['higgs-sample'], 'Higgs')
    df['dataset'] = df['dataset'].replace(['adult'], 'Adult')

    df = df.sort_values(["args", "dataset"])

    # sns.scatterplot(data=df, x="dataset", y="test_auc", size="num_trees", hue="args", legend=True, palette=cm, sizes=(20, 2000))
    ax = sns.stripplot(data=df, x="dataset", y="test_auc", hue="args", palette=cm, jitter=0.5)
    plt.tight_layout()
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Test AUC")

    plt.savefig(out_path + "bubble_plot_full.pdf", bbox_inches="tight")
    plt.clf()

    group = df.groupby(by=["args", "dataset", "num_trees"]).mean().reset_index()

    exclude_list = ["DP-TR Batch Newton IH EBM (p=1)", "DP-TR Newton IH", 'DP-EBM Newton']
    for arg in exclude_list:
        group = group[group["args"] != arg]

    ax = None
    plt.figure(figsize=(6,5))
    for i, arg in enumerate(group["args"].unique()):
        filter_df = group[group["args"] == arg]
        # ax = sns.stripplot(data=filter_df, x="dataset", y="test_auc", color=cm_map[arg], marker=markers[arg], dodge=True, ax=ax, alpha=0.75)
        ax = sns.stripplot(data=filter_df, x="dataset", y="test_auc", color=cm_map[arg], marker=markers[arg], jitter=0.2, ax=ax, alpha=0.75)

    plt.tight_layout()

    handles = []
    for i, arg in enumerate(group["args"].unique()):
        handles.append(mlines.Line2D([], [], color=cm_map[arg], marker=markers[arg], linestyle='None',
                                  markersize=5, label=arg))

    legend = plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
    # legend = plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(1.6, 1), ncol=1)
    legend = legend.get_frame().set_alpha(None)
    # plt.legend(handles=handles, ncol=2, bbox_to_anchor=(0.1, 1.1))
    # plt.ylim(0.5,1)
    plt.ylim(0.52,0.98)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Test AUC")
    plt.savefig(out_path + "bubble_plot_tree_mean.pdf", bbox_inches="tight")
    plt.clf()

def rank_table(in_path=None, replication=False,
                     reduced=False):
    set_fontsize(11)

    if not replication:
        df = pd.read_csv("./paper_results/E6_comparisons_fix.csv")
        df2 = pd.read_csv("./paper_results/E6_comparisons_IH.csv")
        df = df[~df["args"].str.contains("adaptive_hessian")]
        df = df[~df["args"].str.contains("batched_boosting")]
        df = pd.concat([df,df2])
        non_dp_df = pd.read_csv("./paper_results/non_dp_comparisons.csv")
    else:
        df = pd.read_csv(in_path)
        non_dp_df = pd.read_csv("../paper_experiments/paper_results/non_dp_comparisons.csv")

    df = df.reset_index()

    df = process_df_for_plotting(df)
    non_dp_df = process_df_for_plotting(non_dp_df)
    non_dp_df = non_dp_df[non_dp_df["args"].str.contains("exact_quantiles")]

    df["args"] = df["split_method"] + df["sketch_type"] + \
                   + df["weight_update"] + df["training_method"] +\
                 df["feature_interaction_method"] + df["feature_interaction_k"]

    df["args"] = df["args"].str.replace(" ", "")
    df["args"] = df["args"].str.replace("'", "")

    non_dp_df["args"] = non_dp_df["args"].str.replace("'", "")

    arg_map = {'hist_basedfeverlessxgboostboostingstandardd' : "FEVERLESS (sketch)",
               'totally_randomuniformbatched_boostingcyclical1' : "DP-TR Batch EBM",
               'totally_randomadaptive_hessianbatched_boostingstandardd' : "DP-TR Batch IH",
               'totally_randomuniformgbmboostingcyclical1' : "DP-EBM",
               'totally_randomuniformxgboostboostingrandom2' : "DP-TR Newton (Random k=2)",
               'totally_randomuniformxgboostboostingstandardd' : "DP-TR Newton",
               'totally_randomadaptive_hessianxgboostboostingstandardd' : "DP-TR Newton IH",
               'totally_randomuniformrfrfstandardd' : "DP-RF",
               'totally_randomuniformbatched_boostingstandardd' : "DP-TR Batch",
               'totally_randomuniformxgboostboostingrandom5' : "DP-TR Newton (Random k=5)",
               'hist_baseduniformgbmboostingstandardd' : "DP-GBM",
               'totally_randomuniformxgboostboostingcyclical2' : "DP-TR Newton (Cyclical k=2)",
               'totally_randomuniformxgboostboostingcyclical1' : "DP-EBM Newton",
               'hist_baseduniformxgboostboostingstandardd' : "FEVERLESS (uniform)",
               'totally_randomadaptive_hessianbatched_boostingcyclical1': "DP-TR Batch Newton IH EBM",
               'totally_randomadaptive_hessianxgboostboostingcyclical1': "DP-TR Newton IH EBM",
               }

    df = df.replace({"args":arg_map})
    non_dp_df = non_dp_df.replace({"args": arg_map})
    non_dp_df["args"] = "XGBoost (Non-private)"

    a = 0.05

    df["batch_perc"] = round(df["batch_perc"]/ a) * a
    df["batch_perc"] = df["batch_perc"].astype("str")

    df["batch_perc"][df["batch_perc"] == "0.3"] = "0.25" # It got rounded up...
    df["batch_perc"][df["batch_perc"] == "0.25"] = "(p=0.25)"
    df["batch_perc"][~df["args"].str.contains("Batch")] = ""
    df["batch_perc"][df["batch_perc"] == "1.0"] = "(p=1)"

    # print(df["batch_perc"])

    df["args"] += " " + df["batch_perc"].astype("str")
    df["num_trees"] = df["num_trees"].astype('int')

    print(set(df["args"].values))

    filter_list = ["FEVERLESS (uniform)",
                   "DP-GBM",
                   "DP-RF",
                   "DP-EBM",
                   "DP-EBM Newton",
                   "DP-TR Newton",
                   "DP-TR Newton IH",
                   "DP-TR Batch IH (p=0.25)",
                   "DP-TR Batch IH (p=1)",
                   "DP-TR Batch Newton IH EBM (p=1)",
                   "DP-TR Batch Newton IH EBM (p=0.25)",
                   "DP-TR Newton IH EBM"]

    filter_list = ["FEVERLESS (uniform)",
                   "DP-GBM",
                   "DP-RF",
                   "DP-EBM",
                   "DP-EBM Newton",
                   # "DP-TR Newton",
                   "DP-TR Newton IH",
                   # "DP-TR Batch IH (p=0.25)",
                   # "DP-TR Batch IH (p=1)",
                   "DP-TR Batch Newton IH EBM (p=1)",
                   "DP-TR Batch Newton IH EBM (p=0.25)",
                   "DP-TR Newton IH EBM"]

    if reduced:
        filter_list = ["FEVERLESS (uniform)",
               "DP-RF",
               "DP-EBM",
               "DP-EBM Newton",
               "DP-TR Newton IH",
               "DP-TR BatchBoost IH 0.25",
               "DP-TR BatchBoost IH"]


    styles = {
        "FEVERLESS (uniform)": "",
       "DP-GBM": "",
       "DP-RF": "",
       "DP-EBM": (1, 1),
       "DP-EBM Newton": (1, 1),
       "DP-TR Newton IH": (4, 1.5),
       "DP-TR Batch Newton IH EBM (p=1)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Batch Newton IH EBM (p=0.25)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Newton IH EBM": (3, 1.25, 1.5, 1.25),
        "XGBoost (Non-private)": ""
    }

    styles = {
        "FEVERLESS (uniform)": (4, 1.5),
       "DP-GBM": (4, 1.5),
       "DP-RF": (4, 1.5),
       "DP-EBM": (4, 1.5),
       "DP-EBM Newton": "",
       "DP-TR Newton IH": "",
       "DP-TR Batch Newton IH EBM (p=1)": "",
       "DP-TR Batch Newton IH EBM (p=0.25)": "",
       "DP-TR Newton IH EBM": "",
        "XGBoost (Non-private)": (4, 1.5)
    }

    markers = {
       "FEVERLESS (uniform)": ".",
       "DP-GBM": ".",
       "DP-RF": ".",
       "DP-EBM": "v",
       "DP-EBM Newton": "v",
       "DP-TR Newton IH": "",
       "DP-TR Batch Newton IH EBM (p=1)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Batch Newton IH EBM (p=0.25)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Newton IH EBM": (3, 1.25, 1.5, 1.25),
        "XGBoost (Non-private)": ""
    }

    df["args"] = df["args"].str.strip(" ")
    df = df[df["args"].isin(filter_list)]

    # print(set(df["epsilon"].values))
    # print(set(df["max_depth"].values))
    # print(set(df["dataset"]))

    # Filter df
    # df = df[df["max_depth"] == "4"]

    df = df.sort_values("args")

    print(set(df["args"].values))
    print(len(set(df["args"].values)))
    print(df.columns)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    cm = None

    print(set(df["args"].values))
    print(len(filter_list)+1)

    df['dataset'] = df['dataset'].replace(['nomao'], 'Nomao')
    df['dataset'] = df['dataset'].replace(['higgs-sample'], 'Higgs')
    df['dataset'] = df['dataset'].replace(['adult'], 'Adult')

    cm = None
    cm = sns.color_palette("deep", len(filter_list)+1)
    cm[-1] = "black"
    cm[5] = "#933136"
    cm[7] = "#2C760A"

    data, max_data = [], []
    for eps in df["epsilon"].unique():
        col_data, col_max_data = [], []
        arg_rank_map = defaultdict(list)
        max_arg_rank_map = defaultdict(list)

        for dataset in df["dataset"].unique():
            print(f"Dataset {dataset}")
            temp_df = df[df["epsilon"] == eps]
            temp_df = temp_df[temp_df["dataset"] == dataset]
            group_rank = temp_df[["args", "test_auc"]].groupby("args").mean().rank(ascending=False).reset_index()
            max_rank = temp_df[["args", "test_auc"]].groupby("args").max().rank(ascending=False).reset_index()
            print(group_rank)

            for arg in group_rank["args"].unique():
                rank = group_rank[group_rank["args"] == arg]["test_auc"].values[0]
                arg_rank_map[arg].append(rank)

            for arg in max_rank["args"].unique():
                rank = max_rank[max_rank["args"] == arg]["test_auc"].values[0]
                max_arg_rank_map[arg].append(rank)

        for arg, ranks in arg_rank_map.items():
            col_data.append([arg, eps, str(round(np.mean(ranks),2))])

        for arg, ranks in max_arg_rank_map.items():
            col_max_data.append([arg, eps, str(round(np.mean(ranks),2))])

        # Bold max in column
        min_data_rank = float("inf")
        min_rank_val = float("inf")

        for i, tup in enumerate(col_data):
            print(float(tup[2]), float(min_data_rank))
            if float(tup[2]) < float(min_rank_val):
                min_data_rank = i
                min_rank_val = tup[2]

        col_data[min_data_rank][2] = "\\textbf{" + str(col_data[min_data_rank][2]) + "}"
        for i, tup in enumerate(col_max_data):
            if float(tup[2]) < float(min_rank_val):
                min_data_rank = i
                min_rank_val = tup[2]

        col_max_data[min_data_rank][2] = "\\textbf{" + str(col_max_data[min_data_rank][2]) + "}"

        data.extend(col_data)
        max_data.extend(col_max_data)

    rank_df = pd.DataFrame(data, columns=["args", "eps", "rank"])
    max_rank_df = pd.DataFrame(max_data, columns=["args", "eps", "rank"])

    mean_rank_table = rank_df.groupby(["args", "eps"]).agg(lambda x : x).unstack(level=1)
    max_rank_table = max_rank_df.groupby(["args", "eps"]).agg(lambda x : x).unstack(level=1)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(mean_rank_table)
        print(max_rank_table)

        print(mean_rank_table.to_latex(escape=False))
        print(max_rank_table.to_latex(escape=False))

def synthetic_comm(out_path="./paper_plots/"):
    set_fontsize(11)
    df = pd.read_csv("./paper_results/synthetic_comm.csv")

    df["args"] = df["args"].astype("str")

    print(df)
    print(df.columns)

    df = process_df_for_plotting(df)
    print("df processed...")

    df["args"] = df["dp_method"] + df["split_method"] + df["sketch_type"] + \
                   + df["weight_update"] + df["training_method"] +\
                 df["feature_interaction_method"] + df["feature_interaction_k"]

    df["args"] = df["args"].str.replace(" ", "")
    df["args"] = df["args"].str.replace("'", "")

    arg_map = {'DPhist_basedfeverlessxgboostboostingstandardd' : "FEVERLESS (sketch)",
               'DPtotally_randomuniformbatched_boostingcyclical1' : "DP-TR Batch EBM",
               'DPtotally_randomadaptive_hessianbatched_boostingstandardd' : "DP-TR Batch IH",
               'DPtotally_randomuniformgbmboostingcyclical1' : "DP-EBM",
               'DPtotally_randomuniformxgboostboostingrandom2' : "DP-TR Newton (Random k=2)",
               'DPtotally_randomuniformxgboostboostingstandardd' : "DP-TR Newton",
               'gaussian_ldphist_baseduniformxgboostboostingstandardd' : "LDP",
               'gaussian_ldptotally_randomuniformxgboostboostingstandardd' : "LDP-TR",
               'DPtotally_randomadaptive_hessianxgboostboostingstandardd' : "DP-TR Newton IH",
               'DPtotally_randomuniformrfrfstandardd' : "DP-RF",
               'DPtotally_randomuniformbatched_boostingstandardd' : "DP-TR Batch",
               'DPtotally_randomuniformxgboostboostingrandom5' : "DP-TR Newton (Random k=5)",
               'DPhist_baseduniformgbmboostingstandardd' : "DP-GBM",
               'DPtotally_randomuniformxgboostboostingcyclical2' : "DP-TR Newton (Cyclical k=2)",
               'DPtotally_randomuniformxgboostboostingcyclical1' : "DP-EBM Newton",
               'DPhist_baseduniformxgboostboostingstandardd' : "FEVERLESS (uniform)",
               'DPtotally_randomadaptive_hessianbatched_boostingcyclical1': "DP-TR Batch Newton IH EBM",
               'DPtotally_randomadaptive_hessianxgboostboostingcyclical1': "DP-TR Newton IH EBM",
               }

    df = df.replace({"args":arg_map})
    a = 0.05

    df["batch_perc"] = round(df["batch_perc"]/ a) * a
    df["batch_perc"] = df["batch_perc"].astype("str")

    df["batch_perc"][df["batch_perc"] == "0.3"] = "0.25" # It got rounded up...
    df["batch_perc"][df["batch_perc"] == "0.25"] = "(p=0.25)"
    df["batch_perc"][~df["args"].str.contains("Batch")] = ""
    df["batch_perc"][df["batch_perc"] == "1.0"] = "(p=1)"

    df["args"] += " " + df["batch_perc"].astype("str")
    df["num_trees"] = df["num_trees"].astype('int')

    print(f"Unique args {df['args'].unique}")

    filter_list = ["FEVERLESS (uniform)",
                   "DP-GBM",
                   "DP-RF",
                   "DP-EBM",
                   "DP-EBM Newton",
                   "DP-TR Newton",
                   "DP-TR Newton IH",
                   "DP-TR Batch IH (p=0.25)",
                   "DP-TR Batch IH (p=1)",
                   "DP-TR Batch Newton IH EBM (p=1)",
                   "DP-TR Batch Newton IH EBM (p=0.25)",
                   "DP-TR Newton IH EBM"]

    filter_list = ["FEVERLESS (uniform)",
                   "DP-GBM",
                   "DP-RF",
                   "DP-EBM",
                   "DP-EBM Newton",
                   "DP-TR Newton",
                   "DP-TR Newton IH",
                   "DP-TR Newton IH EBM"]

    # filter_list = [
    #                "DP-TR Batch IH (p=0.25)",
    #                "DP-TR Batch IH (p=1)",
    #                "DP-TR Batch Newton IH EBM (p=1)",
    #                "DP-TR Batch Newton IH EBM (p=0.25)",]
    #
    # filter_list = ["FEVERLESS (uniform)",
    #                "DP-GBM",
    #                "DP-RF",
    #                "DP-EBM",
    #                "DP-EBM Newton",
    #                "DP-TR Newton IH",
    #                "DP-TR Batch Newton IH EBM (p=1)",
    #                "DP-TR Batch Newton IH EBM (p=0.25)",
    #                "DP-TR Newton IH EBM",
    #                "LDP"]

    styles = {
        "FEVERLESS (uniform)": "",
       "DP-GBM": "",
       "DP-RF": "",
       "DP-EBM": (1, 1),
       "DP-EBM Newton": (1, 1),
       "DP-TR Newton IH": (4, 1.5),
       "DP-TR Batch Newton IH EBM (p=1)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Batch Newton IH EBM (p=0.25)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Newton IH EBM": (3, 1.25, 1.5, 1.25),
        "XGBoost (Non-private)": "",
        "LDP": ""
    }

    styles = {
        "FEVERLESS (uniform)": (4, 1.5),
       "DP-GBM": (4, 1.5),
       "DP-RF": (4, 1.5),
       "DP-EBM": (4, 1.5),
       "DP-EBM Newton": "",
       "DP-TR Newton IH": "",
       "DP-TR Batch Newton IH EBM (p=1)": "",
       "DP-TR Batch Newton IH EBM (p=0.25)": "",
       "DP-TR Newton IH EBM": "",
        "XGBoost (Non-private)": (4, 1.5),
        "LDP": (4, 1.5)
    }

    markers = {
       "FEVERLESS (uniform)": ".",
       "DP-GBM": ".",
       "DP-RF": ".",
       "DP-EBM": "v",
       "DP-EBM Newton": "v",
       "DP-TR Newton IH": "",
       "DP-TR Batch Newton IH EBM (p=1)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Batch Newton IH EBM (p=0.25)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Newton IH EBM": (3, 1.25, 1.5, 1.25),
        "XGBoost (Non-private)": ""
    }

    df["args"] = df["args"].str.strip(" ")
    df = df[df["args"].isin(filter_list)]

    # Filter df
    # df = df[df["max_depth"] == "4"]
    df = df.sort_values("args")

    print(set(df["args"].values))
    print(len(set(df["args"].values)))
    print(df.columns)
    print(df["max_depth"].unique())

    cm = None
    cm = sns.color_palette("deep", len(filter_list)+1)
    # cm[-1] = "black"
    # cm[5] = "#933136"
    # cm[7] = "#2C760A"

    df["total_client_rounds_sent_payload"] = df["total_client_rounds_sent_payload"]/(1024**2)
    # ax = sns.lineplot(data=df, x="num_trees", y="total_client_rounds_sent_payload", hue="args", ci=99)

    # Min/max bands
    ax = sns.lineplot(data=df, x="num_trees", y="total_client_rounds_sent_payload", hue="args", ci=None)
    for i, arg in enumerate(df["args"].unique()):
        filter_df = df[df["args"] == arg].sort_values("num_trees")

        max_df = df[df["args"] == arg].groupby("num_trees").max()["total_client_rounds_sent_payload"]
        min_df = df[df["args"] == arg].groupby("num_trees").min()["total_client_rounds_sent_payload"]

        ax.fill_between(filter_df["num_trees"].unique(), min_df.values, max_df.values, color=cm[i], alpha=0.2)

    plt.ylabel("Total communication cost (Mb)")
    plt.xlabel("Number of trees (T)")
    ax.set_yscale("log")

    # leg = plt.legend(loc=(0.5,0.5), title="Method")
    leg = plt.legend(loc=(0.05,1.05), title="Method", ncol=2)
    ax.add_artist(leg)

    plt.savefig(out_path + f"comm.pdf", bbox_inches="tight")
    plt.clf()

    ax = sns.lineplot(data=df, x="num_trees", y="total_client_rounds_sent_payload", hue="args")
    plt.ylabel("Total communication cost (Mb)")
    # ax.set_yscale("log")
    plt.ylim(0, 0.5)
    plt.savefig(out_path + f"comm_ZOOM.pdf", bbox_inches="tight")
    plt.clf()

    print(df.groupby(["num_trees", "args"]).max()["total_client_rounds_sent_payload"])

def vary_clients(out_path="./paper_plots/"):
    set_fontsize(11)
    df1 = pd.read_csv("./paper_results/vary_clients3.csv")
    df2 = pd.read_csv("./paper_results/vary_clients4.csv")

    df = pd.concat([df1,df2])
    df = df.reset_index()

    df["args"] = df["args"].astype("str")

    print(df)
    print(df.columns)

    df = process_df_for_plotting(df)
    print("df processed...")

    df["args"] = df["dp_method"] + df["split_method"] + df["sketch_type"] + \
                   + df["weight_update"] + df["training_method"] +\
                 df["feature_interaction_method"] + df["feature_interaction_k"]

    df["args"] = df["args"].str.replace(" ", "")
    df["args"] = df["args"].str.replace("'", "")

    print(df["n"].unique())

    arg_map = {'DPhist_basedfeverlessxgboostboostingstandardd' : "FEVERLESS (sketch)",
               'DPtotally_randomuniformbatched_boostingcyclical1' : "DP-TR Batch EBM",
               'DPtotally_randomadaptive_hessianbatched_boostingstandardd' : "DP-TR Batch IH",
               'DPtotally_randomuniformgbmboostingcyclical1' : "DP-EBM",
               'DPtotally_randomuniformxgboostboostingrandom2' : "DP-TR Newton (Random k=2)",
               'DPtotally_randomuniformxgboostboostingstandardd' : "DP-TR Newton",
               'gaussian_ldphist_baseduniformxgboostboostingstandardd' : "LDP",
               'gaussian_ldptotally_randomuniformxgboostboostingstandardd' : "LDP-TR",
               'DPtotally_randomadaptive_hessianxgboostboostingstandardd' : "DP-TR Newton IH",
               'DPtotally_randomuniformrfrfstandardd' : "DP-RF",
               'DPtotally_randomuniformbatched_boostingstandardd' : "DP-TR Batch",
               'DPtotally_randomuniformxgboostboostingrandom5' : "DP-TR Newton (Random k=5)",
               'DPhist_baseduniformgbmboostingstandardd' : "DP-GBM",
               'DPtotally_randomuniformxgboostboostingcyclical2' : "DP-TR Newton (Cyclical k=2)",
               'DPtotally_randomuniformxgboostboostingcyclical1' : "DP-EBM Newton",
               'DPhist_baseduniformxgboostboostingstandardd' : "FEVERLESS (uniform)",
               'DPtotally_randomadaptive_hessianbatched_boostingcyclical1': "DP-TR Batch Newton IH EBM",
               'DPtotally_randomadaptive_hessianxgboostboostingcyclical1': "DP-TR Newton IH EBM",
               "hist_basedexact_quantilesxgboostboostingstandardd": "XGBoost"
               }

    df = df.replace({"args":arg_map})
    a = 0.05

    df["batch_perc"] = round(df["batch_perc"]/ a) * a
    df["batch_perc"] = df["batch_perc"].astype("str")

    df["batch_perc"][df["batch_perc"] == "0.3"] = "0.25" # It got rounded up...
    df["batch_perc"][df["batch_perc"] == "0.25"] = "(p=0.25)"
    df["batch_perc"][~df["args"].str.contains("Batch")] = ""
    df["batch_perc"][df["batch_perc"] == "1.0"] = "(p=1)"

    df["args"] += " " + df["batch_perc"].astype("str")
    df["num_trees"] = df["num_trees"].astype('int')

    print(f"Unique args {df['args'].unique}")

    filter_list = ["FEVERLESS (uniform)",
                   "DP-GBM",
                   "DP-RF",
                   "DP-EBM",
                   "DP-EBM Newton",
                   "DP-TR Newton",
                   "DP-TR Newton IH",
                   "DP-TR Batch IH (p=0.25)",
                   "DP-TR Batch IH (p=1)",
                   "DP-TR Batch Newton IH EBM (p=1)",
                   "DP-TR Batch Newton IH EBM (p=0.25)",
                   "DP-TR Newton IH EBM"]


    filter_list = [ "DP-TR Newton",
                   "DP-TR Newton IH", "XGBoost"]

    # filter_list = [
    #                "DP-TR Batch IH (p=0.25)",
    #                "DP-TR Batch IH (p=1)",
    #                "DP-TR Batch Newton IH EBM (p=1)",
    #                "DP-TR Batch Newton IH EBM (p=0.25)",]
    #
    # filter_list = ["FEVERLESS (uniform)",
    #                "DP-GBM",
    #                "DP-RF",
    #                "DP-EBM",
    #                "DP-EBM Newton",
    #                "DP-TR Newton IH",
    #                "DP-TR Batch Newton IH EBM (p=1)",
    #                "DP-TR Batch Newton IH EBM (p=0.25)",
    #                "DP-TR Newton IH EBM",
    #                "LDP"]

    styles = {
        "FEVERLESS (uniform)": "",
       "DP-GBM": "",
       "DP-RF": "",
       "DP-EBM": (1, 1),
       "DP-EBM Newton": (1, 1),
       "DP-TR Newton IH": (4, 1.5),
       "DP-TR Batch Newton IH EBM (p=1)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Batch Newton IH EBM (p=0.25)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Newton IH EBM": (3, 1.25, 1.5, 1.25),
        "XGBoost (Non-private)": "",
        "LDP": ""
    }

    styles = {
        "FEVERLESS (uniform)": (4, 1.5),
       "DP-GBM": (4, 1.5),
       "DP-RF": (4, 1.5),
       "DP-EBM": (4, 1.5),
       "DP-EBM Newton": "",
       "DP-TR Newton IH": "",
       "DP-TR Batch Newton IH EBM (p=1)": "",
       "DP-TR Batch Newton IH EBM (p=0.25)": "",
       "DP-TR Newton IH EBM": "",
        "XGBoost (Non-private)": (4, 1.5),
        "LDP": (4, 1.5)
    }

    markers = {
       "FEVERLESS (uniform)": ".",
       "DP-GBM": ".",
       "DP-RF": ".",
       "DP-EBM": "v",
       "DP-EBM Newton": "v",
       "DP-TR Newton IH": "",
       "DP-TR Batch Newton IH EBM (p=1)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Batch Newton IH EBM (p=0.25)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Newton IH EBM": (3, 1.25, 1.5, 1.25),
        "XGBoost (Non-private)": ""
    }

    df["args"] = df["args"].str.strip(" ")
    df = df[df["args"].isin(filter_list)]

    # Filter df
    df = df[df["num_trees"] == 100]
    df = df.sort_values(["args", "hist_bin"])

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     test_df = df[df["args"] == "DP-TR Newton"]
    #     test_df = test_df[test_df["hist_bin"] == 64]
    #     test_df = test_df[test_df["n"] == 7000]
    #     print(test_df)

    df["args"] = df["args"] + " Q=" + df["hist_bin"].astype("str")

    print(set(df["args"].values))
    print(len(set(df["args"].values)))
    print(df.columns)
    print(df["max_depth"].unique())

    cm = None

    reds = sns.color_palette("Reds", 9)
    blues = sns.color_palette("Blues", 9)
    greens = sns.color_palette("Greens", 9)
    blacks = sns.color_palette("Greys", 9)

    cm = reds + blues + greens
    # cm[-1] = "black"
    # cm[5] = "#933136"
    # cm[7] = "#2C760A"

    ax = sns.lineplot(data=df, x="n", y="test_auc", hue="args", palette=cm)
    plt.ylabel("Test AUC")

    # axins = ax.inset_axes([0.5, 0.1, 0.45, 0.4])
    axins = ax.inset_axes([0.3, 0.1, 0.45, 0.4])
    sns.lineplot(data=df, x="n", y="test_auc", hue="args", palette=cm, ax=axins)
    axins.set_xlim(60000, 70000)
    axins.set_ylim(0.88)
    axins.set_ylabel(None)
    axins.set_xlabel(None)
    axins.set_title('Zoomed')
    axins.grid(False)
    axins.get_legend().remove()

    marker_titles = ["2", "4", "8", "16", "32", "64", "128", "256", "512"]

    h = [plt.plot([],[], color=blacks[i])[0] for i in range(0,9)]
    # leg = plt.legend(handles=h, labels=marker_titles, title="Q", loc=(1.03,0))
    leg = plt.legend(handles=h, labels=marker_titles, title="Q", loc=(0.83,0.03))
    leg.get_frame().set_alpha(None)
    ax.add_artist(leg)

    marker_titles = filter_list
    colors = [reds[-1], blues[-1], greens[-1]]
    h = [plt.plot([],[], color=colors[i])[0] for i in range(0,3)]
    # plt.legend(handles=h, labels=marker_titles, loc=(1.03,0.7), title="Method")
    plt.legend(handles=h, labels=marker_titles, loc=(-0.05,1.05), title="Method", ncol=3)
    plt.xlabel("Number of Clients ($n$)")

    path = out_path + f"vary_clients.pdf"
    print(df["n"].unique())

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        group_df = df[df["hist_bin"] == 64].groupby(["args", "n"]).mean()["test_auc"]
        print(group_df)

    print(f"Saving to {path}")
    plt.savefig(path, bbox_inches="tight")
    plt.clf()

def computation_benchmark(out_path="./paper_plots/"):
    set_fontsize(11)
    df = pd.read_csv("./paper_results/computation_benchmark.csv")
    df["args"] = df["args"].astype("str")

    print(df)
    print(df.columns)

    df = process_df_for_plotting(df)
    print("df processed...")

    df["args"] = df["dp_method"] + df["split_method"] + df["sketch_type"] + \
                   + df["weight_update"] + df["training_method"] +\
                 df["feature_interaction_method"] + df["feature_interaction_k"]

    df["args"] = df["args"].str.replace(" ", "")
    df["args"] = df["args"].str.replace("'", "")

    print(df["n"].unique())

    arg_map = {'DPhist_basedfeverlessxgboostboostingstandardd' : "FEVERLESS (sketch)",
               'DPtotally_randomuniformbatched_boostingcyclical1' : "DP-TR Batch EBM",
               'DPtotally_randomadaptive_hessianbatched_boostingstandardd' : "DP-TR Batch IH",
               'DPtotally_randomuniformgbmboostingcyclical1' : "DP-EBM",
               'DPtotally_randomuniformxgboostboostingrandom2' : "DP-TR Newton (Random k=2)",
               'DPtotally_randomuniformxgboostboostingstandardd' : "DP-TR Newton",
               'gaussian_ldphist_baseduniformxgboostboostingstandardd' : "LDP",
               'gaussian_ldptotally_randomuniformxgboostboostingstandardd' : "LDP-TR",
               'DPtotally_randomadaptive_hessianxgboostboostingstandardd' : "DP-TR Newton IH",
               'DPtotally_randomuniformrfrfstandardd' : "DP-RF",
               'DPtotally_randomuniformbatched_boostingstandardd' : "DP-TR Batch",
               'DPtotally_randomuniformxgboostboostingrandom5' : "DP-TR Newton (Random k=5)",
               'DPhist_baseduniformgbmboostingstandardd' : "DP-GBM",
               'DPtotally_randomuniformxgboostboostingcyclical2' : "DP-TR Newton (Cyclical k=2)",
               'DPtotally_randomuniformxgboostboostingcyclical1' : "DP-EBM Newton",
               'DPhist_baseduniformxgboostboostingstandardd' : "FEVERLESS (uniform)",
               'DPtotally_randomadaptive_hessianbatched_boostingcyclical1': "DP-TR Batch Newton IH EBM",
               'DPtotally_randomadaptive_hessianxgboostboostingcyclical1': "DP-TR Newton IH EBM",
               "hist_basedexact_quantilesxgboostboostingstandardd": "XGBoost"
               }

    df = df.replace({"args":arg_map})
    a = 0.05

    df["batch_perc"] = round(df["batch_perc"]/ a) * a
    df["batch_perc"] = df["batch_perc"].astype("str")

    df["batch_perc"][df["batch_perc"] == "0.3"] = "0.25" # It got rounded up...
    df["batch_perc"][df["batch_perc"] == "0.25"] = "(p=0.25)"
    df["batch_perc"][~df["args"].str.contains("Batch")] = ""
    df["batch_perc"][df["batch_perc"] == "1.0"] = "(p=1)"

    df["args"] += " " + df["batch_perc"].astype("str")
    df["num_trees"] = df["num_trees"].astype('int')

    print(f"Unique args {df['args'].unique}")

    filter_list = ["FEVERLESS (uniform)",
                   "DP-GBM",
                   "DP-RF",
                   "DP-EBM",
                   "DP-EBM Newton",
                   "DP-TR Newton",
                   "DP-TR Newton IH",
                   "DP-TR Batch IH (p=0.25)",
                   "DP-TR Batch IH (p=1)",
                   "DP-TR Batch Newton IH EBM (p=1)",
                   "DP-TR Batch Newton IH EBM (p=0.25)",
                   "DP-TR Newton IH EBM"]


    # filter_list = [ "DP-TR Newton",
    #                "DP-TR Newton IH", "XGBoost"]

    # filter_list = [
    #                "DP-TR Batch IH (p=0.25)",
    #                "DP-TR Batch IH (p=1)",
    #                "DP-TR Batch Newton IH EBM (p=1)",
    #                "DP-TR Batch Newton IH EBM (p=0.25)",]
    #
    # filter_list = ["FEVERLESS (uniform)",
    #                "DP-GBM",
    #                "DP-RF",
    #                "DP-EBM",
    #                "DP-EBM Newton",
    #                "DP-TR Newton IH",
    #                "DP-TR Batch Newton IH EBM (p=1)",
    #                "DP-TR Batch Newton IH EBM (p=0.25)",
    #                "DP-TR Newton IH EBM",
    #                "LDP"]

    styles = {
        "FEVERLESS (uniform)": "",
       "DP-GBM": "",
       "DP-RF": "",
       "DP-EBM": (1, 1),
       "DP-EBM Newton": (1, 1),
       "DP-TR Newton IH": (4, 1.5),
       "DP-TR Batch Newton IH EBM (p=1)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Batch Newton IH EBM (p=0.25)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Newton IH EBM": (3, 1.25, 1.5, 1.25),
        "XGBoost (Non-private)": "",
        "LDP": ""
    }

    styles = {
        "FEVERLESS (uniform)": (4, 1.5),
       "DP-GBM": (4, 1.5),
       "DP-RF": (4, 1.5),
       "DP-EBM": (4, 1.5),
       "DP-EBM Newton": "",
       "DP-TR Newton IH": "",
       "DP-TR Batch Newton IH EBM (p=1)": "",
       "DP-TR Batch Newton IH EBM (p=0.25)": "",
       "DP-TR Newton IH EBM": "",
        "XGBoost (Non-private)": (4, 1.5),
        "LDP": (4, 1.5)
    }

    markers = {
       "FEVERLESS (uniform)": ".",
       "DP-GBM": ".",
       "DP-RF": ".",
       "DP-EBM": "v",
       "DP-EBM Newton": "v",
       "DP-TR Newton IH": "",
       "DP-TR Batch Newton IH EBM (p=1)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Batch Newton IH EBM (p=0.25)": (3, 1.25, 1.3, 1.25, 1.3, 1.3, 1.25, 1.3),
       "DP-TR Newton IH EBM": (3, 1.25, 1.5, 1.25),
        "XGBoost (Non-private)": ""
    }

    df["args"] = df["args"].str.strip(" ")
    df = df[df["args"].isin(filter_list)]

    df = df[df["num_trees"].isin([75, 100, 125])]

    # Filter df
    df = df.sort_values(["args", "hist_bin"])

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     test_df = df[df["args"] == "DP-TR Newton"]
    #     test_df = test_df[test_df["hist_bin"] == 64]
    #     test_df = test_df[test_df["n"] == 7000]
    #     print(test_df)

    # df["args"] = df["args"] + " Q=" + df["hist_bin"].astype("str")

    print(set(df["args"].values))
    print(len(set(df["args"].values)))
    print(df["max_depth"].unique())
    cm = None

    print(df.columns)

    filter_col = [col for col in df if col.startswith('t_client')]

    cm = sns.color_palette("deep", len(filter_list))
    cm[5] = "#933136"
    cm[7] = "#2C760A"
    # new_df_list = []
    # for arg in df["args"].values:
    #     filter_df = df[df["args"] == arg]
    #     for col in filter_col:
    #         for val in filter_df[col].values:
    #             new_df_list.append([arg, col, val])
    #
    # new_df = pd.DataFrame(new_df_list, columns=["args", "checkpoint", "time"])
    #
    # print(new_df)
    # print(new_df.columns)
    # sns.barplot(data=new_df, y="checkpoint", x="time", hue="args")
    # plt.xlabel("Time (seconds)")
    # plt.tight_layout()
    # path = out_path + "/client_checkpoint.pdf"
    # print(f"Saving to {path}")
    # plt.savefig(path, bbox_inches="tight")
    # plt.clf()


    df["total_client_time"] = df["t_client_histogram_building"] + df["t_client_computing_gradients"] + df["t_client_initialise_private_histogram"] +df["t_client_forming_grad_histogram"] + df["t_client_retrieving_grads_for_node"]

    sns.barplot(data=df, y="args", x="total_client_time", palette=cm)
    plt.xlabel("Total Time (seconds)")
    plt.ylabel("Methods")
    plt.tight_layout()
    path = out_path + "/total_client_computation.pdf"
    print(f"Saving to {path}")
    plt.savefig(path, bbox_inches="tight")
    plt.clf()

    df["total_server_time"] =  df["t_server_initial_split_candidates"] + df["t_server_privacy_accountant_initialisation"] + df["t_server_init_model_weights"] + df["t_server_split_candidates"] + df["t_server_pre_tree_ops"] + df["t_server_post_tree ops"] + df["t_server_initialise_priv_hist"] + df["t_server_adding_noise_to_hist"] + df["t_server_sampling_features"] + df["t_server_calculating_internal_split"]  + df["t_server_split_constraints"] + df["t_server_leaf_weight"]

    sns.barplot(data=df, y="args", x="total_server_time", palette=cm)
    plt.xlabel("Total Time (seconds)")
    plt.ylabel("Methods")
    plt.tight_layout()
    path = out_path + "/total_server_computation.pdf"
    print(f"Saving to {path}")
    plt.savefig(path, bbox_inches="tight")
    plt.clf()

# ============================================= Appendix =============================================

# datasets = ["Credit 1", "Credit 2", "adult", "Bank", "nomao"]
datasets = ["Credit 2", "adult", "Bank", "nomao"]


# Appendix - Split methods
# Used to plot figures 7-10
def appendix_E1():
    for dataset in datasets:
        # clear_dir(base_path+dataset+"/E1")
        plot_split_methods_with_update(base_path, dataset=dataset, show_dp=False, legends=[False, False, True])

# Appendix - Split methods + weight updates table
# Used to plot tables 7-10
def appendix_E1_table():
    epsilons = 0.1, 0.25, 0.75, 1
    for eps in epsilons:
        print("EPS", eps)
        table_split_methods_with_update(eps)

# Used to plot Figure 12
def appendix_E2(dataset="Credit 1", depth="4", epsilon=1,):
    datasets = ["Credit 2", "adult", "nomao", "Bank", "higgs-sample"]
    for dataset in datasets:
        # clear_dir(base_path+dataset+"/E2")
        ylim=0.7
        if dataset == "higgs-sample":
            ylim = 0.5
        if dataset == "nomao":
            ylim = 0.87
        if dataset == "adult":
            ylim = 0.81
        plot_split_candidates(base_path, dataset=dataset, ylim=ylim)

# Not used
def appendix_E3():
    for dataset in datasets:
        # clear_dir(base_path+dataset+"/E3")
        plot_k_way(dataset=dataset)
        plot_non_dp_ebm(dataset=dataset)
        plot_ebm_comparisons(dataset=dataset)

# Used to plot Figure 13
def appendix_E4():
    for dataset in datasets:
        # clear_dir(base_path+dataset+"/E4")
        plot_low_eps_bb(dataset=dataset)

# Appendix - Comparisons
# Used to plot figures 14-18
def appendix_E5():
    # datasets = ["Credit 2", "adult"]
    datasets = ["Credit 2", "adult", "nomao", "Bank", "higgs-sample"]
    for dataset in datasets:
        ylim2 = 0.82
        if dataset == "higgs-sample":
            ylim2 = 0.65
        elif dataset == "Credit 2":
            ylim2 = 0.7
        elif dataset == "adult":
            ylim2 = 0.85
        # clear_dir(base_path+dataset+"/E5")
        plot_comparisons(dataset=dataset, ylim2=ylim2)


# exp_plot() # test
# ============================================= Plotting Funcs =============================================

base_path = "./paper_plots/"

set_fontsize()

# ---------- REVISIONS ----------

# comparison_bubble_plot()
# rank_table()
# synthetic_comm()
# vary_clients()
# computation_benchmark()

comparison_bubble_plot()

# ---------- MAIN PAPER TABLES ----------

# table_split_methods_with_update() # Table 2
# table_split_candidate() # Table 3
# table_low_eps_bb() # Table 4

# table_comparisons() # Not used

# ---------- MAIN PAPER PLOTS ----------

# plot_split_methods_with_update(y_lims=[None, 0.67, None], show_dp=False) # Figure 1
# plot_split_candidates() # Figure 2
# plot_k_way(filepath=base_path) # Figure 3
# plot_ebm_comparisons() # Figure 4
# plot_low_eps_bb() # Figure 5
# plot_comparisons(reduced=False) # Figure 6

# plot_non_dp_ebm() # Not used

# ---------- APPENDIX ----------

# appendix_E1()
# appendix_E1_table()
# appendix_E2()
# appendix_E3()
# appendix_E4()
# appendix_E5()

# ================== Not used.... ==================
# plot_grad_budgets()
# boosting_rf_plot()
# ebm_top_bottom_plot()