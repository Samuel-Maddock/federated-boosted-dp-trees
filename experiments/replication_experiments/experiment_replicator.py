from experiments.paper_experiments.paper_experiments import *
from experiments.paper_experiments.paper_plotter import *
import os.path

base_path = "./replication_data/"

class ExperimentReplicator():
    def __init__(self):
        pass

    def replicate(self, figure_num, dataset="Credit 1", overwrite=False):
        if figure_num == 1:
            filename = "replication_fig1"
            if os.path.isfile(base_path + filename + ".csv") and not overwrite:
                print("Replicated data already exists...")
            else:
                dp_split_methods_with_update_methods(filename=filename, save_data=True, replication=True, iters=3, datasets=[dataset], seeds=[1])
            print("Plotting data...")
            plot_split_methods_with_update(in_path=base_path+filename+".csv", out_path=base_path, replication=True)

        elif figure_num == 2:
            filename = "replication_fig2"
            if os.path.isfile(base_path + filename + ".csv") and not overwrite:
                print("Replicated data already exists...")
            else:
                dp_split_candidate_methods(filename=filename, save_data=True, replication=True, iters=3, datasets=[dataset], seeds=[1])
            print("Plotting data...")
            plot_split_candidates(in_path=base_path+filename+".csv", out_path=base_path, replication=True)

        elif figure_num == 3:
            filename = "replication_fig3"
            if os.path.isfile(base_path + filename + ".csv") and not overwrite:
                print("Replicated data already exists...")
            else:
                feature_interaction_experiments(filename=filename, save_data=True, replication=True, iters=6, datasets=[dataset], seeds=[1])
            print("Plotting data...")
            plot_k_way(in_path=base_path+filename+".csv", out_path=base_path, replication=True)

        elif figure_num == 4:
            filename = "replication_fig4"
            if os.path.isfile(base_path + filename + ".csv") and not overwrite:
                print("Replicated data already exists...")
            else:
                dp_ebm_experiment(filename=filename, save_data=True, replication=True, iters=10, datasets=[dataset], seeds=[1])
            print("Plotting data...")
            plot_ebm_comparisons(in_path=base_path+filename+".csv", out_path=base_path, replication=True)

        elif figure_num == 5:
            filename = "replication_fig5"
            if os.path.isfile(base_path + filename + ".csv") and not overwrite:
                print("Replicated data already exists...")
            else:
                batched_boosting(filename=filename, save_data=True, replication=True, iters=3, datasets=[dataset], seeds=[1])
            print("Plotting data...")
            plot_low_eps_bb(in_path=base_path+filename+".csv", out_path=base_path, replication=True)

        elif figure_num == 6:
            filename = "replication_fig6"
            if os.path.isfile(base_path + filename + ".csv") and not overwrite:
                print("Replicated data already exists...")
            else:
                comparisons_experiment(filename=filename, save_data=True, replication=True, iters=3, datasets=[dataset], seeds=[1])
            print("Plotting data...")
            plot_comparisons(in_path=base_path+filename+".csv", out_path=base_path, replication=True)

if __name__ == "__main__":
    replicator = ExperimentReplicator()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('fig_num', type=int, default=1, choices=range(6),nargs='+', help='Figure number to replicate')
    # parser.add_argument('overwrite', type=bool, default=False, help='Whether to overwrite the existing data')
    # args = parser.parse_args()
    # replicator.replicate(args.fig_num, overwrite=args.overwrite)

    replicator.replicate(1, overwrite=False, dataset="Credit 1")