import numpy as np
import time

from collections import defaultdict

class TrainMonitor():
    def __init__(self, num_clients, num_classes=2):
        self.gradient_info = [] # List of tuples containing (gradient, hess) info
        self.leaf_gradient_tracker = [[], []]
        self.root_gradient_tracker = [[],[]]

        self.gradient_total = [0,0]
        self.current_tree_weights = []
        self.previous_tree_weights = []
        self.y_weights = []
        self.batched_weights = []

        self.last_feature = -1
        self.node_count = -1

        self.leaf_count_tracker = []
        self.leaf_count = 0
        self.internal_node_count = defaultdict(int)
        self.internal_node_count_tracker = []
        self.bin_tracker = defaultdict(int)
        self.tree_bin_tracker = []

        self.current_tree_depth = 0

        self.num_classes = num_classes

        self.client_rounds_sent = [0]
        self.client_rounds_received = [0]
        self.client_payload_sent = [0]
        self.client_payload_received = [0]


        self.num_clients = num_clients

        self.client_timer = 0
        self.server_timer = 0

        self.client_total_time = [0]
        self.server_total_time = [0]

        self.client_time_dict = {"histogram building": 0, "computing gradients": 0, 'initialise private histogram': 0, "forming gradient + hess histogram": 0,
                                 "retrieving grads/hess for node": 0,}

        self.server_time_dict = {"initial split candidates": 0, "privacy_accountant initialisation": 0, "initialise model weights": 0, "split_candidates": 0,
                                 "pre-tree ops": 0, "post-tree ops": 0, "initialise private histogram": 0, "adding noise to gradient + hess histogram": 0,
                                 "sampling features for node": 0, "calculating internal split": 0, "updating split constraints": 0, "leaf weight": 0}

    def start_timing_event(self, device_type, tag=""):
        if device_type == "client":
            self.client_timer = time.time()
        else:
            self.server_timer = time.time()

    def end_timing_event(self, device_type, tag=""):
        if device_type == "client":
            time_elapsed = time.time() - self.client_timer
            self.client_total_time[-1] += time_elapsed
            self.client_time_dict[tag] += time_elapsed
            self.client_timer = 0
        else:
            time_elapsed = time.time() - self.server_timer
            self.server_total_time[-1] += time_elapsed
            self.server_time_dict[tag] += time_elapsed
            self.server_timer = 0

    def update_num_clients(self, num_clients):
        self.num_clients = num_clients

        self.client_rounds_received = [0]
        self.client_rounds_sent = [0]
        self.client_payload_sent = [0]
        self.client_payload_received = [0]

    def update_received(self, client_ids, payload_size):
        self.client_payload_received[-1] += payload_size
        self.client_rounds_received[-1] += 1

    def update_sent(self, client_ids, payload_size, increment_round=True):
        if len(client_ids) > 0:
            self.client_payload_sent[-1] += payload_size
            if increment_round:
                self.client_rounds_sent[-1] += 1

    def output_summary(self):
        print(f"\nNumber of clients {self.num_clients}")
        print(f"Max client rounds sent {np.max(self.client_rounds_sent)}")
        print(f"Avg client rounds sent {np.mean(self.client_rounds_sent)}")
        print(f"Total client sent {np.sum(self.client_rounds_sent)}")

        print(f"Max client rounds received {np.max(self.client_rounds_received)}")
        print(f"Avg client rounds received {np.mean(self.client_rounds_received)}")

        print(f"Max client sent {np.max(self.client_payload_sent) / 1024}Kb")
        print(f"Average client sent {np.mean(self.client_payload_sent) / 1024}Kb")
        print(f"Total client sent {np.sum(self.client_payload_sent) / 1024}Kb")

        print(f"Total leaf count {self.leaf_count_tracker}")
        # print(f"Total internal nodes {self.internal_node_count_tracker}")
        print("\n")

        for i, t in enumerate(self.client_total_time):
            print(f"Tree {i} client total time {self.client_total_time[i]}")
            print(f"Tree {i} server total time {self.server_total_time[i]}")

        print(f"Client time dict {self.client_time_dict}")
        print(f"Server time dict {self.server_time_dict}")

    def _update_comm_stats(self, split_method, training_method):
        # print(f"Stats before updating rounds={self.client_rounds_sent[-1]}, payload={self.client_payload_sent[-1]}")

        # Internal nodes
        if split_method != "totally_random":
            total = 0
            for level in self.bin_tracker:
                num_bins = self.bin_tracker[level]
                total += 8*2*num_bins
            self.client_payload_sent[-1] += total

            for level in self.internal_node_count:
                if self.internal_node_count[level] > 0:
                    self.client_rounds_sent[-1] += 1

        # Leaf nodes
        if training_method != "batched_boosting":
            self.update_sent(range(0, self.num_clients), payload_size=8*2*self.leaf_count, increment_round=True)

        # print(f"Stats after updating rounds={self.client_rounds_sent[-1]}, payload={self.client_payload_sent[-1]}")

    def reset(self):
        # For comm tracking
        self.leaf_count_tracker.append(self.leaf_count)
        self.leaf_count = 0
        self.internal_node_count_tracker.append(self.internal_node_count)
        self.internal_node_count = defaultdict(int)
        self.tree_bin_tracker.append(self.bin_tracker)
        self.bin_tracker = defaultdict(int)

        self.client_rounds_sent.append(0)
        self.client_payload_sent.append(0)
        self.client_rounds_received.append(0)
        self.client_payload_received.append(0)

        self.client_timer, self.server_timer = 0,0
        self.client_total_time.append(0)
        self.server_total_time.append(0)

        self.gradient_total = [0,0]
        self.current_tree_depth = 0
        self.previous_tree_weights = self.current_tree_weights
        self.current_tree_weights = np.zeros(len(self.current_tree_weights)) if self.num_classes == 2 else np.zeros((len(self.current_tree_weights), self.num_classes))

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes