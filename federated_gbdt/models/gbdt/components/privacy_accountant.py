import numpy as np
import math

from collections import defaultdict
from federated_gbdt.core.moments_accountant.compute_noise_from_budget_lib import compute_noise

from autodp.mechanism_zoo import GaussianMechanism, SubsampleGaussianMechanism
from autodp.transformer_zoo import Composition, ComposeGaussian, AmplificationBySampling
from autodp.calibrator_zoo import eps_delta_calibrator,generalized_eps_delta_calibrator, ana_gaussian_calibrator
from autodp.privacy_calibrator import RDP_mech

class PrivacyAccountant():
    def __init__(self, accounting_method, epsilon, delta, quantile_epsilon, dp_method,  # DP params
                 num_trees, max_depth, split_method, training_method, weight_update_method, split_method_per_level=None, task_type="classification", # Tree params
                 tree_budgets=None, gradient_budgets=None, level_budgets=None, # Budgets
                 feature_interaction_method="", feature_interaction_k=None, # Feature interaction
                 sample_method=None, subsample=1,
                 sigma=None, grad_clip_const=None, gradient_clipping=None, sketch_type=None, sketch_rounds=None, verbose=True): # Budget allocations

        self.epsilon = epsilon
        self.delta = delta
        self.quantile_epsilon = quantile_epsilon
        self.sigma = sigma

        self.grad_clip_const = grad_clip_const if grad_clip_const else 20
        self.gradient_clipping = gradient_clipping
        self.max_gradient = 1 if task_type == "classification" else self.grad_clip_const
        self.min_gradient = -1 if task_type == "classification" else -self.grad_clip_const
        self.max_hess = 1/4 if task_type == "classification" else 1
        self.min_hess = 0 if task_type == "classification" else 1

        if training_method == "rf":
            self.max_gradient, self.min_gradient, self.max_hess, self.min_hess = 1,1,1,1

        if weight_update_method == "gbm":
            self.max_hess, self.min_hess = 1,0

        self.grad_sensitivity = max(self.max_gradient, abs(self.min_gradient))
        self.hess_sensitivity = max(self.max_hess, abs(self.min_hess))

        self.num_trees = num_trees
        self.max_depth = max_depth
        self.split_method = split_method
        self.split_method_per_level = split_method_per_level

        self.accounting_method = accounting_method
        self.subsample = subsample
        self.sample_method = sample_method

        self.sketch_type = sketch_type
        self.sketch_rounds = sketch_rounds

        self.feature_interaction_method = feature_interaction_method
        self.feature_interaction_k = feature_interaction_k

        self.num_features = 0
        self.num_rows = 0

        self.tree_budgets = tree_budgets if tree_budgets is not None else [1 / self.num_trees] * self.num_trees # Budget proportions for the trees
        self.level_budgets = level_budgets  # Budget proportions for levels in the tree

        if self.level_budgets is None:
            level_budgets = []
            num_dp = sum([1 if method !="totally_random" else 0 for method in self.split_method_per_level])
            if self.split_method_per_level[-1] == "totally_random":
                num_dp += 1
            for i, method in enumerate(self.split_method_per_level):
                if method != "totally_random" or (method == "totally_random" and i == self.max_depth-1):
                    level_budgets.append(1/num_dp)
                else:
                    level_budgets.append(0)
            self.level_budgets = level_budgets

        if self.feature_interaction_k==1 and self.split_method != "hybrid_random" and self.split_method_per_level[0] == "hist_based":
            self.level_budgets = [0]*self.max_depth
            self.level_budgets[0] = 1

        self.gradient_budgets = gradient_budgets if gradient_budgets is not None else [0.5, 0.5] # Budget proportions for gradients vs hessians
        self.feature_budgets = None
        self.adapt_budget = False
        self.gradient_method = "scalar"

        if self.gradient_budgets == "vector_mechanism":
            self.gradient_budgets = [1,1]
            self.gradient_method = "vector_mechanism"

        self.queries = 0 if "ldp" in dp_method else max_depth # Queries of gradients per-tree, for LDP it is 1 and for DDP it is the height of the tree
        self.budget_info = np.zeros(shape=(self.num_trees, self.max_depth, 2, 3)) # Calculates the (eps, sigma) for each tree, level, gradient statistic that needs to be perturbed

        self.current_spent_budget = 0 # Track the current budget being spent at a node, is reset regularly by the training algo
        self.current_num_queries = 0
        self.current_tree = 0
        self.current_node = 0
        self.feature_candidate_size = {}
        self.dp_method = dp_method
        self.verbose = verbose

        if self.verbose:
            print("[Accountant] Level Budgets:", self.level_budgets)

    @staticmethod
    # Used to precompute sigmas for experiments
    def compute_exp_sigma(num_trees, eps, delta, subsample=1, method="totally_random", gradient_budgets=None):
        total_queries = 0
        if method == "totally_random":
            total_queries = num_trees*2
            if gradient_budgets == "vector_mechanism":
                total_queries /= 2

        params = {"prob": subsample, "coeff": total_queries, "sigma": None}
        general_calibrate = generalized_eps_delta_calibrator()
        # poisson_sampler = AmplificationBySampling(PoissonSampling=True)
        mech4 = general_calibrate(SubsampleGaussianMechanism, eps, delta, [0,1000],params=params, para_name='sigma', name='Subsampled_Gaussian')
        return mech4.params["sigma"]

    def update_feature_candidate_size(self, split_candidates):
        for feature_i, candidates in enumerate(split_candidates):
            self.feature_candidate_size[feature_i] = len(candidates)

    def update_tree(self):
        self.current_tree += 1

    def assign_budget(self, epsilon, delta, num_rows=None, num_features=None, quantile_epsilon=None):
        if quantile_epsilon is not None:
            self.quantile_epsilon = quantile_epsilon

        self.num_features = num_features
        self.num_rows = num_rows
        self.ledger = np.zeros(self.num_rows) # Ledger of participants and the budget they have spent participating in the protocol

        if self.feature_interaction_k is None:
            self.feature_interaction_k = self.num_features

        self.epsilon = epsilon-self.quantile_epsilon
        self.delta = delta

        self.feature_budgets = [1/self.num_features]*self.num_features

        num_dp = sum([1 if method !="totally_random" else 0 for method in self.split_method_per_level])
        num_queries = num_dp*self.num_trees*2*self.feature_interaction_k

        if num_queries == 0:
            num_queries = 2*self.num_trees
        elif self.split_method_per_level[self.max_depth-1] == "totally_random":
            num_queries += 2*self.num_trees

        if self.split_method == "node_based" or self.split_method == "partially_random":
            num_queries += self.num_trees

        if self.gradient_method == "vector_mechanism":
            num_queries /= 2

        # Accounting for extra budget needed to compute gradient histograms to generate new split proposals...
        if self.sketch_type == "adaptive_hessian" and self.split_method == "totally_random":
            num_queries += self.sketch_rounds*self.feature_interaction_k

        if self.sigma:
            self.sigma_arr = np.zeros(shape=(self.num_trees, self.max_depth, 2))
            for t in range(0, self.num_trees):
                self.sigma_arr[t][-1] = [self.sigma, self.sigma]
            self.delta = 1e-6
        else:
            # Compute sigma values
            self.sigma_arr, _ = self._compute_gaussian_composition(self.accounting_method, self.epsilon, self.delta, 1, num_queries, all_values=True)

        if self.verbose:
            self._autodp_check(self.sigma_arr) # auto-dp check

        self.budget_info[:,:,:,2] = self.sigma_arr

        # Assign budget - used to track budget spent during execution
        for i in range(self.num_trees):
            for j in range(self.max_depth):
                for k in range(0,2):
                    if self.budget_info[i,j,k,2] != 0:
                        self.budget_info[i, j, k, 0] = self.epsilon*self.tree_budgets[i]*self.level_budgets[j]*self.gradient_budgets[k] # Scaled epsilon
                        self.budget_info[i, j, k, 1] = self.delta*self.tree_budgets[i]*self.level_budgets[j]*self.gradient_budgets[k] # Scaled delta

        # NOTE: These budget values are not scaled by the number of features (!), this happens later in composition...
        if self.verbose:
            # print("[Accountant] Epsilon Values\n", self.budget_info[0, :, :, 0])
            print("[Accountant] Sigma Values\n", self.budget_info[0, :, :, 2])

    def commit_budget_to_ledger(self, participant_ids):
        for i in participant_ids:
            self.ledger[i] += self.current_num_queries
        self.current_spent_budget = 0
        self.current_num_queries = 0

    def _gaussian_variance(self, epsilon, delta, sensitivity):
        return (sensitivity / epsilon) * np.sqrt(2 * np.log(1.25 / delta))

    # Can be used to verify RDP scaling of budget works
    def _autodp_check(self, sigma_arr, es=None):
        if self.adapt_budget:
            new_sigma_arr = []
            for i in range(0, self.num_trees):
                for j in range(self.max_depth):
                    for k in range(0,2):
                        for l in range(0, self.num_features):
                            if self.feature_budgets[l] > 0:
                                new_sigma_arr.append(sigma_arr[i,j,k]*math.sqrt(1/self.feature_budgets[l]))
            sigma_arr = np.array(new_sigma_arr)

        mechs = []
        compose_freq = []

        sigma_method_dict = defaultdict(int)

        num_trees = self.num_trees if not es else es

        for i in range(0, num_trees):
            for j in range(self.max_depth):
                for k in range(0,2):
                    sigma = sigma_arr[i,j,k]
                    sigma_method_dict[sigma, self.split_method_per_level[j]] += 1

        poisson_sampler = AmplificationBySampling(PoissonSampling=True)

        for key, freq in sigma_method_dict.items():
            sigma, method = key
            if sigma != 0:
                if self.sample_method == "poisson":
                    mechs.append(poisson_sampler(GaussianMechanism(sigma, name='GM'), self.subsample, improved_bound_flag=True))
                else:
                    mechs.append(GaussianMechanism(sigma, name='GM'))

                if self.adapt_budget:
                    compose_freq.append(freq*1)
                else:
                    if method == "totally_random":
                        freq = freq
                    elif method == "node_based" or self.split_method == "partially_random":
                        # freq = freq*self.num_features+1
                        freq = freq*self.num_features+self.num_trees
                    else:
                        freq = freq*self.feature_interaction_k

                    if self.gradient_method == "vector_mechanism":
                        freq = freq/2

                    if self.split_method == "totally_random" and self.sketch_type == "adaptive_hessian":
                        freq += self.sketch_rounds*self.feature_interaction_k

                    compose_freq.append(freq)

        rdp_compose = Composition()
        rdp_composed_mech = rdp_compose(mechs, compose_freq)

        # GDP
        # compose = ComposeGaussian()
        # composed_mech = compose(mechs, compose_freq)

        # Query for eps given delta
        delta = self.delta
        eps = rdp_composed_mech.get_approxDP(delta)
        if self.verbose:
            print("\n[Accountant] Frequency of composed noise mechanism", compose_freq)
            # print("[Accountant] Auto-DP Eps (Gaussian Composition):", composed_mech.get_approxDP(delta))
            print("[Accountant] Auto-DP Eps (Generic RDP)", eps, "\n")

        return eps

    def _compute_gaussian_composition(self, method, eps, delta, q, total_queries, feature_index=None, verbose=False, all_values=False):
        opt_alpha = None
        sigma_arr = np.zeros(shape=(self.num_trees, self.max_depth, 2))
        if self.sample_method == "disjoint":
            total_queries = total_queries * self.subsample
            print(total_queries, self.subsample)

        # Compute single sigma value based on number of total queries and the budget
        if "rdp" in method:
            if self.sample_method == "poisson":
                params = {"prob": self.subsample, "coeff": total_queries, "sigma": None}
                general_calibrate = generalized_eps_delta_calibrator()
                # poisson_sampler = AmplificationBySampling(PoissonSampling=True)
                mech4 = general_calibrate(SubsampleGaussianMechanism, eps, delta, [0,500],params=params, para_name='sigma', name='Subsampled_Gaussian')
                sigma = mech4.params["sigma"]
            else:
                sigma, opt_alpha = compute_noise(1, 1, eps, total_queries, delta, 1e-5, verbose)
        elif method == "basic":
            sigma = total_queries * math.sqrt(2 * math.log(1.25*total_queries / delta)) / eps  # Basic composition - scalar mechanism
        elif method == "advanced":
            eps_prime = eps / (2 * math.sqrt(2 * total_queries * math.log(2 * total_queries / delta)))
            sigma = math.sqrt(2 * math.log(1.25 / delta)) / eps_prime  # Advanced composition
        else:
            a = (-2 * (math.log(delta) - eps) + math.sqrt((2 * (math.log(delta) - eps)) ** 2 + 4 * eps * (math.log(delta) + eps))) / ( 2 * eps)  # Optimal alpha value for RDP can be solved exactly in the Gaussian case by applying the weak conversion bound
            C = math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
            sigma = math.sqrt(total_queries * a * (a - 1) / (2 * (math.log(delta) + (a - 1) * eps)))  # RDP using the stronger conversion bound (a,r)-RDP to (eps,delta)-DP
            opt_alpha = a

        # If all_values=True then fill out a matrix with sigma values for each tree, level and gradient
        if all_values and "scaled" not in method: # Uniform budget across the model
            sigma_arr.fill(sigma)
            sigma = sigma_arr
        else: # Else we are scaling different components of the tree with different budgets and thus need to scale sigmas
            for i in range(0,self.num_trees):
                for j in range(0,self.max_depth):
                    for k in range(0,2):
                        if (self.split_method_per_level[j] == "totally_random" and j != self.max_depth-1) or (self.feature_interaction_k==1 and self.split_method_per_level[j] == "hist_based" and j > 0):
                            sigma_arr[i,j,k] = 0
                        elif "rdp" in method: # Tighter scaling of the budget for hetereogenous mechanisms via RDP composition
                            sigma_arr[i,j,k] = (sigma/math.sqrt(self.num_trees*np.count_nonzero(self.level_budgets)*2)) * (1/math.sqrt(self.tree_budgets[i]*self.level_budgets[j]*self.gradient_budgets[k]))

                            if self.gradient_method == "vector_mechanism":
                                sigma_arr[i,j,k] = (sigma/math.sqrt(self.num_trees*np.count_nonzero(self.level_budgets))) * (1/math.sqrt(self.tree_budgets[i]*self.level_budgets[j]*self.gradient_budgets[k]))

                            if self.adapt_budget:
                                sigma_arr[i,j,k] = (sigma/math.sqrt(self.num_trees*np.count_nonzero(self.level_budgets)*2*self.num_features)) * (1/math.sqrt(self.tree_budgets[i]*self.level_budgets[j]*self.gradient_budgets[k]))

                            if self.feature_interaction_k==1 and self.split_method_per_level[j] == "hist_based" :
                                sigma_arr[i,j,k] = sigma*math.sqrt(1/(self.max_depth))

                        else: # Basic composition budget scaling...
                            sigma_arr[i,j,k] = self._gaussian_variance(self.epsilon*self.tree_budgets[i]*self.level_budgets[j]*self.gradient_budgets[k]/self.num_features, self.delta*self.tree_budgets[i]*self.level_budgets[j]*self.gradient_budgets[k]/self.num_features, 1)
            sigma = sigma_arr

        return sigma, opt_alpha

    def gaussian_var(self, depth, gradient_type="gradient", feature=None, adaptive_hessian=False):
        # Calculate sensitivity of the query
        if gradient_type == "gradient":
            grad_type = 0
            sensitivity = self.grad_sensitivity
        else:
            grad_type = 1
            sensitivity = self.hess_sensitivity

        if self.gradient_method == "vector_mechanism":
            sensitivity = math.sqrt(self.grad_sensitivity**2 + self.hess_sensitivity**2)

        if adaptive_hessian:
            sensitivity = self.hess_sensitivity

        # Retrieve sigma and scale appropriately
        depth = max(0,depth)
        sigma = self.budget_info[self.current_tree, depth, grad_type, 2] # Retrieve sigma value for this node
        if self.adapt_budget and feature is not None:
            sigma = sigma*math.sqrt(1/self.feature_budgets[feature]) # Number of features has already been scaled for in _compute_gaussian_composition(),
                                                                        # only thing left to do is scale by the feature budget which is determind on function call

        # Calculate privacy budget spent from this query (not used)
        if self.split_method_per_level[depth] == "totally_random" or (self.feature_interaction_k == 1 and self.split_method_per_level[depth] == "hist_based"):
            budget_spent = self.budget_info[self.current_tree, depth, grad_type, 0]
        elif self.split_method_per_level[depth] == "partially_random":
            budget_spent = self.budget_info[self.current_tree, depth, grad_type, 0]/self.num_features
        elif feature is not None:
            budget_spent = self.budget_info[self.current_tree, depth, grad_type, 0]/(self.num_features*(self.feature_candidate_size[feature]+1))
        else:
            budget_spent = self.budget_info[self.current_tree, depth, grad_type, 0]/self.num_features

        self.current_spent_budget += budget_spent
        self.current_num_queries += 1

        return sensitivity*sigma

    def _add_dp_noise(self, grad_sum, hess_sum, depth, feature=None, histogram_row=False, noise_size=None, num_obs=None, adaptive_hessian=False):
        """
        Called at every node in the tree, returns perturbed (if using DP) sums of gradients/hessians

        :param grad_sum: List of gradients (first-derivative of the loss)
        :param hess_sum: List of hessians (second-derivative of the loss)
        :param depth: Current level in the tree
        :return: Perturbed sum of gradients and hessians
        """
        perturbed_gradients = grad_sum
        perturbed_hessians = hess_sum

        if histogram_row and self.dp_method == "gaussian_cdp":
            perturbed_gradients = grad_sum + np.random.normal(0, self.gaussian_var(depth, gradient_type="gradient", feature=None), size=noise_size)
            perturbed_hessians = hess_sum + np.random.normal(0, self.gaussian_var(depth, gradient_type="hessian", feature=None, adaptive_hessian=adaptive_hessian), size=noise_size)
        elif self.dp_method == "gaussian_cdp":
            perturbed_gradients += np.random.normal(0, self.gaussian_var(depth, gradient_type="gradient", feature=feature), size=1)[0]
            perturbed_hessians += np.random.normal(0, self.gaussian_var(depth, gradient_type="hessian", feature=feature), size=1)[0]

        if self.gradient_clipping:
            perturbed_gradients = self.min_gradient * num_obs if perturbed_gradients < self.min_gradient * num_obs else perturbed_gradients
            perturbed_gradients = self.max_gradient * num_obs if perturbed_gradients > self.max_gradient * num_obs else perturbed_gradients
            perturbed_hessians = self.min_hess * num_obs if perturbed_hessians < self.min_hess * num_obs else perturbed_hessians
            perturbed_hessians = self.max_hess * num_obs if perturbed_hessians > self.max_hess * num_obs else perturbed_hessians

        return perturbed_gradients, perturbed_hessians
