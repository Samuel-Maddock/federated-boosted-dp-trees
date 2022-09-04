from federated_gbdt.core.moments_accountant.compute_noise_from_budget_lib import compute_noise
import math

# Sam Comment:
    # Method uses the RDP moments accountant which works as follows
    # 1) For a fixed eps,delta compute the (alpha, tau)-RDP guarantee of the Gaussian mechanism
    # 2) Perform a binary search over alpha values to find a sigma value that guarantees an epsilon < target_epsilon
    # 3) Perform a bisection on the (alpha,tau)-RDP => (eps,delta)-DP conversion bound with the (loose) sigma found
    #       to find the exact noise needed for the target_epsilon given

class RDPAccountant():
    def __init__(self, eps, delta, q, clip, total_queries, method="rdp", verbose=False):
        self.eps = eps
        self.delta = delta
        self.q = q
        self.clip = clip
        self.total_leaf_nodes = total_queries
        self.method = method
        self.sigma, self.opt_alpha = self.compute_sigma(method=method, eps=eps, delta=delta, q=q, total_queries=total_queries, verbose=verbose)

    @staticmethod
    def compute_sigma(method, eps, delta, q, total_queries, verbose):
        opt_alpha = None
        if method == "rdp":
            sigma, opt_alpha = compute_noise(1, q, eps, total_queries, delta, 1e-5, verbose)
        elif method == "basic":
            sigma = total_queries * math.sqrt(2 * math.log(total_queries*1.25 / delta)) / eps  # Basic composition - scalar mechanism
        elif method == "advanced":
            eps_prime = eps / (2 * math.sqrt(2 * total_queries * math.log(2 * total_queries / delta)))
            sigma = math.sqrt(2 * math.log(1.25 / delta)) / eps_prime  # Advanced composition
        elif "rdp_weak":
            a = (-2 * (math.log(delta) - eps) + math.sqrt((2 * (math.log(delta) - eps)) ** 2 + 4 * eps * (math.log(delta) + eps))) / ( 2 * eps)  # Optimal alpha value for RDP can be solved exactly in the Gaussian case by applying the weak conversion bound
            C = math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
            sigma = math.sqrt(total_queries * a * (a - 1) / (2 * (math.log(delta) + (a - 1) * eps)))  # RDP using the stronger conversion bound (a,r)-RDP to (eps,delta)-DP
            opt_alpha = a

        return sigma, opt_alpha

def budget_examples():
    eps = 1
    delta = 1e-5
    total_queries = 6 * 10 * 10  # Suppose 10 features over 10 trees with a maximum depth of 6
    # total_queries = 1

    sigma_basic = total_queries * math.sqrt(2 * math.log(1.25 / delta)) / eps

    eps_prime = eps / (2 * math.sqrt(2 * total_queries * math.log(2 * total_queries / delta)))

    sigma_basic = total_queries * math.sqrt(2 * math.log(total_queries * 1.25 / delta)) / eps  # Basic composition
    sigma_advanced = math.sqrt(2 * math.log(1.25 / delta)) / eps_prime  # Advanced composition
    sigma_moments = 2 * math.sqrt(total_queries * math.log(1 / delta)) / eps  # Moments accountant asymptotic bound

    a = (-2 * (math.log(delta) - eps) + math.sqrt(
        (2 * (math.log(delta) - eps)) ** 2 + 4 * eps * (math.log(delta) + eps))) / (
                    2 * eps)  # Optimal alpha value for RDP can be solved exactly in the Gaussian case by applying the weak conversion bound
    C = math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
    sigma_rdp_weak = math.sqrt(total_queries * a * (a - 1) / (2 * (math.log(delta) + (
                a - 1) * eps)))  # RDP using the stronger conversion bound (a,r)-RDP to (eps,delta)-DP


    obj = RDPAccountant(eps, delta, 1, None, total_queries,
                        verbose=False)  # RDP using the tf implementation which uses the stronger conversion bound (also supports tight subsampling analysis)

    eps=eps/2
    a = (-2 * (math.log(delta) - eps) + math.sqrt(
        (2 * (math.log(delta) - eps)) ** 2 + 4 * eps * (math.log(delta) + eps))) / (
                    2 * eps)  # Optimal alpha value for RDP can be solved exactly in the Gaussian case by applying the weak conversion bound
    C = math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
    sigma_rdp_weak_2 = math.sqrt(total_queries * a * (a - 1) / (2 * (math.log(delta) + (
                a - 1) * eps)))  # RDP using the stronger conversion bound (a,r)-RDP to (eps,delta)-DP

    print("Alpha found directly using weak bound:", a)
    print("Optimal Alpha:", obj.opt_alpha)

    print("\n")
    print("SIGMA VALUES")
    print("Basic Composition:", sigma_basic)
    print("Advanced Composition:", sigma_advanced)
    print("Sigma Moments:", sigma_moments)
    print("RDP Accountant via weak bound:", sigma_rdp_weak)
    print("RDP Accountant via weak bound:", sigma_rdp_weak_2)
    print("RDP Accountant", obj.sigma)
    print("RDP Accountant", RDPAccountant(1.5, delta, 1, None, total_queries,
                        verbose=False).sigma)

budget_examples()
