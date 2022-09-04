from federated_gbdt.core.pure_ldp.core import FreqOracleServer
import numpy as np
import math
import scipy
import random

from numba import jit

class SWServer(FreqOracleServer):
    def __init__(self, epsilon, d=1024, d_prime=1024, smooth=True, smc=False, index_mapper=None):
        super().__init__(epsilon, d=None, index_mapper=index_mapper)
        self.smc = smc
        self.smooth = smooth
        self.d = d  # Domain bins B_i, n
        self.d_prime = d_prime  # Randomised Bins \tilde{B}_j, m
        self.update_params(epsilon, d=None, index_mapper=index_mapper)
        self.aggregated_data = []

    def update_params(self, epsilon=None, d=None, index_mapper=None):
        super().update_params(epsilon, d, index_mapper)
        ee = np.exp(self.epsilon)
        if epsilon is not None or d is not None:
            self.b = ((self.epsilon * ee) - ee + 1) / (2 * ee * (ee - 1 - self.epsilon))
            self.p = ee / ((2 * self.b * ee) + 1)
            self.q = 1 / ((2 * self.b * ee) + 1)
            self.w = ((self.epsilon * ee) - ee + 1) / (2 * ee * (ee - 1 - self.epsilon)) * 2
            self.M = self.generate_M(self.d_prime, self.d)

    def aggregate(self, priv_data):
        self.aggregated_data.append(priv_data)
        self.n += 1

    def generate_M(self, m=1024, n=1024):
        # report matrix
        m_cell = (1 + self.w) / m
        n_cell = 1 / n

        transform = np.ones((m, n)) * self.q * m_cell
        for i in range(n):
            left_most_v = (i * n_cell)  # For bin B_i, this is the left boundary - v_min
            right_most_v = ((i + 1) * n_cell)  # Right boundary of B_i - v_max

            ll_bound = int(left_most_v / m_cell)
            lr_bound = int((left_most_v + self.w) / m_cell)
            rl_bound = int(right_most_v / m_cell)
            rr_bound = int((right_most_v + self.w) / m_cell)

            ll_v = left_most_v - self.w / 2
            rl_v = right_most_v - self.w / 2
            l_p = ((ll_bound + 1) * m_cell - self.w / 2 - ll_v) * (self.p - self.q) + self.q * m_cell
            r_p = ((rl_bound + 1) * m_cell - self.w / 2 - rl_v) * (self.p - self.q) + self.q * m_cell
            if rl_bound > ll_bound:
                transform[ll_bound, i] = (l_p - self.q * m_cell) * (
                        (ll_bound + 1) * m_cell - self.w / 2 - ll_v) / n_cell * 0.5 + self.q * m_cell
                transform[ll_bound + 1, i] = self.p * m_cell - (self.p * m_cell - r_p) * (
                        rl_v - ((ll_bound + 1) * m_cell - self.w / 2)) / n_cell * 0.5
            else:
                transform[ll_bound, i] = (l_p + r_p) / 2
                transform[ll_bound + 1, i] = self.p * m_cell

            lr_v = left_most_v + self.w / 2
            rr_v = right_most_v + self.w / 2
            r_p = (rr_v - (rr_bound * m_cell - self.w / 2)) * (self.p - self.q) + self.q * m_cell
            l_p = (lr_v - (lr_bound * m_cell - self.w / 2)) * (self.p - self.q) + self.q * m_cell
            if rr_bound > lr_bound:
                if rr_bound < m:
                    transform[rr_bound, i] = (r_p - self.q * m_cell) * (
                            rr_v - (rr_bound * m_cell - self.w / 2)) / n_cell * 0.5 + self.q * m_cell

                transform[rr_bound - 1, i] = self.p * m_cell - (self.p * m_cell - l_p) * (
                        (rr_bound * m_cell - self.w / 2) - lr_v) / n_cell * 0.5

            else:
                transform[rr_bound, i] = (l_p + r_p) / 2
                transform[rr_bound - 1, i] = self.p * m_cell

            if rr_bound - 1 > ll_bound + 2:
                transform[ll_bound + 2: rr_bound - 1, i] = self.p * m_cell

        return transform

    def difference_intervals(self, I1, I2):
        a_start, a_end = I1
        b_start, b_end = I2
        return min(abs(a_start - b_start), abs(a_start - b_end), abs(a_end - b_start), abs(a_end - b_end)), max(
            abs(a_start - b_start), abs(a_start - b_end), abs(a_end - b_start), abs(a_end - b_end))

    def EMS(self, priv_hist, iterations, threshold, smooth=False):
        if smooth:
            # smoothing matrix
            smoothing_factor = 2
            binomial_tmp = [scipy.special.binom(smoothing_factor, k) for k in range(smoothing_factor + 1)]
            smoothing_matrix = np.zeros((self.d, self.d))
            central_idx = int(len(binomial_tmp) / 2)
            for i in range(int(smoothing_factor / 2)):
                smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
            for i in range(int(smoothing_factor / 2), self.d - int(smoothing_factor / 2)):
                smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
            for i in range(self.d - int(smoothing_factor / 2), self.d):
                remain = self.d - i - 1
                smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]
            row_sum = np.sum(smoothing_matrix, axis=1)
            smoothing_matrix = (smoothing_matrix.T / row_sum).T

        # EMS
        theta = np.ones(self.d) / float(self.d)
        theta_old = np.zeros(self.d)
        r = 0
        sample_size = sum(priv_hist)
        old_logliklihood = 0

        while np.linalg.norm(theta_old - theta, ord=1) > 1 / sample_size and r < iterations:
            theta_old = np.copy(theta)
            X_condition = np.matmul(self.M, theta_old)

            TMP = self.M.T / X_condition

            P = np.copy(np.matmul(TMP, priv_hist))
            P = P * theta_old

            theta = np.copy(P / sum(P))

            # Smoothing step
            if smooth:
                theta = np.matmul(smoothing_matrix, theta)
                theta = theta / sum(theta)

            logliklihood = np.inner(priv_hist, np.log(np.matmul(self.M, theta)))
            imporve = logliklihood - old_logliklihood

            if r > 1 and abs(imporve) < threshold:
                # print("stop when", imporve / old_logliklihood, loglikelihood_threshold)
                break

            old_logliklihood = logliklihood

            r += 1
        return theta

    def g_density(self, v_prime, v):
        out = np.zeros(shape=v_prime.shape)
        out.fill(self.q)
        p_indexes = np.abs(v - v_prime) < self.b
        out[p_indexes] = self.p
        return out

    def _update_estimates(self):
        histogram, _ = np.histogram(self.aggregated_data, bins=self.d_prime, range=(-self.b, 1 + self.b))
        self.estimated_density = self.EMS(histogram, 100, 1e-3, self.smooth)

        return self.estimated_density

    def estimate(self, data, suppress_warnings=False):
        self.check_and_update_estimates()
        return self.estimated_density[data]

    def estimate_all(self, data_list, suppress_warnings=False):
        return [self.estimate(item) for item in data_list]

    def estimate_density(self, N=None, suppress_warnings=False):
        self.check_and_update_estimates()
        return self.estimated_density
