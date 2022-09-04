from federated_gbdt.core.pure_ldp.core import FreqOracleClient
import numpy as np
import random
import math

class HMClient(FreqOracleClient):
    def __init__(self, epsilon, max, min, index_mapper=None, perturb_type="hybrid"):

        super().__init__(epsilon=epsilon, d=None, index_mapper=index_mapper)
        self.update_params(epsilon, d=None, index_mapper=index_mapper)
        self.perturb_type = perturb_type
        self.max = max
        self.min = min
        self.normalised_input = np.array([])

    def update_params(self, epsilon=None, d=None, index_mapper=None):
        super().update_params(epsilon, d, index_mapper)
        ee = np.exp(self.epsilon)

        if epsilon is not None or d is not None:
            self.p = ee/(ee + 1)
            self.q = 1/(ee + 1)
            self.ee2 = np.exp(self.epsilon/2)
            self.s = (self.ee2 + 1) / (self.ee2 - 1)
            self.alpha = 1 - (np.exp(-self.epsilon/2))

    def _stochastic_rounding(self, norm_data):
        # perturb mechanism for Stochastic Rounding
        if random.random() <= self.q + (((self.p - self.q)*(1 - norm_data)) / 2):
            v_prime = -1
        else:
            v_prime = +1

        result = v_prime/(self.p-self.q)
        return result

    def _piecewise_mechanism(self, norm_data):
        # perturb for piecewise mechanism
        if random.random() <= self.ee2/(self.ee2 + 1):
            v_prime = random.uniform(((self.ee2 * norm_data) - 1) / (self.ee2 - 1),
                                     ((self.ee2 * norm_data) + 1) / (self.ee2 - 1))
        else:
            if random.random() <= (((self.ee2 * norm_data) - 1) / (self.ee2 - 1) + self.s) / (self.s+1): # Weight sampling uniformly from the left-region by it's size
                v_prime = random.uniform(-self.s, ((self.ee2 * norm_data) - 1) / (self.ee2 - 1))
            else:
                v_prime = random.uniform(((self.ee2 * norm_data) + 1) / (self.ee2 - 1), self.s)
        return v_prime

    def _pm2(self, v):
        """
        Piecewise Mechanism, from paper: Collecting and Analyzing Multidimensional Data with Local Differential Privacy
        """
        z = np.e ** (self.epsilon / 2)
        P1 = (v + 1) / (2 + 2 * z)
        P2 = z / (z + 1)
        P3 = (1 - v) / (2 + 2 * z)

        C = (z + 1) / (z - 1)
        g1 = (C + 1) * v / 2 - (C - 1) / 2
        g2 = (C + 1) * v / 2 + (C - 1) / 2

        rnd = np.random.random()
        if rnd < P1:
            result = -C + np.random.random() * (g1 - (-C))
        elif rnd < P1 + P2:
            result = (g2 - g1) * np.random.random() + g1
        else:
            result = (C - g2) * np.random.random() + g2
        return result

    def _perturb(self, data):
        # normalise the input data into the domain [-1,1]
        norm_data = ((2*(data - self.min)) / (self.max - self.min)) - 1
        result = 0
        if self.perturb_type == "hybrid":
            # when epsilon > 0.61 use PW with prob alpha and SR with 1-alpha
            if self.epsilon > 0.61:
                if random.random() <= self.alpha:
                    result = self._piecewise_mechanism(norm_data)
                else:
                    result = self._stochastic_rounding(norm_data)
            # when epsilon <= 0.61 use SR only
            else:
                result = self._stochastic_rounding(norm_data)
        elif self.perturb_type == "sr":
            result = self._stochastic_rounding(norm_data)
        elif self.perturb_type == "pm":
            result = self._piecewise_mechanism(norm_data)

        result = ((result + 1) * (self.max - self.min) / 2) + self.min
        return result

    def privatise(self, data):
        return self._perturb(data)
