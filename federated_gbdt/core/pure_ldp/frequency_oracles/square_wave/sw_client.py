from federated_gbdt.core.pure_ldp.core import FreqOracleClient
import numpy as np
import random
import math


class SWClient(FreqOracleClient):
    def __init__(self, epsilon, index_mapper=None):
        super().__init__(epsilon=epsilon, d=None, index_mapper=index_mapper)
        self.update_params(epsilon, d=None, index_mapper=index_mapper)

    def update_params(self, epsilon=None, d=None, index_mapper=None):
        super().update_params(epsilon, d, index_mapper)
        ee = np.exp(self.epsilon)
        if epsilon is not None or d is not None:
            self.b = ((self.epsilon * ee) - ee + 1) / (2 * ee * (ee - 1 - self.epsilon))
            self.p = ee / ((2 * self.b * ee) + 1)
            self.q = 1 / ((2 * self.b * ee) + 1)

    def _perturb(self, data):
        if random.random() <= 2 * self.b * self.p:
            perturbed_val = random.uniform(data - self.b, data + self.b)
        else:
            if random.random() < 0.5:
                perturbed_val = random.uniform(-self.b, data - self.b)
            else:
                perturbed_val = random.uniform(data + self.b, 1 + self.b)

        return perturbed_val

    def privatise(self, data):
        # index = self.index_mapper(data)
        return self._perturb(data)
