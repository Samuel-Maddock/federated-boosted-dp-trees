from federated_gbdt.core.pure_ldp.core import FreqOracleServer
import numpy as np


class HMServer(FreqOracleServer):
    def __init__(self, epsilon, d, index_mapper=None):
        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.update_params(epsilon, d, index_mapper=index_mapper)
        self.aggregated_data = []

    def update_params(self, epsilon=None, d=None, index_mapper=None):
        super().update_params(epsilon, d, index_mapper)

    def aggregate(self, priv_data):
        self.aggregated_data.append(priv_data)
        self.n += 1

    def _update_estimates(self):
        mean = np.mean(self.aggregated_data)
        return mean

    def estimate(self, suppress_warnings=False):
        """
        Calculates a frequency estimate of the given data item
        Args:
            data: data item
            suppress_warnings: Optional boolean - Supresses warnings about possible inaccurate estimations
        Returns: float - frequency estimate
        """
        self.check_warnings(suppress_warnings=suppress_warnings)
        return self._update_estimates()
