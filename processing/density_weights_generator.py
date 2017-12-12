
import numpy as np
import pandas as pd

from processing.helpers import silverman_factor

from sklearn.neighbors import KernelDensity


class DensityWeights(object):

    def __init__(self, qr):
        self.qr = qr

    def process(self):
        density_estimates = pd.DataFrame([])
        educ_groups = self.qr.groupby("educ")
        for tau in [10, 25, 50, 75, 90]:
            silverman_fact = silverman_factor(self.qr["epsilon_q%i" % tau])
            kde = KernelDensity(bandwidth=silverman_fact)
            values_tau = np.zeros(16)
            for index in range(16):
                group_educ = educ_groups.get_group(index + 5)
                values_tau[index] = self.estimate_density(group_educ["epsilon_q%i" % tau], kde)
            density_estimates["dweight_q%i" % tau] = values_tau
            density_estimates["dwqr2_q%i" % tau] = self.normalize(
                density_estimates["dweight_q%i" % tau])
        density_estimates["educ"] = range(5, 21)
        return density_estimates

    @staticmethod
    def estimate_density(x, kde_instance):
        kde_instance.fit(
            x.values.reshape(-1, 1)
        )
        return np.exp(kde_instance.score_samples(0)) / 2.

    @staticmethod
    def normalize(x):
        return x / sum(x)
