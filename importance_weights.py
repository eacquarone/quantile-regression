import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity

from Data.helpers import silverman_factor


def main():
    """Computes importance weights and saves them to a csv file."""

    data = pd.read_csv("Data/census80delta.csv")

    data_qr = pd.read_stata("Data/census80qr.dta")
    groups = data_qr.groupby("educ")

    for tau in [10, 25, 50, 75, 90]:
        for educ in range(5, 20 + 1):
            estimation_points = groups.get_group(educ)
            estimation_points = estimation_points["epsilon_q{}".format(tau)]
            kde = KernelDensity(
                kernel='gaussian',
                bandwidth=silverman_factor(estimation_points)
            )

            kde.fit(estimation_points.values.reshape(-1, 1))

            s = "{}_q{}".format(educ, tau)
            data["epsilon" + s] = data["delta" + s] * (data["u"] - 1) / 100.

            data["wdensity" + s] = 1 - (data["u"] - 1) / 100.
            data["wdensity" + s] *= np.exp(
                kde.score_samples(data["epsilon" + s].values.reshape(-1, 1))
            )

    means = data.mean(axis=0)
    for tau in [10, 25, 50, 75, 90]:
        data_qr["wdensity_q{}".format(tau)] = 0

        for educ in range(5, 20 + 1):
            s = "{}_q{}".format(educ, tau)
            idx = groups.get_group(educ).index

            data_qr.loc[
                idx, "wdensity_q{}".format(tau)
            ] = means["wdensity" + s]

    weights = groups.mean()[[c for c in data_qr if "density" in c]]
    weights.columns = ["impweight_" + c.split('_')[1] for c in weights]

    weights_rescaled = (weights / weights.sum(axis=0)).copy()
    weights_rescaled.columns = ["wqr5_" + c.split('_')[1] for c in weights]

    data = weights.join(weights_rescaled)
    data.to_csv("Data/census80gimp.csv")
    print(data.head())


if __name__ == '__main__':
    main()
