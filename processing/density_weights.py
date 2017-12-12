import pandas as pd
from processing.density_weights_generator import DensityWeights


def main():

    census80qr = pd.read_csv("Data/census80qr.csv")
    density_weights = DensityWeights(census80qr)
    density_weights_estimates = density_weights.process()
    density_weights_estimates.to_csv("Data/census80gd.csv", index=False)


if True:
    main()
