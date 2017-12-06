import pandas as pd
from density_weights_generator import DensityWeights


def main():

    census80qr = pd.read_stata("Data/census80qr.dta")
    density_weights = DensityWeights(census80qr)
    density_weights_estimates = density_weights.process()
    density_weights_estimates.to_csv("Data/census80gd.csv")


if __name__ == '__main__':
    main()
