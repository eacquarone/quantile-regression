import pandas as pd
from processing.delta_generator import DeltaGenerator

import warnings
warnings.filterwarnings('ignore')


def main():

    census80qr = pd.read_csv("Data/census80qr.csv")

    delta_generator = DeltaGenerator(census80qr)

    result = delta_generator.process()

    result.to_csv("Data/census80delta.csv", index=False)


if True:
    main()
