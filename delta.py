import pandas as pd
from delta_generator import DeltaGenerator


def main():

    census80qr = pd.read_stata("Data/census80qr.dta")

    delta_generator = DeltaGenerator(census80qr)

    result = delta_generator.process()

    result.to_csv("Data/census80delta.csv")


if __name__ == '__main__':
    main()

