import numpy as np
import pandas as pd


def main():
    data = pd.read_stata("Data/census80qr.dta")
    group = data.groupby("educ")
    print(group.head())

    generated_columns = {}
    for tau in [10, 25, 50, 75, 90]:
        for educ in range(5, 20 + 1):
            name = "epsilon{}_q{}".format(educ, tau)
            delta_name = "delta_q{}".format(tau)

            delta = group.get_group(educ)[delta_name]
            assert len(set([v for v in delta.values])) == 1
            delta = delta.values[0]

            generated_columns[name] = delta * np.linspace(0, 1, 100)

    # WIP


if __name__ == '__main__':
    main()
