import pandas as pd


def main():
    census80 = pd.read_stata("Data/census80.dta")
    census80i = census80[["educ"]]
    census80i.to_csv("Data/census80i.csv", index=False)


if True:
    main()
