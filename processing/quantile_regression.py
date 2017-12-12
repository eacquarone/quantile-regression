import pandas as pd
from processing.qr_generator import QRGenerator

import warnings
warnings.filterwarnings('ignore')


def main():
    census80 = pd.read_stata("Data/census80.dta")
    census80cq = pd.read_csv("Data/census80cq.csv")

    qr_generator = QRGenerator(census80, census80cq)

    result = qr_generator.process()

    result[0].to_csv("Data/census80qr.csv", index=False)
    result[1].to_csv("Data/census80g.csv", index=False)


if True:
    main()
