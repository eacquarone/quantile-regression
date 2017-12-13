import pandas as pd
from generators import (
    QRGenerator, DeltaGenerator, ImportanceWeights, DensityWeights
)

import warnings
warnings.filterwarnings('ignore')


def conditional_quantiles(year):
    census_data = pd.read_stata("Data/census{}.dta".format(year))

    for i in [10, 25, 50, 75, 90]:
        census_data["cqlogwk_q%i" % i] = census_data.logwk

    columns = [
        'educ', 'cqlogwk_q10', 'cqlogwk_q25',
        'cqlogwk_q50', 'cqlogwk_q75', 'cqlogwk_q90'
    ]

    census_data_cq = census_data[columns]

    result = census_data_cq.groupby('educ').agg({
        'cqlogwk_q10': (lambda x: x.quantile(.1)),
        u'cqlogwk_q25': (lambda x: x.quantile(.25)),
        u'cqlogwk_q50': (lambda x: x.quantile(.5)),
        u'cqlogwk_q75': (lambda x: x.quantile(.75)),
        u'cqlogwk_q90': (lambda x: x.quantile(.9))
    })

    result.to_csv("Data/census{}cq.csv".format(year))


def quantile_regression(year):
    census_data = pd.read_stata("Data/census{}.dta".format(year))
    census_cq = pd.read_csv("Data/census{}cq.csv".format(year))

    qr_generator = QRGenerator(census_data, census_cq)
    result = qr_generator.process()

    result[0].to_csv("Data/census{}qr.csv".format(year), index=False)
    result[1].to_csv("Data/census{}g.csv".format(year), index=False)


def delta(year):
    census_qr = pd.read_csv("Data/census{}qr.csv".format(year))

    delta_generator = DeltaGenerator(census_qr)
    result = delta_generator.process()

    result.to_csv("Data/census{}delta.csv".format(year), index=False)


def importance_weights(year):
    census_qr = pd.read_csv("Data/census{}qr.csv".format(year))
    census_delta = pd.read_csv("Data/census{}delta.csv".format(year))

    imp_generator = ImportanceWeights(census_qr, census_delta)
    result = imp_generator.process()

    result.to_csv("Data/census{}gimp.csv".format(year), index=False)


def density_weights(year):
    census_qr = pd.read_csv("Data/census{}qr.csv".format(year))

    density_weights = DensityWeights(census_qr)
    result = density_weights.process()

    result.to_csv("Data/census{}gd.csv".format(year), index=False)


def histogram(year):
    census_data = pd.read_stata("Data/census{}.dta".format(year))
    census_i = census_data[["educ"]]
    census_i.to_csv("Data/census{}i.csv".format(year), index=False)
