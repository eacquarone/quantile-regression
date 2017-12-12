import pandas as pd


def main():
    census_data = pd.read_stata("Data/census80.dta")

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

    result.to_csv("Data/census80cq.csv")


if True:
    main()
