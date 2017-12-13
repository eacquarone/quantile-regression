import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.neighbors import KernelDensity


QUANTILES_LIST = [10, 25, 50, 75, 90]


def silverman_factor(x):
    '''
    Computes the Silverman factor.
    :param x: A pandas.DataFrame column
    :return: the ideal bandwidth according to the Silverman rule of thumb
    '''
    iqr = x.quantile(.75) - x.quantile(.25)
    m = min(x.std()**2, iqr / 1.349)
    n = len(x)
    return (.9 * m) / n**(.2)


class CQGenerator(object):
    def __init__(self, data):
        self.data = data

    def process(self):
        census_data = self.data

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

        return result


class QRGenerator(object):
    def __init__(self, data, data_cq):
        self.data = data
        self.data_cq = data_cq

    @staticmethod
    def fit(model, q):
        res = model.fit(q=q)
        return np.int(q * 100), res.params['Intercept'], res.params['educ']

    @staticmethod
    def predict(intercept, slope, x):
        f = np.vectorize(lambda a, b, x: a + b * x)
        y = f(intercept, slope, x)
        return y

    def process(self):
        # data_qr creation
        data_qr = pd.merge(self.data_cq, self.data, how="inner", on="educ")
        data_qr["logwk_weighted"] = data_qr["logwk"] * data_qr["perwt"]

        QR = smf.quantreg('logwk_weighted ~ educ', data_qr)

        models = [self.fit(QR, q / 100.) for q in QUANTILES_LIST]
        models = pd.DataFrame(models, columns=['tau', 'a', 'b'])

        for tau in QUANTILES_LIST:
            a = models.loc[models.tau == tau, "a"]
            b = models.loc[models.tau == tau, "b"]
            x = data_qr["educ"]

            data_qr["qrlogwk_q" + str(tau)] = self.predict(a, b, x)

        ols = smf.ols('logwk_weighted ~ educ', data_qr).fit()
        intercept = ols.params['Intercept']
        slope = ols.params['educ']
        x = data_qr["educ"]
        data_qr["qrlogwk_ols"] = self.predict(intercept, slope, x)
        data_qr["cqlogwk_ols"] = data_qr["logwk"]

        for tau in QUANTILES_LIST:
            data_qr["delta_q" + str(tau)] = data_qr["qrlogwk_q" + str(tau)]
            data_qr["delta_q" + str(tau)] += - data_qr["cqlogwk_q" + str(tau)]

            data_qr["epsilon_q" + str(tau)] = data_qr["logwk"]
            data_qr["epsilon_q" + str(tau)] += - data_qr["cqlogwk_q" + str(tau)]

        # data_g creation
        collapse_mean = []
        for name in ['delta', 'qrlogwk', 'cqlogwk']:
            collapse_mean += [
                col for col in data_qr.columns if col.startswith(name)
            ]

        collapse_dico = {}
        for column in collapse_mean:
            collapse_dico[column] = 'mean'
        collapse_dico['perwt'] = 'sum'

        data_g = data_qr.groupby("educ").agg(collapse_dico)
        data_g.reset_index(inplace=True)
        data_g['educ'] = data_g['educ'].apply(lambda x: np.int(x))

        sperwt = sum(data_g['perwt'])
        data_g['preduc'] = data_g['perwt'] / sperwt

        for tau in QUANTILES_LIST:
            data_g['weighted_cqlogwk_q' + str(tau)] = data_g['cqlogwk_q' + str(tau)]
            data_g['weighted_cqlogwk_q' + str(tau)] *= data_g['preduc']

            instructions = 'weighted_cqlogwk_q' + str(tau) + ' ~ educ'
            ols_tau = smf.ols(instructions, data_g).fit()
            a = ols_tau.params['Intercept']
            b = ols_tau.params['educ']
            x = data_g["educ"]

            data_g["cqrlogwk_q" + str(tau)] = self.predict(a, b, x)

        keep_columns = []
        for name in ['delta', 'preduc', 'educ',
                     'cqlogwk', 'qrlogwk', 'cqrlogwk']:

            keep_columns += [
                col for col in data_g.columns if col.startswith(name)
            ]

        data_g = data_g[keep_columns]
        data_g.sort_values('educ', inplace=True)

        result = [data_qr, data_g]
        return(result)


class DeltaGenerator(object):
    def __init__(self, qr):
        self.qr = qr

    def process(self):
        df = self.qr[["educ"] + ["delta_q%i" % i for i in QUANTILES_LIST]]
        for i in [10, 25, 50, 75, 90]:
            df["delta_q%i" % i] = df["delta_q%i" % i] * self.qr['perwt']

        df_educ = df.groupby("educ").mean()

        dict_values = []
        for tau in QUANTILES_LIST:
            for index in range(5, 21):
                dict_values.append(
                    ("delta%i_q%i" % (index, tau),
                     df_educ.loc[index]["delta_q%i" % tau])
                )
        dft = pd.DataFrame(dict_values).set_index(0).T
        result = pd.concat([dft] * 101).reset_index(drop='True')
        result["u"] = range(1, 102)
        return result


class IWGenerator(object):
    def __init__(self, qr, delta):
        self.qr = qr
        self.delta = delta

    def process(self):
        data = self.delta
        data_qr = self.qr
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
        return data


class DWGenerator(object):

    def __init__(self, qr):
        self.qr = qr

    def process(self):
        density_estimates = pd.DataFrame([])
        educ_groups = self.qr.groupby("educ")
        for tau in [10, 25, 50, 75, 90]:
            silverman_fact = silverman_factor(self.qr["epsilon_q%i" % tau])
            kde = KernelDensity(bandwidth=silverman_fact)
            values_tau = np.zeros(16)
            for index in range(16):
                group_educ = educ_groups.get_group(index + 5)
                values_tau[index] = self.estimate_density(
                    group_educ["epsilon_q%i" % tau], kde
                )

            density_estimates["dweight_q%i" % tau] = values_tau
            density_estimates["dwqr2_q%i" % tau] = self.normalize(
                density_estimates["dweight_q%i" % tau])
        density_estimates["educ"] = range(5, 21)
        return density_estimates

    @staticmethod
    def estimate_density(x, kde_instance):
        kde_instance.fit(
            x.values.reshape(-1, 1)
        )
        return np.exp(kde_instance.score_samples(0)) / 2.

    @staticmethod
    def normalize(x):
        return x / sum(x)
