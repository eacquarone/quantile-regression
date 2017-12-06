import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

class QRGenerator(object):

    QUANTILES_LIST = [10, 25, 50, 75, 90]

    def __init__(self, data, data_cq):
        self.data = data
        self.data_cq = data_cq

    def fit_model(self, q, modele):
        res = modele.fit(q=q)
        return [np.int(q*100), res.params['Intercept'], res.params['educ']]

    def process(self):
        # census80qr creation
        census80qr = pd.merge(self.data, self.data_cq, how ="inner", on = "educ")
        census80qr["logwk_weighted"] = census80qr["logwk"]*census80qr["perwt"]

        QR = smf.quantreg('logwk_weighted ~ educ', census80qr)

        models = [self.fit_model(x/100, QR) for x in self.QUANTILES_LIST]
        models = pd.DataFrame(models, columns=['tau', 'a', 'b'])

        get_y = lambda a, b, x: a + b * x

        for tau in self.QUANTILES_LIST:
            census80qr["qrlogwk_q"+str(tau)] = get_y(models.loc[models.tau == tau, "a"],
                                                     models.loc[models.tau == tau, "b"],
                                                     census80qr["educ"])

        ols = smf.ols('logwk_weighted ~ educ', census80qr).fit()

        census80qr["qrlogwk_ols"] = get_y(ols.params['Intercept'],
                                          ols.params['educ'], 
                                          census80qr["educ"])

        # No fuckin' idea why we are doing this shit
        census80qr["cqlogwk_ols"] = census80qr["logwk"]

        for tau in self.QUANTILES_LIST:
            census80qr["delta_q"+str(tau)] = census80qr["qrlogwk_q"+str(tau)] - census80qr["cqlogwk_q"+str(tau)]
            census80qr["epsilon_q"+str(tau)] = census80qr["logwk"] - census80qr["cqlogwk_q"+str(tau)]


        # census80g creation
        collapse_mean = []
        for name in ['delta', 'qrlogwk', 'cqlogwk']:
            collapse_mean += [col for col in census80qr.columns if col.startswith(name)]

        collapse_dico = {}
        for column in collapse_mean:
            collapse_dico[column] = 'mean'
        collapse_dico['perwt'] = 'sum'

        census80g = census80qr.groupby("educ").agg(collapse_dico)

        census80g.reset_index(inplace = True)
        census80g['educ'] =census80g['educ'].apply(lambda x : np.int(x))

        sperwt = sum(census80g['perwt'])
        census80g['preduc'] =  census80g['perwt']/sperwt

        for tau in self.QUANTILES_LIST:
            census80g['weighted_cqlogwk_q' + str(tau)] = census80g['cqlogwk_q' + str(tau)]*census80g['preduc']
            instructions = 'weighted_cqlogwk_q' + str(tau) + ' ~ educ'
            ols_tau = smf.ols(instructions, census80g).fit()
            census80g["cqrlogwk_q" + str(tau)] = get_y(ols_tau.params['Intercept'], ols_tau.params['educ'], census80g["educ"])

        keep_columns = []
        for name in ['delta', 'preduc', 'educ', 'cqlogwk', 'qrlogwk', 'cqrlogwk']:
            keep_columns += [col for col in census80g.columns if col.startswith(name)]
        census80g = census80g[keep_columns]

        census80g.sort_values('educ')

        result = [census80qr, census80g]
        return(result)