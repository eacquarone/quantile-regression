import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.regression.quantile_regression import QuantReg, QuantRegResults
import os
Directory = os.getcwd()

census80 = pd.read_stata(Directory + "\census80.dta")
census80cq = pd.read_stata(Directory + "\census80cq.dta")
census80qr = pd.merge(census80, census80cq, how ="inner", on = "educ")

census80qr["logwk_weighted"] = census80qr["logwk"]*census80qr["perwt"]

QR = smf.quantreg('logwk_weighted ~ educ', census80qr)

def fit_model(q, modele):
    res = modele.fit(q=q)
    return [q, res.params['Intercept'], res.params['educ']] + \
            res.conf_int().ix['educ'].tolist()

quantiles = [.1, .25, .5, .75, .9]
models = [fit_model(x, QR) for x in quantiles]
models = pd.DataFrame(models, columns=['tau', 'a', 'b','lb','ub'])

get_y = lambda a, b, x: a + b * x
for i in range(models.shape[0]):
    census80qr["qrlogwk_q"+str(np.int(models.tau[i]*100))] = get_y(models.a[i], models.b[i], census80qr["educ"] )

ols = smf.ols('logwk_weighted ~ educ', census80qr).fit()

census80qr["qrlogwk_ols"] = get_y(ols.params['Intercept'], ols.params['educ'], census80qr["educ"])

# No fuckin' idea why we are doing this shit
census80qr["cqlogwk_ols"] = census80qr["logwk"]

tau_quantiles = [np.int(x*100) for x in quantiles]
for tau in tau_quantiles:
	census80qr["delta_q"+str(tau)] = census80qr["qrlogwk_q"+str(tau)] - census80qr["cqlogwk_q"+str(tau)]
	census80qr["epsilon_q"+str(tau)] = census80qr["logwk"] - census80qr["cqlogwk_q"+str(tau)]

census80qr.to_stata(Directory + '\census80qr.dta')

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

for tau in tau_quantiles:
	census80g['weighted_cqlogwk_q' + str(tau)] = census80g['cqlogwk_q' + str(tau)]*census80g['preduc']
	instructions = 'weighted_cqlogwk_q' + str(tau) + ' ~ educ'
	ols_tau = smf.ols(instructions, census80g).fit()
	census80g["cqrlogwk_q" + str(tau)] = get_y(ols_tau.params['Intercept'], ols_tau.params['educ'], census80g["educ"])

keep_columns = []
for name in ['delta', 'preduc', 'educ', 'cqlogwk', 'qrlogwk', 'cqrlogwk']:
	keep_columns += [col for col in census80g.columns if col.startswith(name)]
census80g = census80g[keep_columns]

census80g.sort_values('educ')

census80g.to_stata(Directory + '\census80g.dta')