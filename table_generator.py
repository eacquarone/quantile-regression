import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from functools import partial

class TableGenerator(object):

	QUANTILES_LIST = [10, 25, 50, 75, 90]

	def __init__(self, census):
		self.census = census

	def fit_model(self, q, modele):
		res = modele.fit(q=q)
		return [np.int(q*100), 
				res.params['Intercept'], 
				res.params['educ'],
				res.params['black'],
				res.params['exper'],
				res.params['exper2']]

	def process(self):
		census80t2 = self.census.copy()

		census80t2["highschool"] = (census80t2["educ"] == 12).apply(lambda x: np.int(x))
		census80t2["college"] = (census80t2["educ"] == 16).apply(lambda x: np.int(x))

		# census80["logwk_weighted"] = census80["logwk"]*census80["perwt"]
		QR = smf.quantreg('logwk ~ educ + black + exper + exper2', self.census)

		models = [self.fit_model(x/100, QR) for x in self.QUANTILES_LIST]
		models = np.array(models)

		get_y = lambda a, b, x: a + np.dot(x,b)

		for tau in self.QUANTILES_LIST:
			census80t2["q"+str(tau)] = census80t2["logwk"]
			census80t2["aq"+str(tau)] = get_y(models[models[:,0] == tau,1], 
											   models[models[:,0] == tau,2:6].T,
											   census80t2[["educ","black", "exper", "exper2"]])

		census80t2.sort_values(["educ","black","exper"], inplace = True)

		collapse_mean = [col for col in census80t2.columns if col.startswith("aq")]
		collapse_mean += ["highschool", "college", "exper2"]

		collapse_dico = {}
		for col in collapse_mean:
			collapse_dico[col] = "mean"

		for tau in self.QUANTILES_LIST:
			collapse_dico["q"+str(tau)] = lambda x : np.percentile(x, q = tau)

		collapse_dico["perwt"] = "count"

		census80t2g = census80t2.groupby(["educ","black", "exper"]).agg(collapse_dico)
		census80t2g.reset_index(inplace = True)
		
		result = census80t2g
		return(result)