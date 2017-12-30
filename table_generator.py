import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

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

	def process(self, census = None):
		if census is not None:
			self.census = census
		census80t2 = self.census.copy()

		census80t2["highschool"] = (census80t2["educ"] == 12).apply(lambda x: np.int(x))
		census80t2["college"] = (census80t2["educ"] == 16).apply(lambda x: np.int(x))

		# census80t2["logwk"] = census80t2["logwk"]*census80t2["perwt"]
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
			collapse_dico[col] = lambda x : np.mean(x)

		# Moche mais la boucle bug
		collapse_dico["q10"] = lambda x : np.percentile(x, q = 10)
		collapse_dico["q25"] = lambda x : np.percentile(x, q = 25)
		collapse_dico["q50"] = lambda x : np.percentile(x, q = 50)
		collapse_dico["q75"] = lambda x : np.percentile(x, q = 75)
		collapse_dico["q90"] = lambda x : np.percentile(x, q = 90)

		for col in collapse_dico:
			census80t2[col] = census80t2[col]*census80t2["perwt"]

		collapse_dico["perwt"] = "sum"

		census80t2g = census80t2.groupby(["educ","black","exper"]).agg(collapse_dico)
		census80t2g.reset_index(inplace = True)

		# Second part
		census80t2g["True"] = 1
		list_tab = []
		for cond in [census80t2g['educ'] >= 0,census80t2g['educ'] == 12,census80t2g['educ'] == 16]:
			census80t2g_filtered = census80t2g.loc[cond,:]
			mean_list = [col for col in census80t2g_filtered.columns if col.startswith("aq") or col.startswith("q")]
			mean_list += ["highschool", "college"]

			perwt = np.sum(census80t2g_filtered["perwt"])
			census80t2g_filtered['perwt'] = census80t2g_filtered["perwt"]/perwt

			census80t2g_agg = census80t2g_filtered[mean_list].aggregate(lambda x: np.average(x, weights = census80t2g_filtered['perwt']))
			census80t2g_agg['perwt'] = np.int(perwt)

			for prefix in ["","a"]:
				census80t2g_agg[prefix + 'd9010'] = census80t2g_agg[prefix + 'q90']-census80t2g_agg[prefix + 'q10']
				census80t2g_agg[prefix + 'd7525'] = census80t2g_agg[prefix + 'q75']-census80t2g_agg[prefix + 'q25']
				census80t2g_agg[prefix + 'd9050'] = census80t2g_agg[prefix + 'q90']-census80t2g_agg[prefix + 'q50']
				census80t2g_agg[prefix + 'd5010'] = census80t2g_agg[prefix + 'q50']-census80t2g_agg[prefix + 'q10']
			list_tab += [census80t2g_agg]
		return(census80t2g, list_tab)