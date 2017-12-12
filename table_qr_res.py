import numpy as np
from scipy.linalg import sqrtm
import statsmodels.formula.api as smf

def table_rq_res(formula, taus, data, alpha, n, sigma, jacobian):
    m = len(taus)
    tab = pd.DataFrame([])
    setab = pd.DataFrame([])
    for i in range(m):
        fit_model = smf.quantreg(formula, data)
        fit = sqr_model.fit(q=taus[i])
        coeff = np.dot(R.T, np.array(fit.params))
        tab[str(i)] = coeff
        sigmatau = sigma(data,n,tasu[i],fit.resid)
        jacobtau = jacobian(data, n, taus[i], fit.resid, alpha)
        solved_jacobtau = np.linalg.solve(jacobtau)
        V = np.dot(np.dot(solved_jacobtau, sigmatau), solved_jacobtau)/n
        secoeff = np.linalg.matrix_power(np.dot(np.dot(R.T, V), R), 0.5)
        setab[str(i)] = secoeff
    tab = tab.transpose()
    setab = setab.transpose()
    return(tab, setab)