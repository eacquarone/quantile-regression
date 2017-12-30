import numpy as np
from scipy.linalg import sqrtm
import statsmodels.formula.api as smf

# A partir de data
def subsamplek(formula, V, tau, coeffs, data, n, b, B, R):
    k =  pd.DataFrame([])
    RVR = np.linalg.matrix_power(np.dot(np.dot(R.T, V), R)/b, -0.5)
    for s in range(B):
        sing = 0
        while sing == 0:
            sdata = data.sample(b, replace=True, weights=df['perwt'])
            x = np.matrix(sdata["perw"]*sdata["educ"],
                       sdata["perw"]*sdata["exper"],
                       sdata["perw"]*sdata["exper2"],
                       sdata["perw"]*sdata["black"],
                       sdata["perw"])
            sing = np.linalg.det(np.dot(x.T,x))
        # Didn't use weights here
        sqr_model = smf.quantreg(formula, sdata)
        sqr = sqr_model.fit(q=tau)
        new_column = np.abs(np.dot(np.dot(RVR, R.T), coeffs-np.array(sqr.params)))
        k[str(s)] = new_column
    return(k)