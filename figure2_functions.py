import numpy as np
import pandas as pd
from scipy.stats import norm


def main():

    import numpy as np

    def add_columns(df):
        df['weight * educ'] = df["weight"] * df["educ"]
        df["weight * exper"] = df["weight"] * df["exper"]
        df["weight * black"] = df["weight"] * df["black"]
        df["weight * exper2"] = df["weight"] * df["exper"] ** 2
        return df

    def sigma(df, n, tau, res):
        df['sub'] = (tau - (res <= 0)).values
        df["weight"] = df["perwt"] * df['sub']
        add_columns(df)
        x = df[['weight', 'weight * educ', 'weight * exper', 'weight * exper2', 'weight * black']]
        return np.dot(x.T.values, x.values) / float(n)

    def sigma2(df, n, tau):
        df["weight"] = df["perwt"] * np.sqrt(tau * (1 - tau))
        add_columns(df)
        x = df[["weight"]]
        return np.dot(x.T.values, x.values) / float(n)

    def sigma0(df, n, tau):
        df["weight"] = df["perwt"] * np.sqrt(tau * (1 - tau))
        add_columns(df)
        x = ["weight", "weight * educ", "weight * exper", "weight * exper2", "weight * black"];
        return np.dot(df[x].T.values, df[x].values) / float(n)

    def jacobian(df, n, tau, res, alpha):
        hn = (norm.pdf(1. - alpha / 2.) ** (2. / 3.)) * ((1.5 * norm.pdf(norm.ppf(tau)) ** 2.)
                                                      / (2. * norm.ppf(tau)**2 + 1.))**(1. / 3) * (n**(-1. / 3))
        df["sub"] = (abs(res) <= hn).values
        df["weight"] = np.sqrt(df["perwt"]) * df["sub"]
        add_columns(df)
        x = ["weight", "weight * educ", "weight * exper", "weight * exper2", "weight * black"]
        return np.dot(df[x].T.values, df[x].values) / 2*hn*float(n)

    def jacobian2(df, n, tau, res, alpha):
        hn = (norm.pdf(1. - alpha / 2.) ** (2. / 3.)) * ((1.5 * norm.pdf(norm.ppf(tau)) ** 2.)
                                                      / (2. * norm.ppf(tau)**2 + 1.))**(1. / 3) * (n**(-1. / 3))
        df["sub"] = (abs(res) <= hn).values
        df["weight"] = np.sqrt(df["perwt"]) * df["sub"]
        x = ["weight"]
        return np.dot(df[x].T.values, df[x].values) / 2*hn*float(n)

    df = pd.read_stata("~/Documents/Semi_And_Non_Parametric_Econometrics/quantile-regression/Data/census80.dta")
    res = pd.read_csv("~/Documents/Semi_And_Non_Parametric_Econometrics/quantile-regression/Data/residuals.csv",index_col = 0)

    print jacobian2(df, len(df), 0.01, res, 0.05)

if __name__ == '__main__':
    main()

