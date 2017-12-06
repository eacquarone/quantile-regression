import pandas as pd


class DeltaGenerator(object):

    QUANTILES_LIST = [10, 25, 50, 75, 90]

    def __init__(self, qr):
        self.qr = qr

    def process(self):
        df = self.qr[["educ"] + ["delta_q%i" % i for i in self.QUANTILES_LIST]]
        for i in [10, 25, 50, 75, 90]:
            df["delta_q%i" % i] = df["delta_q%i" % i] * self.qr['perwt']

        df_educ = df.groupby("educ").mean()

        dict_values = []
        for tau in self.QUANTILES_LIST:
            for index in range(5, 21):
                dict_values.append(
                    ("delta%i_q%i" % (index, tau),
                     df_educ.loc[index]["delta_q%i" % tau]
                    )
                )
        dft = pd.DataFrame(dict_values).set_index(0).T
        result = pd.concat([dft] * 101).reset_index(drop='True')
        result["u"] = range(1, 102)
        return result
