import pandas as pd
from matplotlib import pyplot as plt


def main():
    year = 80

    data = pd.read_csv("Data/census{}g.csv".format(year),
                       index_col="educ")
    data_gimp = pd.read_csv("Data/census{}gimp.csv".format(year),
                            index_col="educ")
    data_gd = pd.read_csv("Data/census{}gd.csv".format(year),
                          index_col="educ")
    data = data.join([data_gimp, data_gd]).reset_index()

    for i, (letter, q) in enumerate(zip(["A", "B", "C"], [10, 50, 90])):
        plt.subplot(2, 3, i + 1)
        q = str(q)
        plt.plot(data["educ"], data["cqlogwk_q" + q], "o", label="CQ")
        plt.plot(data["educ"], data["qrlogwk_q" + q], "-", label="QR")
        plt.plot(data["educ"], data["cqrlogwk_q" + q], "--", label="MD")
        plt.xlabel("Schooling")
        plt.ylabel("Log-earnings")
        plt.title(letter + ". $\\tau$ = 0." + q)
        plt.legend()

    for i, (letter, q) in enumerate(zip(["D", "E", "F"], [10, 50, 90])):
        plt.subplot(2, 3, 3 + i + 1)
        q = str(q)
        plt.plot(data["educ"], data["awqr5_q" + q], "-.", label="QR")
        plt.plot(data["educ"], data["wqr5_q" + q], "-", label="Imp.")
        plt.plot(data["educ"], data["dwqr2_q" + q], "--", label="Dens.")
        plt.ylim(0, 0.5)
        plt.xlabel("Schooling")
        plt.ylabel("Weight")
        plt.title(letter + ". $\\tau$ = 0." + q)
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
