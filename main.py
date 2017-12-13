from processing import (
    conditional_quantiles, quantile_regression, delta,
    importance_weights, density_weights, histogram
)


def main():
    year = 80

    # Obtains nonparametric estimates of the conditional quantiles of log
    # earnings given schooling. These estimates are just the sample quantiles
    # of log earnings for each level of schooling.
    conditional_quantiles(year)

    # Quantile regressions and Chamberlain’s minimum distance estimates
    # fitting conditional quantiles to schooling across cells. The outcomes of
    # this programs are the csv data files census80g.csv, which contains
    # estimates of the conditional quantiles, and the quantile regression
    # and Chamberlain fitted values for each level of schooling;
    # and census80qr.csv, which contains the quantile regression residuals
    # and specification errors for each individual.
    quantile_regression(year)

    # Auxiliary program for estimating the importance weights.
    # This program obtains, for each level of schooling, the grid of values
    # for log earnings where the density is estimated. Using the csv file
    # census80qr.csv, the outcome of this program is the file census80delta.csv
    # that contains the grid of values for each level of schooling.
    delta(year)

    # Derives the importance weights by estimating kernel densities in the grid
    # of points obtained from delta() and weighted-averaging these densities.
    # The results, together with the quantile regression weights
    # (importance weights × histogram of schooling), are saved in the csv file
    # census80gimp.csv.
    importance_weights(year)

    # Obtains the density weights by estimating kernel densities at the
    # nonparametric estimates of the conditional quantiles. This program uses
    # the csv file census80qr.csv created by quantile_regression() and saves
    # the results to census80gd.csv.
    density_weights(year)

    # Saves individual levels of schooling in the csv file census80i.csv to
    # generate the histogram.
    histogram(year)


if __name__ == '__main__':
    main()
