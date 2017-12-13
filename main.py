from processing import (
    conditional_quantiles, quantile_regression, delta,
    importance_weights, density_weights, histogram
)


def main():
    year = 80
    conditional_quantiles(year)
    quantile_regression(year)
    delta(year)
    importance_weights(year)
    density_weights(year)
    histogram(year)


if __name__ == '__main__':
    main()
