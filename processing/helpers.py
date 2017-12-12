def silverman_factor(x):
    '''

    :param x: A pandas.DataFrame column
    :return: the ideal bandwidth according to silverman rule of thumb
    '''
    iqr = x.quantile(.75) - x.quantile(.25)
    m = min(x.std()**2, iqr / 1.349)
    n = len(x)
    return (.9 * m) / n**(.2)
