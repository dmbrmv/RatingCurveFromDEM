from lmfit import Model


def power_function(x, power, c):
    """A function that models a power law given by:

    y = x^power + c

    This function is used to model the power law decay in the observed data.

    Parameters
    ----------
    x : float or numpy.array
        The x values of the power law
    power : float
        The exponent of the power law
    c : float
        The y intercept of the power law

    Returns
    -------
    float or numpy.array
        The y values of the power law
    """
    return (x**power) + c


def line(x, b, slope):
    """This function models a line given by:

    y = slope * x + b

    Args:
    ----
        x (float or numpy.array): The x values of the line
        b (float): The y intercept of the line
        slope (float): The slope of the line

    Returns:
    -------
        float or numpy.array: The y values of the line

    """
    return slope * x + b


division_model = Model(power_function) / Model(line)
sum_model = Model(power_function) + Model(line)
