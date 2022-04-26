import numpy as np
from numbers import Number

from .numbaSpecialFns import numba_gammaln as gammaln
from .numbaSpecialFns import numba_betainc as betainc

# from   scipy.special import gammaln, hyp2f1, hyp1f1
from scipy.integrate import quad
from scipy.stats import t as scipy_trv
from scipy.optimize import minimize
from numba import cfunc
import warnings

from numba import njit

np.seterr(all='raise')


@njit
def tcdf_1d(x, df):
    """
    Computes the CDF of the t-distribution. See https://archive.lib.msu.edu/crcmath/math/math/s/s814.htm for relevant
    formulas. This implementation is relatively fast as it is compiled using Numba.

    :param x: Numpy array of real numbers where we wish to evaluate the CDF.
    :param df: Positive scalar corresponding to the degrees of freedom
    :return: CDF values at each of the x locations
    """
    tcdf_vals_v2 = 0.5 * betainc(0.5 * df, 0.5, df / (df + x ** 2))
    inds = x > 0
    tcdf_vals_v2[inds] = 1 - tcdf_vals_v2[inds]
    return tcdf_vals_v2


# This implementation is fairly slow, suggest not using this
def tspdf_1d_scipy(x, loc, scale, df, skew):
    """
    Computes the PDF of the skew-t-distribution. This uses the Azzalini and Capitanio as described in https://doi.org/10.1111/1467-9868.00391
    Note, this implementation uses the scipy stack and is relatively slow when used to compute a large number of values.

    :param x: Numpy array corresponding to locations where we wish to evaluate the pdf
    :param loc: Location parameter; real number
    :param scale: Scale parameter; positive real number
    :param df: Degrees of freedom; positive real number
    :param skew: Skewness parameter; real number
    :return: Values of the PDF at each location x
    """
    z = (x - loc) / np.sqrt(scale)
    return 2 * scipy_trv.pdf(z, df + 1) * scipy_trv.cdf(skew * z, df + 1)


@njit
def tspdf_1d(x, loc, scale, df, skew):
    """
    Computes the PDF of the skew-t-distribution. This uses the Azzalini and Capitanio as described in https://doi.org/10.1111/1467-9868.00391
    Note, this implementation is compiled using Numba and is significantly faster than the one based on the scipy stack.

    :param x: Numpy array corresponding to locations where we wish to evaluate the pdf
    :param loc: Location parameter; real number
    :param scale: Scale parameter; positive real number
    :param df: Degrees of freedom; positive real number
    :param skew: Skewness parameter; real number
    :return: Values of the PDF at each location x
    """
    return np.exp(tslogpdf_1d(x, loc, scale, df, skew))


@njit
def tslogpdf_1d(x, loc, scale, df, skew):
    """
    Computes the log PDF of the skew-t-distribution. This uses the Azzalini and Capitanio as described in https://doi.org/10.1111/1467-9868.00391
    Note, this implementation is compiled using Numba and is significantly faster than the one based on the scipy stack.
    This snippet borrows heavily from Gregory Gundersen's multivariate t implementation as described here: https://gregorygundersen.com/blog/2020/01/20/multivariate-t/

    :param x: Numpy array corresponding to locations where we wish to evaluate the pdf
    :param loc: Location parameter; real number
    :param scale: Scale parameter; positive real number
    :param df: Degrees of freedom; positive real number
    :param skew: Skewness parameter; real number
    :return: Values of the PDF at each location x
    """
    dim = 1
    vals, vecs = scale, np.array([1])

    logdet = np.log(scale)
    valsinv = np.array([1. / vals])
    U = vecs * np.sqrt(valsinv)
    dev = x - loc
    maha = np.square(dev * U)

    t = 0.5 * (df + dim)
    A = gammaln(t)
    B = gammaln(0.5 * df)
    C = (dim / 2.) * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -t * np.log(1 + (1. / df) * maha)

    w = np.sqrt(scale)
    J = dev * skew / w

    rad = np.sqrt((dim + df) / (maha + df))
    Fval = tcdf_1d(J * rad, dim + df)
    F = np.log(2) + np.log(Fval)

    return A - B - C - D + E + F


def sampleCvM(dataSort, CDF):
    """
    Computes Cramer von Mises statistic

    :param dataSort: Sorted data values
    :param CDF: CDF function
    :return: Scalar value corresponding to the CvM statistic
    """
    N = len(dataSort)
    CDF_vals = CDF(dataSort)
    empirical_CDF = np.linspace(1, 2 * N - 1, 2) / (2 * N)

    diff_CDFs = CDF_vals - empirical_CDF
    CvM = 1 / (12 * N) + np.sum(diff_CDFs ** 2)
    return CvM


def getIntegrand(loc, scale, df, skew):
    """
    Closure that returns a frozen pdf.

    :param loc: Location parameter; real number
    :param scale: Scale parameter; positive real number
    :param df: Degrees of freedom; positive real number
    :param skew: Skewness parameter; real number
    :return: Function handle for single-argument function that accepts real numbers.
    """

    @njit(cache=True)
    def integrand(y):
        return tspdf_1d(y, loc, scale, df, skew)[0]

    return integrand


def tscdf(x, loc, scale, df, skew):
    """
    Computes the CDF of the t-skew distribution using numerical integration.
    WARNING: To speed up computation this function breaks up the integral into disjoint intervals and combines the resulting values in a cumulative sum.
    This can lead to incorrect values for the CDF due to accumulation of floating point errors.

    :param x: Numpy array corresponding to locations where we wish to evaluate the pdf
    :param loc: Location parameter; real number
    :param scale: Scale parameter; positive real number
    :param df: Degrees of freedom; positive real number
    :param skew: Skewness parameter; real number
    :return: Values of the CDF at each location x
    """
    if isinstance(x, Number):
        x = np.array([x])
    integrand = getIntegrand(loc, scale, df, skew)  # Closure that captures the parameters of the distribution
    nb_integrand = cfunc("float64(float64)")(integrand)  # Convert it cfunc for faster integration

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(x) == 1:
            tscdf_vals = quad(nb_integrand.ctypes, -np.inf, x)[0]
        else:
            sort_inds = np.argsort(x)
            x_sorted = x[sort_inds]

            partial_integrals = np.zeros_like(x)
            partial_integrals[0] = quad(nb_integrand.ctypes, -np.inf, x_sorted[0])[0]

            sliding_windows = np.lib.stride_tricks.sliding_window_view(x_sorted, 2)

            for index, window in enumerate(sliding_windows):
                integral_val, abs_err = quad(nb_integrand.ctypes, window[0], window[1])

                partial_integrals[index + 1] = integral_val

            tscdf_vals = np.cumsum(partial_integrals)

            # Undoes the sort operation
            tscdf_vals = tscdf_vals[np.argsort(sort_inds)]

    return tscdf_vals


def getObjectiveFunction(data, use_loglikelihood=True):
    """
    Closures that captures the data that we wish to fit. Currently only max-likelihood estimation is supported.

    :param data: 1-D numpy array of real numbers.
    :param use_loglikelihood: Default: True. Future implementation will support additional fitting criteria
    :return: Function handle that accepts a 4-vector of real numbers corresponding to the skew-t distribution parameters.
    """

    sorted_data = np.sort(data)
    N = len(data)

    def CvM_fn(theta):
        loc = theta[0]
        scale = theta[1]
        df = theta[2]
        skew = theta[3]

        tscdf_vals = tscdf(sorted_data, loc, scale, df, skew)

        empirical_CDF = np.arange(1, 2 * N, 2) / (2 * N)

        diff_CDFs = tscdf_vals - empirical_CDF
        CvM = 1 / (12 * N) + np.sum(diff_CDFs ** 2)
        return CvM

    @njit
    def loglikelihood(theta):
        loc = theta[0]
        scale = theta[1]
        df = theta[2]
        skew = theta[3]

        llvals = tslogpdf_1d(sorted_data, loc, scale, df, skew)

        return -np.mean(llvals)

    if use_loglikelihood:
        return loglikelihood
    else:
        return CvM_fn


def tskew_moments(loc, scale, df, skew):
    """
    Computes moments of the t-skew distribution according to the procedure described in https://doi.org/10.1111/1467-9868.00391

    :param loc: Location parameter; real number
    :param scale: Scale parameter; positive real number
    :param df: Degrees of freedom; positive real number
    :param skew: Skewness parameter; real number
    :return: Expected value, variance, skewness and kurtosis in that order.
    """
    w = np.sqrt(scale)
    alpha = skew

    omega = scale
    omega_bar = scale / (w * w)

    delta = (alpha * omega_bar) / np.sqrt(1 + alpha * omega_bar * alpha)

    gamma_div = np.exp(gammaln(0.5 * (df - 1)) - gammaln(0.5 * df))
    mu = delta * np.sqrt(df / np.pi) * gamma_div

    expected_value_zero_loc = omega * mu
    expected_value = expected_value_zero_loc + loc

    second_moment = w ** 2 * (df / (df - 2))

    variance = second_moment - expected_value_zero_loc ** 2

    skew_f1 = mu
    skew_f2 = (df * (3 - delta ** 2) / (df - 3) - 3 * df / (df - 2) + 2 * mu ** 2)
    skew_f3 = np.power(df / (df - 2) - mu ** 2, -3 / 2)

    skewness = skew_f1 * skew_f2 * skew_f3

    kurt_f1_s1 = 3 * df ** 2 / ((df - 2) * (df - 4))
    kurt_f1_s2 = -(4 * mu ** 2 * df * (3 - delta ** 2) / (df - 3))
    kurt_f1_s3 = 6 * mu ** 2 * df / (df - 2)
    kurt_f1_s4 = -3 * mu ** 4
    kurt_f1 = kurt_f1_s1 + kurt_f1_s2 + kurt_f1_s3 + kurt_f1_s4

    kurt_f2_s1 = df / (df - 2)
    kurt_f2_s2 = -mu ** 2
    kurt_f2 = np.power(kurt_f2_s1 + kurt_f2_s2, -2)

    kurtosis = kurt_f1 * kurt_f2 - 3

    return expected_value, variance, skewness, kurtosis


def fit_tskew(realization):
    """
    Fits a t skew distribution to a set of real values. By default uses a an iterative solver (Nelder-Meade) with numerical derivatives and a
    max-likelihood fitting criteria. Does not check input data for invalid values.

    :param realization: 1-D numpy array containing data values to be fit.
    :return: location, scale, degrees of freedom, and skewness parameters for the fitted distribution
    """

    # TODO: Adjust these quantities according to the moment calculations
    loc_init = np.mean(realization)
    scale_init = np.var(realization)
    median_init = np.median(realization)
    df_init = 1000
    skew_init = (3 * loc_init - median_init) / np.sqrt(scale_init)

    theta = np.array([loc_init, scale_init, df_init, skew_init])
    obj_fun = getObjectiveFunction(realization, use_loglikelihood=True)
    res = minimize(obj_fun, x0=theta,
                   method='Nelder-Mead',
                   options={'maxiter': 5000,
                            'adaptive': True,
                            'xatol': 1e-6,
                            'fatol': 1e-6})

    return res.x
