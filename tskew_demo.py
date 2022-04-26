import numpy as np
from scipy.stats import norm, expon, gamma, rayleigh, beta
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except:
    pass
import matplotlib.pyplot as plt; plt.ion()

from tskew.tskew import tspdf_1d, tscdf, fit_tskew



def gen_data_poly(N=2000):
    # Normal
    nmean = 1
    nloc = 2
    normal_data = norm.rvs(loc=1, scale=2, size=N)
    normal_dist = norm(loc=1, scale=2)

    # Exponential
    lam = 2
    exponential_data = expon.rvs(scale=1 / lam, size=N)
    exponential_dist = expon(scale=1 / lam)

    # Gamma
    a = 1
    b = 2
    gamma_data = gamma.rvs(a, scale=2, size=N)
    gamma_dist = gamma(a, scale=2)

    # Rayleigh
    rayleigh_data = rayleigh.rvs(loc=nmean, scale=nloc, size=N)
    rayleigh_dist = rayleigh(loc=nmean, scale=nloc)

    # Beta
    beta_a = 3
    beta_b = 4
    beta_data = beta.rvs(a=beta_a, b=beta_b, scale=2, size=N)
    beta_dist = beta(a=beta_a, b=beta_b, scale=2)

    data_dict = {
        'Normal': ([nmean, nloc], normal_data, normal_dist),
        'Exponential': ([lam], exponential_data, exponential_dist),
        'Gamma': ([a, b], gamma_data, gamma_dist),
        'Rayleigh': ([nmean, nloc], rayleigh_data, rayleigh_dist),
        'Beta': ([beta_a, beta_b], beta_data, beta_dist),
    }
    return data_dict


if __name__ == "__main__":
    # Visualize skew t
    xvals = np.linspace(-6, 6, 5_000)
    loc = -1
    scale = 0.5
    df = 5
    skew_param = 10

    cdf = tscdf(xvals, loc, scale, df, skew_param)
    pdf = tspdf_1d(xvals, loc, scale, df, skew_param)

    plt.figure()
    plt.plot(xvals, cdf, linewidth=5, label='CDF')
    plt.plot(xvals, pdf, linewidth=5, label='PDF')
    plt.legend()
    plt.title('Skew t PDF and CDF')

    data = gen_data_poly()
    for dist_name, dist_data in data.items():
        realization = dist_data[1]
        distribution = dist_data[2]

        loc_est, scale_est, df_est, skew_param_est = fit_tskew(realization)

        extent = np.max(realization) - np.min(realization)
        domain_vals = np.linspace(np.min(realization) - 0.1 * extent, np.max(realization) + 0.1 * extent, 5_000)

        plt.figure()
        plt.rc('axes', titlesize=20)
        est_pdf = tspdf_1d(domain_vals, loc_est, scale_est, df_est, skew_param_est)
        plt.hist(realization, bins=20, density=True, color='green', alpha=.5)
        plt.plot(domain_vals, distribution.pdf(domain_vals), label=f'True: {dist_name}', linewidth=5, )
        plt.plot(domain_vals, est_pdf, linestyle='--', label='Estimated skew t', linewidth=5, )
        plt.title(f'{dist_name}-distributed data')
        plt.legend(fontsize=12)
        plt.show()
