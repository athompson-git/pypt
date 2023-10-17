# Solves for false vacuum fraction and 

import warnings

from .constants import *
from .cosmology_functions import *
from .ftpot import *

from scipy.integrate import quad


def a_ratio_rad(ti, tj):
    return power(tj/ti, 1/2)


def scale_factor_int2(ti, t):
    return power(quad(a_ratio_rad, ti, t, args=(ti,))[0], 2)


def fv_filling_frac(t, tstar, beta):
    return 1.238 * np.exp(beta * (t - tstar))


def fv_nuc_rate(t, tc, tstar, GammaStar, beta, vw, n_samples=10000):
    # tc --> (t4, t3, t2, t1) --> t
    ts_unordered = np.random.uniform(tc, t, (n_samples, 4))

    # sort the t's from smallest to highest: (t4, t3, t2, t1)
    for i, ti in enumerate(ts_unordered):
        ts_unordered[i] = np.sort(ti)

    prefactor = 32*pi**4 * vw**9 * fv_filling_frac(t, tstar, beta)
    mc_volume = (t - tc)**4 / n_samples

    print("Prefactor = ", prefactor)

    integrand_list = [scale_factor_int2(ti[0], t) * GammaStar*exp(beta*(ti[0]-tstar)) * a_ratio_rad(t, ti[0]) \
                      * scale_factor_int2(ti[1], t) * GammaStar*exp(beta*(ti[1]-tstar)) * a_ratio_rad(t, ti[1]) \
                        * scale_factor_int2(ti[2], t) * GammaStar*exp(beta*(ti[2]-tstar)) * a_ratio_rad(t, ti[2]) 
                        * scale_factor_int2(ti[3], t) * GammaStar*exp(beta*(ti[3]-tstar)) * a_ratio_rad(t, ti[3]) for ti in ts_unordered]

    print(np.sum(integrand_list))
    return mc_volume * prefactor * np.sum(integrand_list)


def fv_nuc_rate_high_beta(t, tstar, beta, vw):
    Istar = 1.238
    return power(Istar*beta, 4) * exp(4*beta*(t-tstar)) * exp(-Istar*exp(beta*(t-tstar))) / (192 * vw**3)



def get_gamma_star(beta, vw):
    return 1.238 * beta**4 / (8*pi*vw**3)





