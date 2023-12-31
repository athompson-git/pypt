# Math imports


from .constants import *


import numpy as np
from numpy import sqrt, log, exp, log10, power, sin, cos, tan

from scipy.optimize import fsolve, fmin, minimize, shgo, brute
from scipy.signal import argrelmin
from scipy.integrate import quad

#import mpmath as mp
#mp.prec = 10
#mp.dps = 10


###### assumes rad domination

def temp_to_time(T):
    return sqrt(45/16/pi**3) * M_PL / sqrt(GSTAR_SM) / T**2

def time_to_temp(t):
    return sqrt(sqrt(45/16/pi**3) * M_PL / sqrt(GSTAR_SM) / t)


def hubble2_rad(T):
    return pi**2 * GSTAR_SM * T**4 / 90 / M_PL**2


def a_ratio_rad(ti, tj):
    return power(tj/ti, 1/2)


def scale_factor_int2_rad(ti, t):
    return power(-2*ti + 2*t*sqrt(ti/t), 2)



# General

def a_ratio(ti, tj):
    pass
