# Math imports
# 
# Copyright (c) 2025 Adrian Thompson via MIT License

from .constants import *


import numpy as np
from numpy import sqrt, log, exp, log10, power, sin, cos, tan, heaviside

from scipy.optimize import fsolve, fmin, minimize, shgo, brute
from scipy.signal import argrelmin
from scipy.integrate import quad

#import mpmath as mp
#mp.prec = 10
#mp.dps = 10

import cosmoTransitions as cosmo
from cosmoTransitions.tunneling1D import SingleFieldInstanton, PotentialError

import pkg_resources

###### assumes rad domination

def temp_to_time(T, gstar=GSTAR_SM):
    return sqrt(90/8/pi**3/gstar_sm(T)) * M_PL / T**2

def time_to_temp(t, gstar=GSTAR_SM):
    return sqrt(sqrt(90/pi**3/gstar/8) * M_PL / t)

def hubble2_rad(T, gstar=GSTAR_SM):
    return 8*pi*pi**2 * gstar * T**4 / 90 / M_PL**2

def a_ratio_rad(ti, tj):
    # returns the ratio a(tj) / a(ti)
    return power(tj/ti, 1/2)

def scale_factor_int2_rad(ti, t):
    return power(2*ti - 2*sqrt(ti*t), 2)



# General
gstar_dat_path = pkg_resources.resource_filename(__name__, "data/gstar_sm.txt")
gstar_data = np.genfromtxt(gstar_dat_path)

def gstar_sm(T):
    return np.interp(T, gstar_data[:,0], gstar_data[:,1], left=GSTAR_SM_LE, right=GSTAR_SM)


def a_ratio(ti, tj):
    pass
