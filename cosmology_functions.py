# Math imports


from .constants import *


import numpy as np
from numpy import sqrt, log, exp, log10, power, sin, cos, tan

from scipy.optimize import fsolve



def temp_to_time(T):
    return sqrt(45/16/pi**3) * M_PL / sqrt(GSTAR_SM) / T**2

def time_to_temp(t):
    return sqrt(sqrt(45/16/pi**3) * M_PL / sqrt(GSTAR_SM) / t)


def hubble2_rad(T):
    return pi**2 * GSTAR_SM * T**4 / 90 / M_PL**2