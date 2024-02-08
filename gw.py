# Classes and functions for calculating GW spectra from FOPT, PBH, etc.

from pypt.ftpot import VFT
from .constants import *
from .bubble_nucleation import *

class GravitationalWave:
    def __init__(self, alpha=0.1, betaByHstar=1000.0, vw=0.9, Tstar=1000.0):
        self.alpha = alpha
        self.betaByHstar = betaByHstar
        self.vw = vw
        self.Tstar = Tstar
    
    def kappa(self):
        alpha = self.alpha
        vw = self.vw
        cs = 1/sqrt(3)

        vJ = (sqrt(2*alpha/3 + alpha**2) + sqrt(1/3))/(1+alpha)

        kA = power(vw, 6/5) * 6.9*alpha / (1.36 - 0.037*sqrt(alpha) + alpha)
        kB = power(alpha, 2/5) / (0.017 + power(0.997 + alpha, 2/5))
        kC = sqrt(alpha) / (0.135 + sqrt(0.98 + alpha))
        kD = alpha / (0.73 + 0.083*sqrt(alpha) + alpha)

        deltaK = -0.9*log(sqrt(alpha)/(1+sqrt(alpha)))

        if vw < cs:
            return power(cs, 11/5) * kA * kB / ((power(cs, 11/5) - power(vw, 11/5))*kB + vw*kA*power(cs, 6/5))
        elif vw > vJ:
            return (power(vJ - 1, 3)*power(vJ, 5/2)*power(vw,-5/2)*kC*kD)/((power(vJ-1,3) - power(vw-1,3))*power(vJ,5/2)*kC + power(vw-1,3)*kD)
        else:
            return kB + (vw-cs)*deltaK + power((vw-cs)/(vJ-cs), 3) * (kC - kB -(vJ - cs)*deltaK)

    ### Gravitational Wave Spectra Params
    def f_peak(self):
        # returns peak frequency in Hz
        # Tstar in GeV
        return (1.9e-5 / self.vw) * self.betaByHstar * (self.Tstar / 1.0e2) * power(GSTAR_SM/100, 1/6)

    def sw(self, f):
        return power(f/self.f_peak(), 3) * power(7/(4 + 3*power(f/self.f_peak(), 2)), 7/2)

    def omega(self, f):
        # return the gravitational wave energy budget
        kappa = self.kappa()
        alpha = self.alpha
        return 8.5e-6 * self.sw(f) * power(100/GSTAR_SM, 1/3) * power(kappa*alpha / (1+alpha), 2) * self.vw / self.betaByHstar

