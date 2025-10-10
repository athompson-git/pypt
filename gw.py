# Classes and functions for calculating GW spectra from FOPT, PBH, etc.
# 
# Copyright (c) 2025 Adrian Thompson via MIT License

from pypt.ftpot import VFT
from .constants import *
from .bubble_nucleation import *

class GravitationalWave:
    def __init__(self, alpha=0.1, betaByHstar=1000.0, vw=0.9, Tstar=1000.0, gstar_D=4.5):
        self.alpha = alpha
        self.betaByHstar = betaByHstar
        self.vw = vw
        self.Tstar = Tstar
        self.gstar_D = gstar_D
    
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
    def hstar_param(self):
        return 1.65e-5 * (self.Tstar/100.0) * power((self.gstar_D + gstar_sm(self.Tstar))/100, 1/6)
    
    def f_peak_sw(self):
        # returns peak frequency in Hz
        # Tstar in GeV
        return (1.9e-5 / self.vw) * self.betaByHstar * (self.Tstar / 1.0e2) * power(GSTAR_SM/100, 1/6)
    
    def f_peak_col(self):
        return (0.62/(1.8 - 0.1*self.vw + self.vw**2)) * self.betaByHstar * self.hstar_param()
    
    def f_peak_turb(self):
        return 1.64*self.betaByHstar*self.hstar_param()/self.vw
    
    def shock_formation_scale(self):
        # from 2003.07360
        # approximate RMS perfect fluid velocity 
        Uf = sqrt(0.75 * self.kappa() * self.alpha / (1 + self.alpha))
        cs = 1/sqrt(3)

        return power(8*np.pi, 1/3) * max(self.vw, cs) / Uf / self.betaByHstar

    def sw_sw(self, f):
        # spectral function for the sound wave piece
        return power(f/self.f_peak_sw(), 3) * power(7/(4 + 3*power(f/self.f_peak_sw(), 2)), 7/2)
    
    def sw_col(self, f):
        # spectral function for the collisional piece
        hstar = 1.65e-5 * (self.Tstar/100.0) * power((self.gstar_D + gstar_sm(self.Tstar))/100, 1/6)
        f_col = (0.62/(1.8 - 0.1*self.vw + self.vw**2)) * self.betaByHstar * hstar
        return ((0.11*self.vw**3)/(0.42 + self.vw**2)) * ((3.8*power(f/f_col, 2.8))/(1 + 2.8*power(f/f_col, 3.8)))

    def sw_turb(self, f):
        # spectral function for the turbulence piece
        hstar = 1.65e-5 * (self.Tstar/100.0) * power((self.gstar_D + gstar_sm(self.Tstar))/100, 1/6)
        f_turb = 1.64*self.betaByHstar*hstar/self.vw
        return power(f/f_turb, 3)/(power(1 + f/f_turb, 11/3)*(1 + 8*pi*f/hstar))

    def omega_sw(self, f):
        # return the gravitational wave energy budget from sound waves
        kappa = self.kappa()
        alpha = self.alpha
        return 8.5e-6 * self.shock_formation_scale() * self.sw_sw(f) * power(100/gstar_sm(self.Tstar), 1/3) \
            * power(kappa*alpha / (1+alpha), 2) * self.vw / self.betaByHstar

    def omega_turb(self, f):
        # return the gravitational wave energy budget from turbulence
        kappa = 0.05*self.kappa()
        return 3.35e-4 * (1 - self.shock_formation_scale()) * self.vw * (1/self.betaByHstar) \
            * power(100/(self.gstar_D + gstar_sm(self.Tstar)), 1/3) \
            * power(kappa*self.alpha / (1+self.alpha), 3/2) * self.sw_col(f)

    def omega_col(self, f):
        # return the gravitational wave energy budget from collisions
        kappa = np.clip(1 - self.kappa() - 0.05*self.kappa(), a_min=0.0, a_max=1.0)
        return 1.67e-5 * power(1/self.betaByHstar, 2) * power(100/(self.gstar_D + gstar_sm(self.Tstar)), 1/3) \
            * power(kappa*self.alpha / (1+self.alpha), 2) * self.sw_col(f)
    
    def omega(self, f):
        return self.omega_sw(f) + self.omega_col(f) + self.omega_turb(f)
    
    def f_peak(self):
        peak_fs = [self.f_peak_sw(), self.f_peak_col(), self.f_peak_turb()]
        peaks = [self.omega_sw(self.f_peak_sw()), self.omega_col(self.f_peak_col()), self.omega_turb(self.f_peak_turb())]
        max_peak_idx = np.argmax(peaks)
        return peak_fs[max_peak_idx]

