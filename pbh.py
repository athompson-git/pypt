# Solves for false vacuum fraction and 

import warnings

from .constants import *
from .cosmology_functions import *
from .ftpot import *

from scipy.integrate import quad



class FVFilling:
    def __init__(self, Tstar, Tc, beta, vw, n_samples=100):
        self.Tstar = Tstar
        self.tstar = temp_to_time(Tstar)
        self.Tc = Tc
        self.tc = temp_to_time(Tc)
        self.beta = beta
        self.vw = vw
        self.n_samples = n_samples
        self.integrand_list = []
        self.ts_rnd_at_tstar = np.random.uniform(self.tc, self.tstar, (self.n_samples, 4))
        for i, ti in enumerate(self.ts_rnd_at_tstar):
            self.ts_rnd_at_tstar[i] = np.sort(ti)

    def scale_factor_int2(self, ti, t):
        return power(quad(a_ratio, ti, t, args=(ti,))[0], 2)


    # Eq 18 for I(t)
    def fv_filling_frac(self, t):
        return 1.238 * np.exp(self.beta * (t - self.tstar))

    # Eq 3
    def f_fv(self, t):
        return mp.exp(-self.fv_filling_frac(t))
    
    # from Eq 18
    def get_gamma_star(self):
        return 1.238 * self.beta**4 / (8*pi*self.vw**3)


    # Eq 12 in Lu-Kawana-Xie
    def fv_nuc_rate(self, t):
        self.integrand_list = []
        # tc --> (t4, t3, t2, t1) --> t
        ts_unordered = np.random.uniform(self.tc, t, (self.n_samples, 4))

        # sort the t's from smallest to highest: (t4, t3, t2, t1)
        for i, ti in enumerate(ts_unordered):
            ts_unordered[i] = np.sort(ti)

        prefactor = 32*pi**4 * self.vw**9 * self.f_fv(t)
        mc_volume = (t - self.tc)**4 / self.n_samples

        GammaStar = self.get_gamma_star()
        
        self.integrand_list = [power(GammaStar, 4) * mp.exp(self.beta*((ti[0]-self.tstar) + (ti[1]-self.tstar) + (ti[2]-self.tstar) + (ti[3]-self.tstar))) \
                        * scale_factor_int2_rad(ti[0], t) * a_ratio_rad(t, ti[0]) \
                        * scale_factor_int2_rad(ti[1], t) * a_ratio_rad(t, ti[1]) \
                            * scale_factor_int2_rad(ti[2], t) * a_ratio_rad(t, ti[2]) 
                            * scale_factor_int2_rad(ti[3], t) * a_ratio_rad(t, ti[3]) for ti in ts_unordered]

        return mc_volume * prefactor * np.sum(self.integrand_list)
    
    def fv_nuc_rate_tstar(self):
        self.integrand_list = []
        # tc --> (t4, t3, t2, t1) --> t

        prefactor = 32*pi**4 * self.vw**9 * self.f_fv(self.tstar)
        mc_volume = (self.tstar - self.tc)**4 / self.n_samples
        GammaStar = self.get_gamma_star()

        self.integrand_list = [power(GammaStar, 4) * mp.exp(self.beta*((ti[0]-self.tstar) + (ti[1]-self.tstar) + (ti[2]-self.tstar) + (ti[3]-self.tstar))) \
                        * scale_factor_int2_rad(ti[0], self.tstar) * a_ratio_rad(self.tstar, ti[0]) \
                        * scale_factor_int2_rad(ti[1], self.tstar) * a_ratio_rad(self.tstar, ti[1]) \
                            * scale_factor_int2_rad(ti[2], self.tstar) * a_ratio_rad(self.tstar, ti[2]) 
                            * scale_factor_int2_rad(ti[3], self.tstar) * a_ratio_rad(self.tstar, ti[3]) for ti in self.ts_rnd_at_tstar]
        return mc_volume * prefactor * np.sum(self.integrand_list)


    def fv_nuc_rate_high_beta(self, t):
        Istar = 1.238
        return power(Istar*self.beta, 4) * exp(4*self.beta*(t-self.tstar)) * exp(-Istar*exp(self.beta*(t-self.tstar))) / (192 * self.vw**3)


    # Eq 15
    def dndR(self, R):
        tp = self.tstar + R/self.vw
        return (1/self.vw) * (1-self.f_fv(tp)) * self.fv_nuc_rate_tstar() * power(a_ratio_rad(self.tstar, tp), 4)


    def dndM(self, Mpbh, gstar_BSM=5):
        Hstar = sqrt(hubble2_rad(self.Tstar))

        # use Eq 3.5 to convert M into R
        R = power(Mpbh * power(8 * GSTAR_SM / 7 / gstar_BSM, 3/2) * 2 * power(Hstar, -5) / M_PL**2, 1/6)
        
        # use 3.5 to get jacobian
        dMdR = 3 * power(7 * gstar_BSM / 8 / GSTAR_SM, 3/2) * M_PL**2 * power(R * Hstar, 5)
        dndM = np.heaviside(R - 1/Hstar,0.0) * self.dndR(R) / dMdR

        if dndM < 1e120:
            return dndM

        return 0.0
    
    def pbh_mass_fraction(self, Mpbh, gstar_BSM=5):
        # use 3.8 to construct df/dM
        t0 = temp_to_time(T0_SM)

        s_dark = (2*pi**2 / 45) * (GSTAR_SM + gstar_BSM) * self.Tstar**3  # dark entropy
        Mprime = (Mpbh**3 + 3*1.895e-3 * M_PL**4 * t0)

        dndM = self.dndM(Mprime)
        return (Mpbh/Mprime)**3 * (1/OMEGA_DM) * (8*pi/(3*M_PL**2 * HUBBLE**2)) \
            * (S0_SM / s_dark) * (Mprime * dndM)


