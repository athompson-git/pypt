# Solves for false vacuum fraction and BH mass spectrum 

import warnings

from .constants import *
from .cosmology_functions import *
from .ftpot import *

from scipy.integrate import quad

import gmpy2 as mp
from gmpy2 import mpz

import pkg_resources


# Import thermal integral data
bh_dat_path = pkg_resources.resource_filename(__name__, "data/bh_mass_evolution_function.txt")
bh_mass_data = np.genfromtxt(bh_dat_path)

def bh_mass_loss_function(t, m0, gstar=7):
    # t is the time since BH formation in seconds
    # m0 is the initial BH mass in grams
    t_end = 4e-4 * power(m0 / 1e8, 3) * (108 / gstar)
    return m0 * power(10, np.interp(np.log10(t/t_end), np.log10(bh_mass_data[:,0]), np.log10(bh_mass_data[:,1]),
                                    right=-np.inf, left=0.0))

def bh_mass_loss_function_analytic(t, m0, gstar=7):
    t_end = 4e-4 * power(m0 / 1e8, 3) * (108 / gstar)
    return m0 * power(1 - t/t_end, 1/3)
        
        



# class for calculating the false vacuum filling fraction and PBH spectra
class FVFilling:
    def __init__(self, Tstar, Tc, betaByHstar, vw, gstar_D=4.5, n_samples=100):
        self.gstar_D = gstar_D
        self.Tstar = Tstar
        self.tstar = temp_to_time(Tstar, gstar=gstar_D + gstar_sm(self.Tstar))
        self.Tc = Tc
        self.tc = temp_to_time(Tc, gstar=gstar_D + gstar_sm(self.Tstar))
        self.Hstar = sqrt(hubble2_rad(self.Tstar, gstar=gstar_D + gstar_sm(self.Tstar)))
        self.betaByHstar = betaByHstar
        self.beta = betaByHstar*self.Hstar
        self.vw = vw
        self.n_samples = n_samples
        self.integrand_list = []
        self.ts_rnd_at_tstar = np.random.uniform(self.tc, self.tstar, (self.n_samples, 4))
        for i, ti in enumerate(self.ts_rnd_at_tstar):
            self.ts_rnd_at_tstar[i] = np.sort(ti)
    
    def set_params(self, Tstar, Tc, betaByHstar, vw):
        self.Tstar = Tstar
        self.tstar = temp_to_time(Tstar, gstar=self.gstar_D + gstar_sm(Tstar))
        self.Tc = Tc
        self.tc = temp_to_time(Tc, gstar=self.gstar_D + gstar_sm(Tstar))
        self.Hstar = sqrt(hubble2_rad(Tstar, gstar=self.gstar_D + gstar_sm(self.Tstar)))
        self.betaByHstar = betaByHstar
        self.beta = betaByHstar*self.Hstar
        self.vw = vw
        self.integrand_list = []
        self.ts_rnd_at_tstar = np.random.uniform(self.tc, self.tstar, (self.n_samples, 4))
        for i, ti in enumerate(self.ts_rnd_at_tstar):
            self.ts_rnd_at_tstar[i] = np.sort(ti)

    def scale_factor_int2(self, ti, t):
        return power(quad(a_ratio, ti, t, args=(ti,))[0], 2)

    # Eq 18 for I(t)
    def fv_filling_frac(self, t):
        return 1.238 * mp.exp(self.beta * (t - self.tstar))

    def f_fv_R0(self, R0):
            return mp.exp(-1.238 * mp.exp(self.beta * R0 / self.vw))

    # Eq 3
    def f_fv(self, t):
        return mp.exp(-self.fv_filling_frac(t))
    
    # from Eq 18
    def get_gamma_star(self):
        return 1.238 * np.power(self.beta, 4) / (8*pi*self.vw**3)


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
        
        self.integrand_list = [mp.mul(power(GammaStar, 4), mp.exp(self.beta*((ti[0]-self.tstar) + (ti[1]-self.tstar) + (ti[2]-self.tstar) + (ti[3]-self.tstar)))) \
                        * scale_factor_int2_rad(ti[0], t) * a_ratio_rad(t, ti[0]) \
                        * scale_factor_int2_rad(ti[1], t) * a_ratio_rad(t, ti[1]) \
                            * scale_factor_int2_rad(ti[2], t) * a_ratio_rad(t, ti[2]) 
                            * scale_factor_int2_rad(ti[3], t) * a_ratio_rad(t, ti[3]) for ti in ts_unordered]

        return mp.mul(mc_volume, mp.mul(prefactor, np.sum(self.integrand_list)))
    
    def fv_nuc_rate_tstar(self):
        self.integrand_list = []
        # tc --> (t4, t3, t2, t1) --> t

        prefactor = 32*pi**4 * self.vw**9 * self.f_fv(self.tstar)
        mc_volume = (self.tstar - self.tc)**4 / self.n_samples
        GammaStar = self.get_gamma_star()

        self.integrand_list = [mp.mpz(mp.exp(self.beta*((ti[0]-self.tstar) + (ti[1]-self.tstar) \
                                                        + (ti[2]-self.tstar) + (ti[3]-self.tstar))) \
                        * scale_factor_int2_rad(ti[0], self.tstar) * a_ratio_rad(self.tstar, ti[0]) \
                        * scale_factor_int2_rad(ti[1], self.tstar) * a_ratio_rad(self.tstar, ti[1]) \
                        * scale_factor_int2_rad(ti[2], self.tstar) * a_ratio_rad(self.tstar, ti[2]) \
                        * scale_factor_int2_rad(ti[3], self.tstar) * a_ratio_rad(self.tstar, ti[3]))
                            for ti in self.ts_rnd_at_tstar]

        integral_result = mp.mpfr(sum(self.integrand_list))
        gamma4 = mp.mpfr(power(GammaStar, 4))

        return mp.mul(prefactor, mp.mul(mp.mul(integral_result, gamma4), mc_volume))


    def fv_nuc_rate_high_beta(self, t):
        Istar = 1.238
        exp1 = mp.exp(4*mp.mpz(self.beta*(t-self.tstar)))
        exp2 = Istar*mp.exp(mp.mul(self.beta,(t-self.tstar)))
        exp3 = mp.exp(mp.log(exp1) - exp2)  #mp.mul(exp1,mp.exp(-exp2))

        # log(exp1 * exp(-exp2)) = log(exp1) + log(exp(-exp2)) = log(exp1) - exp2 --> exp(log(exp1) - exp2)

        beta_factor = power(Istar*self.beta, 4)
        filling_frac = mp.mul(mp.mul(beta_factor, exp3),  1/(192 * self.vw**3))

        return filling_frac


    # Eq 15
    def dndR(self, R):
        tp = self.tstar + R/self.vw
        if self.betaByHstar > 5.0:
            fv_frac = self.fv_nuc_rate_high_beta(tp) #
        else:
            fv_frac = self.fv_nuc_rate(tp)
        return mp.mul((1/self.vw), mp.mul(mp.mul((1-self.f_fv(tp)) , fv_frac), power(a_ratio_rad(self.tstar, tp), 4)))
    
    def dndR2(self, R):
        return mp.mul((power(1.238 * self.beta, 4)/(192 * self.vw**3)), \
                      mp.mul((1 - self.f_fv_R0(R)), mp.exp(4*self.beta*R/self.vw -1.238*mp.exp(self.beta*R/self.vw))))


    def dndM(self, Mpbh):
        Hstar = sqrt(hubble2_rad(self.Tstar))

        # use Eq 3.5 to convert M into R
        r = power(Mpbh * power(8 * (gstar_sm(self.Tstar)+self.gstar_D) / 7 / self.gstar_D, 3/2) * 2 * power(Hstar, -5) / M_PL**2, 1/6)
        
        # use 3.5 to get jacobian
        dMdR = 6 * Mpbh/r # 3 * power(7 * gstar_BSM / 8 / GSTAR_SM, 3/2) * M_PL**2 * power(r * Hstar, 5)
        dndM = mp.mul(self.dndR2(r), 1/dMdR) #* np.heaviside(r - 1/Hstar,0.0)

        return dndM

    def dfdM(self, Mpbh, gstar_BSM=5):
        # use 3.8 to construct df/dM
        # takes in mass in grams
        # returns df/dM in g^-1
        t0 = TIME_TODAY_SEC #temp_to_time(T0_SM)  # time in GeV^-1

        s_dark = (2*pi**2 / 45) * (gstar_sm(self.Tstar) + gstar_BSM) * self.Tstar**3  # dark entropy
        
        #Mprime = (power(GEV_PER_G*Mpbh,3) + 3*1.895e-3 * M_PL**4 * t0)  # in natural units
        # The mass today after evolving the BH from t* to t0
        Mprime = bh_mass_loss_function_analytic(t0-self.tstar*HBAR, Mpbh, gstar=7)  # interpolate results from BlackHawk

        dndM = self.dndM(Mpbh*GEV_PER_G)
        #dfdM = (1/OMEGA_DM) * power(Mpbh/Mprime, 3) * (8*pi/(3*M_PL**2 * HUBBLE**2)) * (S0_SM / s_dark) * (Mpbh * dndM)
        dfdM = (1/OMEGA_DM) * (8*pi/(3*M_PL**2 * HUBBLE**2)) * (S0_SM / s_dark) * (Mprime * dndM)
        return power(GEV_PER_G, 2) * dfdM

    def mass_peak(self, M_min=1e14, M_max=1e60):
        # returns the approximate peak mass in grams
        #return 5.2e15 * power(1e7 / self.Tstar, 2)
        M_range = np.logspace(np.log10(M_min), np.log10(M_max), 500)
        df_dMs = np.array([np.nan_to_num(float(self.dfdM(M))) for M in M_range])
        peak_loc = np.argmax(df_dMs)
        peak_mass = M_range[peak_loc]

        return peak_mass
        
    def f_pbh(self, gstar_BSM=5):
        m_peak = self.mass_peak()
        integrand = lambda lnM: np.exp(lnM) * self.dfdM(np.exp(lnM))
        integral, error = quad(integrand, np.log(m_peak)-2, np.log(m_peak)+2)
        return np.nan_to_num(integral)

    def pbh_temp(self, Mpbh):
        return M_PL**2 / (8*pi*Mpbh)


