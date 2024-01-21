# Solves for bubble nucleation dynamic params based on bounce action and effective potential

import warnings

from .constants import *
from .cosmology_functions import *

from .ftpot import *




class BubbleNucleation:
    def __init__(self, veff: VFT, Tstar=None, verbose=False):
        self.veff = veff
        self.Tc = veff.Tc
        self.T_test = veff.Tc  # test T for finding Tstar

        if Tstar is not None:
            self.Tstar = Tstar
        else:
            self.get_Tstar(verbose)
        
        self.setup_sfi()
    
    def veff_fixed_T(self, phi):
        return self.veff(phi, self.T_test)
    
    def setup_sfi(self):
        # Construct a SingleFieldInstanton class at Tstar and Tstar+dT
        self.deltaT = self.Tstar*0.001
        def veff_at_T(phi):
            return self.veff(phi, T=self.Tstar)
    
        def veff_at_deltaT(phi):
            return self.veff(phi, T=self.Tstar+self.deltaT)
        
        self.phi_plus = max(self.veff.get_mins(T=self.Tstar))
    
        sfi_T = SingleFieldInstanton(phi_absMin=self.phi_plus, phi_metaMin=0.0, V=veff_at_T)
        sfi_dT = SingleFieldInstanton(phi_absMin=self.phi_plus, phi_metaMin=0.0, V=veff_at_deltaT)

        profile_T = sfi_T.findProfile(phitol=1e-7)
        profile_dT = sfi_dT.findProfile(phitol=1e-7)
        
        self.SE_T = sfi_T.findAction(profile_T)
        self.SE_T_plus_dT = sfi_dT.findAction(profile_dT)

    def get_bounce_action_ct(self):
        mins = self.veff.get_mins(self.T_test)
        if len(mins) < 1:
            return None
        
        phi_plus = max(mins)
        veff_at_min = self.veff(phi_plus, self.T_test)
        if veff_at_min > 0.0:
            return None
        
        sfi = SingleFieldInstanton(phi_absMin=phi_plus, phi_metaMin=0.0, V=self.veff_fixed_T)

        # check the bounce action
        profile = sfi.findProfile(phitol=1e-8)
        SE = sfi.findAction(profile)
        return SE

    def bounce_action(self, T):
        # Returns S3/T given the parameters in Veff in thin-wall approx
        # TODO: integrate CosmoTransition
        delta = 8*self.veff.a4(T) * self.veff.a2(T) / self.veff.a3(T)**2
        beta1 = 8.2938
        beta2 = -5.5330
        beta3 = 0.8180
        return np.real(-pi * self.veff.a3(T) * 8*sqrt(2)*power(2 - delta, -2)*sqrt(delta/2) \
            * (beta1*delta + beta2*delta**2 + beta3*delta**3) / power(self.veff.a4(T), 1.5) / 81 / T)

    def rate(self, T):
        return np.real(T**4 * power(abs(self.bounce_action(T)) / (2*pi), 3/2) * np.exp(-abs(self.bounce_action(T))))

    def hubble2(self, T):
        return pi**2 * GSTAR_SM * T**4 / 90 / M_PL**2  + abs((self.veff.Veff0Min(T)*3/M_PL**2))
    
    def dP(self, T):
        return self.rate(T) / self.hubble2(T)**2 / T
    
    def get_Tstar(self, verbose=False):
        # start from T_critical
        se_1 = self.get_bounce_action_ct()
        while se_1 is None:
            self.T_test = 0.95*self.T_test
            se_1 = self.get_bounce_action_ct()
        
        if se_1 / self.T_test < 140.0:
            if verbose:
                print("Initial SE/T < 140, searching higher")
            T_0 = self.Tc
            self.T_test = 1.05*self.Tc
        else:
            if verbose:
                print("Initial SE/T > 140, skipping to binary search")
            T_0 = self.Tc / 10
            self.T_test = self.Tc

        # Find upper bound
        while se_1 / self.T_test < 140.0:
            se_1 = self.get_bounce_action_ct()

            if verbose:
                print("---- Searching for upper bound, found SE={} at T={}".format(se_1, self.T_test))

            if se_1 is None:
                # Go lower, halfway between T_0 and T_test
                self.T_test = (self.T_test + T_0) / 2
                if verbose:
                    print("---- Found bad Euclidean action, searching lower...")
                
                se_1 = self.T_test * 1000.0
                continue
            
            if se_1 / self.T_test < 140.0:
                # Move up to higher T in 5% increments
                T_0 = self.T_test
                self.T_test = 1.05*self.T_test

        # Now we have found that SE/T = 140 lies between T_0 and T_test,
        # perform a binary search for T_star
        T_low, T_high = T_0, self.T_test
        if verbose:
            print("beggining binary search between T_low = {} and T_high = {}".format(T_low, T_high))
        while abs(se_1 / self.T_test - 140.0) > 20.0:
            self.T_test = (T_high + T_low)/2
            if verbose:
                print("----- Checking T={}".format(self.T_test))
            
            se_1 = self.get_bounce_action_ct()
            if se_1 / self.T_test > 140.0:
                T_high = self.T_test
            else:
                T_low = self.T_test
        
        if verbose:
            print("Found T* at {} for SE/T = {}".format(self.T_test, se_1 / self.T_test))
        self.Tstar = self.T_test

    def alpha(self):
        # Latent heat
        prefactor = 30 / pi**2 / (GSTAR_SM) / self.Tstar**4
        phi_plus = self.phi_plus

        deltaV = -self.veff(phi_plus, self.Tstar)
        dVdT = (self.veff(phi_plus, self.Tstar+self.deltaT) - self.veff(phi_plus, self.Tstar))/(self.deltaT)
        return prefactor * (deltaV + self.Tstar * dVdT / 4)

    def betaByHstar(self):
        # Get the derivative of S3/T
        dSdT = (self.SE_T_plus_dT - self.SE_T) / self.deltaT
        return self.Tstar * dSdT

    def vw(self):
        alpha = self.alpha()
        return (1/sqrt(3) + sqrt(alpha**2 + 2*alpha/3))/(1+alpha)
    
    def kappa(self):
        alpha = self.alpha()
        vw = self.vw()
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
        return (1.9e-5 / self.vw()) * self.betaByHstar() * (self.Tstar / 1.0e5) * power(GSTAR_SM/100, 1/6)

    def sw(self, f):
        return power(f/self.f_peak(), 3) * power(7/(4 + 3*power(f/self.f_peak(), 2)), 7/2)

    def omega(self, f):
        # return the gravitational wave energy budget
        kappa = self.kappa()
        alpha = self.alpha()
        return 8.5e-6 * self.sw(f) * power(100/GSTAR_SM, 1/3) * power(kappa*alpha / (1+alpha), 2) * self.vw() / self.betaByHstar()


