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
        self.verbose = verbose

        if Tstar is not None:
            self.Tstar = Tstar
            self.T_test = Tstar
        else:
            self.get_Tstar(verbose)
        
        self.deltaT = 0.0001*self.Tstar
        self.setup_sfi()
    
    def veff_fixed_T(self, phi):
        return self.veff(phi, self.T_test)
    
    def setup_sfi(self):
        # Construct a SingleFieldInstanton class at Tstar and Tstar+dT
        if self.verbose:
            print("---- Computing dS/dT...")
        
        def veff_at_T(phi):
            return self.veff(phi, T=self.Tstar)
    
        def veff_at_deltaT(phi):
            return self.veff(phi, T=self.Tstar+self.deltaT)
        
        self.phi_plus = max(self.veff.get_mins(T=self.Tstar))
        self.phi_plus_dT = max(self.veff.get_mins(T=self.Tstar+self.deltaT))
    
        sfi_T = SingleFieldInstanton(phi_absMin=self.phi_plus, phi_metaMin=0.0, V=veff_at_T)
        sfi_dT = SingleFieldInstanton(phi_absMin=self.phi_plus_dT, phi_metaMin=0.0, V=veff_at_deltaT)

        profile_T = sfi_T.findProfile(phitol=1e-7*(self.Tstar/self.Tc))
        profile_dT = sfi_dT.findProfile(phitol=1e-7*(self.Tstar/self.Tc))
        
        self.SE_T = sfi_T.findAction(profile_T)
        self.SE_T_plus_dT = sfi_dT.findAction(profile_dT)

    def get_bounce_action_ct(self):
        print("--- Getting bounce action...")
        print("--- --- Getting minima...")
        mins = self.veff.get_mins(self.T_test)
        
        if len(mins) < 1:
            return None
        
        if not np.any(mins > 0.0):
            return None
        
        phi_plus = max(mins)
        veff_at_min = self.veff(phi_plus, self.T_test)

        if veff_at_min > 0.0:
            return None
        
        try:
            print("--- --- trying SingleFieldInstanton...")
            sfi = SingleFieldInstanton(phi_absMin=phi_plus, phi_metaMin=0.0, V=self.veff_fixed_T)
            # check the bounce action
            profile = sfi.findProfile(xtol=1e-8, phitol=1e-8,
                    thinCutoff=.001, npoints=1000, rmin=1e-6, rmax=1e6,
                    max_interior_pts=None)
            SE = sfi.findAction(profile)
            return SE
        
        except PotentialError as e:
            # Check the specific exception message
            if "Barrier height is not positive" in str(e):
                if self.verbose:
                    print("Barrier height not positive error!")
                return 0.0
            else:
                return None
        
        except ValueError as e:
            if "f(a) and f(b) must have different signs" in str(e):
                if self.verbose:
                    print("f(a) and f(b) must have different signs error!")
                return 0.0

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
        T_0 = 0.001
        try_counter = 0
        while se_1 is None:
            if verbose:
                print("SE returned None, looking for lower T")
            self.T_test = 0.5*self.T_test
            se_1 = self.get_bounce_action_ct()

            try_counter += 1
            if try_counter > 10:
                print("Searched too low below initial guess of Tc, stopping")
                self.Tstar = None
                return
        
        # Find upper bound
        if self.verbose:
            print("Starting with se_1 = {}".format(se_1))
        while se_1 / self.T_test < 140.0:
            if verbose:
                print("---- Searching for upper bound, found SE={} at T={}".format(se_1, self.T_test))

            se_1 = self.get_bounce_action_ct()
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
                self.T_test = 2*self.T_test


        # Now we have found that SE/T = 140 lies between T_0 and T_test,
        # perform a binary search for T_star
        T_low, T_high = T_0, self.T_test
        T_tol = 0.001*self.Tc
        if verbose:
            print("beginning binary search between T_low = {} and T_high = {}".format(T_low, T_high))
        while abs(se_1 / self.T_test - 140.0) > 20.0 and abs(T_low - T_high) > T_tol:
            self.T_test = (T_high + T_low)/2
            se_1 = self.get_bounce_action_ct()
            if se_1 is None:
                T_high = self.T_test
                se_1 = 1000.0*self.T_test
                continue

            if verbose:
                print("----- Checking T={}, found SE/T = {}".format(self.T_test, se_1 / self.T_test))
            
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

        deltaV = -self.veff(self.phi_plus, self.Tstar)
        dVdT = (self.veff(self.phi_plus_dT, self.Tstar+self.deltaT) - self.veff(self.phi_plus, self.Tstar))/(self.deltaT)
        return prefactor * (deltaV + self.Tstar * dVdT / 4)

    def betaByHstar(self):
        # Get the derivative of S3/T
        dSdT = abs(self.SE_T_plus_dT - self.SE_T) / self.deltaT
        return self.Tstar * dSdT

    def vw(self):
        alpha = self.alpha()
        deltaV = -self.veff(self.phi_plus, self.Tstar)

        # Jouget velocity
        vJ = (sqrt(2*alpha/3 + alpha**2) + sqrt(1/3))/(1+alpha)

        # radiation density
        rho_r = pi**2 * GSTAR_SM * self.Tstar**4 / 30

        #  previous approx: return (1/sqrt(3) + sqrt(alpha**2 + 2*alpha/3))/(1+alpha)

        if sqrt(deltaV / (alpha*rho_r)) < vJ:
            return sqrt(deltaV / (alpha*rho_r))
        else:
            return 1.0

