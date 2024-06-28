# Solves for bubble nucleation dynamic params based on bounce action and effective potential

import warnings

from .constants import *
from .cosmology_functions import *

from .ftpot import *

import gmpy2 as mp



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
        
        self.deltaT = 0.0000001*self.Tstar
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




class BubbleNucleationQuartic:
    """
    Bubble nucleation class for the generic quartic potential
    uses an analytic approximation of the bounce action
    """
    def __init__(self, veff: VEffGeneric, Tstar=None, gstar_D=4.5, verbose=False):
        self.veff = veff
        self.Tc = veff.Tc
        self.T_test = veff.Tc
        self.verbose = verbose
        self.a = veff.a
        self.c = veff.c
        self.d = veff.d
        self.lam = veff.lam
        self.T0 = veff.T0
        self.vev = veff.vev
        self.gstar_D = gstar_D

        if Tstar is not None:
            self.Tstar = Tstar
        else:
            self.Tstar = self.get_Tstar_from_rate()
        
        try:
            if verbose:
                print("Found T* = {} for S3/T = {}".format(self.Tstar, self.bounce_action(self.Tstar)))

            self.deltaT = 0.0001*self.Tstar

            self.phi_plus = max(self.veff.get_mins(T=self.Tstar))
            self.phi_plus_dT = max(self.veff.get_mins(T=self.Tstar+self.deltaT))
            
            self.SE_T = self.bounce_action(self.Tstar)
            self.SE_T_plus_dT = self.bounce_action(self.Tstar+self.deltaT)
        except:
            raise Exception("Unable to find bounce action solutions or T*!")

    def veff_fixed_T(self, phi):
        return self.veff(phi, self.T_test)

    def bounce_action(self, T):
        # Returns S3/T given the parameters in Veff in thin-wall approx
        delta = 8*self.veff.a4(T) * self.veff.a2(T) / self.veff.a3(T)**2
        beta1 = 8.2938
        beta2 = -5.5330
        beta3 = 0.8180
        return np.clip((-pi * self.veff.a3(T) * 8*sqrt(2)*power(2 - delta, -2) \
                        *sqrt(abs(delta)/2) \
            * (beta1*delta + beta2*delta**2 + beta3*delta**3) \
                / power(self.veff.a4(T), 1.5) / 81 / T), a_min=0.0, a_max=np.inf)

    def rate(self, T):
        return np.real(T**4 * power(abs(self.bounce_action(T)) / (2*pi), 3/2) * np.exp(-abs(self.bounce_action(T))))
    
    def get_Tstar(self):
        # check SE/T close to T=Tc
        if self.verbose:
            print("SE/T = {} at T=Tc".format(self.bounce_action(self.Tc)))
        
        T_grid = np.linspace(self.T0, 1.0*self.Tc, 10000000)
        s3ByTs = self.bounce_action(T_grid)

        mask = (s3ByTs>80.0)*(s3ByTs < 200.0)

        s3ByT_within_140 = s3ByTs[mask]
        T_grid_within_140 = T_grid[mask]
        if len(s3ByT_within_140) == 0:
            return None
        
        s3ByT_within_140 = np.asarray(s3ByT_within_140)
        closest_idx = (np.abs(s3ByT_within_140 - 140.0)).argmin()
        if abs(s3ByT_within_140[closest_idx] - 140.0) > 10.0:
            return None
        return T_grid_within_140[closest_idx]
    
    def get_Tstar_from_rate(self):
        # check SE/T close to T=Tc
        T_grid = np.linspace(self.T0, 1.0*self.Tc, 10000)
        GammaByHstar = np.nan_to_num([self.rate(T)/power(hubble2_rad(T,gstar=gstar_sm(T)+self.gstar_D),2) for T in T_grid])
        star_id = np.argmin(abs(GammaByHstar - 1.0))
        T_star_2 = T_grid[star_id]

        # save critical rate error
        self.rate_star = GammaByHstar[star_id]

        return T_star_2
    
    def dVdT(self, phi, T):
        return 2*self.d*T*phi**2 - self.a*phi**2
    
    def dSbyTdT(self, T):
        beta1 = 8.2938
        beta2 = -5.5330
        beta3 = 0.8180
        numerator = (256*sqrt(2)*self.d*pi*sqrt((self.d*(T-self.T0)*(T+self.T0)*self.lam)/(self.c+self.a*T)**2)\
         * ((self.c+self.a*T)**6 * (3*self.c*T+self.a*(T**2+2*self.T0**2))*beta1+self.d*(self.c+self.a*T)**4 \
           * (T-self.T0)*(T+self.T0)*(-self.a*T**2 * (beta1-2*beta2)+2*self.a*self.T0**2 \
                                      *(beta1+4*beta2)+self.c*T*(beta1+10*beta2)) \
            *self.lam-2*self.d**2 * (self.c+self.a*T)**2 * (T-self.T0)**2 * (T+self.T0)**2 \
                * (T*(self.c+self.a*T)*beta2 - 2*(7*self.c*T+self.a*(T**2+6*self.T0**2))*beta3)*self.lam**2-4*self.d**3 \
                    * (T**2-self.T0**2)**3 * (3*self.c*T+self.a*T**2 + 2*self.a*self.T0**2)*beta3*self.lam**3))
        denomenator = 81*power(self.c+self.a*T, 8) * sqrt(self.lam)*(2+(2*self.d * (-T**2 + self.T0**2)*self.lam)/(self.c+self.a*T)**2)**3
        return abs(numerator/denomenator)

    def alpha(self):
        # Latent heat
        prefactor = 30 / pi**2 / (GSTAR_SM) / self.Tstar**4

        deltaV = -self.veff(self.phi_plus, self.Tstar)
        dVdT = (self.veff(self.phi_plus_dT, self.Tstar+self.deltaT) - self.veff(self.phi_plus, self.Tstar))/(self.deltaT)
        #dVdT = self.dVdT(self.phi_plus, self.Tstar)

        return prefactor * (deltaV + self.Tstar * dVdT / 4)

    def betaByHstar(self):
        # Get the derivative of S3/T
        #dSdT = abs(self.SE_T_plus_dT - self.SE_T) / self.deltaT
        dSdT = self.dSbyTdT(self.Tstar)
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