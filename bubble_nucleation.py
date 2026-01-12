# Solves for bubble nucleation dynamic params based on bounce action and effective potential
# 
# Copyright (c) 2025 Adrian Thompson via MIT License

import warnings

from .constants import *
from .cosmology_functions import *
from .ftpot import *
from .vac_rad_cosmic_history import CosmicHistoryVacuumRadiation

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
    def __init__(self, veff: VEffGeneric, gstar_D=4.5, verbose=False,
                 assume_rad_dom=True, use_fks_action=False) -> None:
        self.veff = veff
        self.Tc = veff.Tc
        self.tc = temp_to_time(veff.Tc)
        self.T_test = veff.Tc
        self.verbose = verbose
        self.a = veff.a
        self.c = veff.c
        self.d = veff.d
        self.lam = veff.lam
        self.T0sq = veff.T0sq
        self.vev = veff.vev
        self.gstar_D = gstar_D

        self.assume_rad_dom = assume_rad_dom
        self.use_fks_action = use_fks_action
        self.Tperc = None
        self.tperc = None

        self.teq = None

        if self.Tperc is None:
            self.Tstar = self.get_Tstar()
            self.Tperc = self.get_Tperc()
            self.tperc = temp_to_time(self.Tperc)

        try:
            if verbose:
                print("Found T* = {} for S3/T = {}".format(self.Tperc, self.bounce_action(self.Tperc)))

            self.deltaT = 0.000001*self.Tperc

            self.phi_plus = self.veff.get_vev(self.Tperc) # max(self.veff.get_mins(T=self.Tperc))
            self.phi_plus_dT = self.veff.get_vev(self.Tperc+self.deltaT) # max(self.veff.get_mins(T=self.Tperc+self.deltaT))
            
            self.SE_T = self.bounce_action(self.Tperc)
            self.SE_T_plus_dT = self.bounce_action(self.Tperc+self.deltaT)
        except:
            raise Exception("Unable to find bounce action solutions or T*!")

    # FUNCTIONS FOR THE ACTION AND NUCLEATION RATE
    def veff_fixed_T(self, phi) -> float:
        return self.veff(phi, self.T_test)
    
    def bounce_action(self, T) -> float:
        # Returns S3/T given the parameters in Veff in thin-wall approx
        # see 2304.10084
        delta = 8*self.veff.a4(T) * self.veff.a2(T) / self.veff.a3(T)**2
        beta1 = 8.2938
        beta2 = -5.5330
        beta3 = 0.8180
        return np.clip((-pi * self.veff.a3(T) * 8*sqrt(2)*power(2 - delta, -2) \
                        *sqrt(abs(delta)/2) \
            * (beta1*delta + beta2*delta**2 + beta3*delta**3) \
                / power(self.veff.a4(T), 1.5) / 81 / T), a_min=0.0, a_max=np.inf)
    
    def kappa_func(self, T) -> float:
        return self.veff.lam * 2 * self.veff.d * (T**2 - self.veff.T0sq) \
            / power(3 * (self.veff.a*T + self.veff.c), 2)

    def b3bar(self, kappa) -> float:
        return (16/243) * (1 - 38.23*(kappa - 2/9) + 115.26*(kappa - 2/9)**2 \
                           + 58.07*sqrt(kappa)*(kappa - 2/9)**2 \
                           + 229.07*kappa*(kappa - 2/9)**2)

    def bounce_action_fks(self, T) -> float:
        prefactor = power(2 * self.veff.d * (T**2 - self.veff.T0sq), 3/2) \
            / power(3 * (self.veff.a*T + self.veff.c), 2)

        kappa = self.kappa_func(T)
        kappa_gtr_zero = kappa > 0

        kappa_c = 0.52696

        return (prefactor*(2*pi/(3*(kappa - kappa_c)**2)) \
                * self.b3bar(kappa) / T) * kappa_gtr_zero \
                + (1 - kappa_gtr_zero) * (prefactor*(27*pi/2) \
                                        * (1 + np.exp(-power(abs(kappa), -0.5))) \
                                        / (1 + abs(kappa)/kappa_c) / T)
    
    def rate(self, T) -> float:
        if self.use_fks_action:
            return np.real(T**4 * power(abs(self.bounce_action_fks(T)) \
                                         / (2*pi), 3/2) \
                                * np.exp(-abs(self.bounce_action_fks(T))))
        return np.real(T**4 * power(abs(self.bounce_action(T)) / (2*pi), 3/2) \
                       * np.exp(-abs(self.bounce_action(T))))

    # FUNCTIONS FOR THE FALSE VACUUM FRACTION AND EVOLUTION
    def R_bubble(self, tprime) -> float:
        # Returns the Radius of the vacuum bubble at time tprime
        # Integrates from self.T_perc
        delta_t = (self.tperc - tprime) / 100
        t_vals = np.arange(tprime, self.tperc + delta_t, delta_t)
        
        return np.sum([delta_t * (self.vw()*a_ratio_rad(t, self.tperc)) \
                    for t in t_vals])
    
    def R_bubble_temperature(self, Tprime, T, n_samples=100) -> float:
        # returns radius of FV bubble nucleating at Tprime at later temp T
        T_vals = np.linspace(T, Tprime, n_samples)  # prefer inexpensive sampling, will get integrated over again late
        dT = T_vals[1] - T_vals[0]
        gamma = 1.0

        hubble_vals = np.array([sqrt(self.hubble_rate_sq(T_)) for T_ in T_vals])
        integrands = self.vw() * power(self.Tc / T_vals, -1/gamma) / (T_vals * hubble_vals * gamma) * dT
        return np.sum(integrands)
    
    def fv_exponent(self, T, n_samples=100) -> float:
        # returns FV exponent function I(T) for computing percolation with
        # exp(-I(T)) = 0.7
        Tprime_vals = np.linspace(T, self.Tc, n_samples)
        dT = Tprime_vals[1] - Tprime_vals[0]
        gamma = 1.0

        r_bubble = np.array([self.R_bubble_temperature(Tprime, T) for Tprime in Tprime_vals])
        rates = self.rate(Tprime_vals)
        hubble_vals = np.array([sqrt(self.hubble_rate_sq(Tprime)) for Tprime in Tprime_vals])

        integrands = (4*pi/3) * rates * power(r_bubble, 3) \
            * power(self.Tc / T, 3/gamma) / (Tprime_vals * gamma * hubble_vals) * dT
        return np.sum(integrands)

    def get_Tperc(self, T_min=None):
        # binary search on fv_exponent between Tc/1000 and Tc
        # adjust minimal temperature as needed
        T_low = 1e-1 * self.Tc
        if T_min is not None:
            T_low = T_min
        T_high = self.Tc

        # check bounds
        p_fv_high = np.nan_to_num(np.exp(-self.fv_exponent(T_high)))
        p_fv_low = np.nan_to_num(np.exp(-self.fv_exponent(T_low)))
        if p_fv_high < 0.7:
            raise Exception("PercolationError!")
        if p_fv_low > 0.7:
            raise Exception("PercolationError!")

        halving_number = 0
        while(halving_number < 20):
            halving_number += 1
            T_trial = (T_high + T_low)/2
            p_fv = np.nan_to_num(np.exp(-self.fv_exponent(T_trial)))
            if p_fv > 0.7:
                T_high = T_trial
            else:
                T_low = T_trial
            
            if abs(p_fv - 0.7) < 0.1:
                return T_trial

        return T_trial

    def p_surv_false_vacuum(self, r_fv) -> float:
        # Uses FKS calculation for survival probability of patches with radius r_fv
        # assume vwall = 1 for the below vacuum fraction
        # Integrand: rate * scale factor^3 * volume factor
        def integrand(tprime):
            return (-4*pi/3) * self.rate(time_to_temp(tprime, gstar=self.gstar_D + gstar_sm(time_to_temp(tprime)))) \
                * np.power(a_ratio_rad(self.tperc, tprime) * (self.R_bubble(tprime) + r_fv), 3)
        res = quad(integrand, self.tc, self.tperc)[0]
        return np.exp(res)

    def hubble_rate_sq(self, T) -> float:
        h2_rad = hubble2_rad(T, gstar=gstar_sm(T)+self.gstar_D)
        phic = self.veff.get_vev(T)
        h2_vac = (1/3/M_PL**2) * (-self.veff(phic, T))

        return h2_rad + h2_vac

    def get_Tstar(self) -> float:
        # check SE/T close to T=Tc
        if self.verbose:
            print("SE/T = {} at T=Tc".format(self.bounce_action(self.Tc)))
        
        # Bounded between T0 and Tc
        T_grid = np.linspace(np.sqrt(abs(self.T0sq)), self.Tc, 100000)
        GammaByHstar = np.nan_to_num([self.rate(T)/power(self.hubble_rate_sq(T),2) for T in T_grid])
        star_id = np.argmin(abs(GammaByHstar - 1.0))
        T_star_candidate = T_grid[star_id]

        # save critical rate error
        self.rate_star = GammaByHstar[star_id]

        return T_star_candidate
    
    def dVdT(self, phi, T) -> float:
        # first derivative of the potential with respect to temperature
        return 2*self.d*T*phi**2 - self.a*phi**3
    
    def d2VdT2(self, phi) -> float:
        # second derivative of the potential with respect to temperature
        return 2*self.d*phi**2
    
    def dRhoRdT(self, T) -> float:
        # first derivative of radiation densiy w.r.t. temperature
        # get the first derivative of g*
        dgdT = (gstar_sm(T + self.deltaT) - gstar_sm(T))/(self.deltaT)

        return (np.pi**2 / 30) * (dgdT * T**4 + 4 * gstar_sm(T) * T**3)

    def d2RhoRdT2(self, T) -> float:
        # second derivative of the radiation density w.r.t. temperature
        return (self.dRhoRdT(T + self.deltaT) - self.dRhoRdT(T)) / self.deltaT
    
    def dtdT(self, T) -> float:
        # gets the temperature-time relation
        # TODO(AT): need to replace vev(T=0) with vev(T)

        return -3*np.sqrt(self.hubble_rate_sq(T)) * (-self.dVdT(self.vev) + self.dRhoRdT(T)/3) \
            / (-self.d2VdT2(self.vev) + self.d2RhoRdT2(T)/3)

    def alpha(self) -> float:
        # Latent heat
        prefactor = 30 / pi**2 / (GSTAR_SM) / self.Tperc**4

        deltaV = -self.veff(self.phi_plus, self.Tperc)
        dVdT = self.dVdT(self.phi_plus, self.Tperc)

        return prefactor * (deltaV + self.Tperc * dVdT / 4)

    def betaByHstar(self) -> float:
        # Get the derivative of S3/T
        dSdT = abs(self.SE_T_plus_dT - self.SE_T) / self.deltaT
        
        return self.Tperc * dSdT

    def vw(self) -> float:
        alpha = self.alpha()
        deltaV = -self.veff(self.phi_plus, self.Tperc)

        # Jouget velocity
        vJ = (sqrt(2*alpha/3 + alpha**2) + sqrt(1/3))/(1+alpha)

        # radiation density
        rho_r = pi**2 * GSTAR_SM * self.Tperc**4 / 30

        #  previous approx: return (1/sqrt(3) + sqrt(alpha**2 + 2*alpha/3))/(1+alpha)

        if sqrt(deltaV / (alpha*rho_r)) < vJ:
            return sqrt(deltaV / (alpha*rho_r))
        else:
            return 1.0




class BounceActionEspinoza:
    """
    Calculates the Euclidean action using Espinoza's method [1805.03680]
    Guesses phi0 = phi_- (take phi0 equal to the VEV at temperature T)
    """
    def __init__(self, veff: VFT, T_test):
        self.veff = veff
        self.dphi = 0.000001

        # get maximum
        test_phis = np.linspace(0.0, max(self.veff.get_mins(T_test)), 1000)
        test_v = self.veff(test_phis, T_test)
        max_id = np.argmax(test_v)
        self.phiT = test_phis[max_id]
    
    def vt1(self, phi, phi0, T):
        return self.veff(phi, T) * (phi / phi0)

    def vt2(self, phi, phi0, T):
        return self.vt1(phi, phi0, T) + (phi / (4*phi0**2))*(3*phi0*self.dV_dphi(phi0, T) - 4*self.veff(phi0, T))*(phi - phi0)

    def vt3(self, phi, phi0, T):
        return self.vt2(phi, phi0, T) + (phi / (4*phi0**3))*(3*phi0*self.dV_dphi(phi0, T) - 8*self.veff(phi0, T))*(phi - phi0)**2

    def vt4(self, phi, phi0, T):
        phiT = self.phiT

        phi0T = phi0 - phiT
        c = 4*power(phiT*phi0, 2)*(phi0**2 - 2*phi0T*phiT)
        Vt3T = self.vt3(phiT, phi0, T)
        VT = self.veff(phiT, T)
        dVt3Tdphi = self.dVt3_dphi(phiT, phi0, T)
        d2Vt3Tdphi2 = self.d2Vt3_dphi2(phiT, phi0, T)

        a0T = -6*(VT - Vt3T)*(phi0**2 - 6*phi0T*phiT) - 8*phiT*(phi0T - phiT)*phi0T*dVt3Tdphi \
                + 3*power(phiT*phi0T, 2)*d2Vt3Tdphi2
        
        Ut3T = 4*(dVt3Tdphi)**2 + 6*(VT-Vt3T)*d2Vt3Tdphi2
        a4 = (1/c)*(a0T - sqrt(a0T**2 - c*Ut3T))

        return self.vt3(phi, phi0, T) + a4*power(phi*(phi-phi0), 2)

    def dV_dphi(self, phi, T):
        return (self.veff(phi+self.dphi, T) - self.veff(phi, T))/self.dphi
    
    def dVt3_dphi(self, phi, phi0, T):
        return (self.vt3(phi+self.dphi, phi0, T) - self.vt3(phi, phi0, T))/self.dphi
    
    def d2Vt3_dphi2(self, phi, phi0, T):
        return (self.dVt3_dphi(phi+self.dphi, phi0, T) - self.dVt3_dphi(phi, phi0, T))/self.dphi
    
    def dVt_dphi(self, phi, phi0, T):
        return (self.vt4(phi+self.dphi, phi0, T) - self.vt4(phi, phi0, T))/self.dphi

    def EuclideanActionVt(self, T, phi0=None):
        # Guess phi0 equal to the minumum phi_- or VEV value
        if phi0 is None:
            phi0 = max(self.veff.get_mins(T))

        # make lambda for integrand and use quad
        integrand = lambda phi: power(self.veff(phi, T) - self.vt4(phi, phi0, T), 2) / power(self.dVt_dphi(phi, phi0, T), 3)

        # TODO: iterate on phi0 assumption to minimize SE
        return quad(integrand, 0.0, phi0)[0]
    
    def EuclideanActionVtIntegrand(self, phi, T, phi0=None):
        # Guess phi0 equal to the minumum phi_- or VEV value
        if phi0 is None:
            phi0 = max(self.veff.get_mins(T))

        # make lambda for integrand and use quad
        return power(self.veff(phi, T) - self.vt3(phi, phi0, T), 2) / power(self.dVt_dphi(phi, phi0, T), 3)