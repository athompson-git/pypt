# Solves for bubble nucleation dynamic params based on bounce action and effective potential

import warnings

from .constants import *
from .pt_math import *

from .ftpot import *

class BubbleNucleation:
    def __init__(self, veff: VFT):
        self.veff = veff
        self.Tstar = self.get_Tstar()

        # init params

    def fx_bounce_action(self, x):
        return 1 + 0.25*x * (1 + 2.4/(1-x) + 0.26 / (1-x)**2)

    def bounce_action(self, T):
        # Returns S3/T given the parameters in Veff in thin-wall approx
        #return 13.72 * power(T / self.veff.a3(T), 2)  * power(abs(self.veff.a2(T)) / T**2, 3/2) \
        #    * self.fx_bounce_action(4*self.veff.a4(T) * self.veff.a2(T) / self.veff.a3(T)**2)
        delta = 8*self.veff.a4(T) * self.veff.a2(T) / self.veff.a3(T)**2
        beta1 = 8.2938
        beta2 = -5.5330
        beta3 = 0.8180
        return np.real(-pi * self.veff.a3(T) * 8*sqrt(2)*power(2 - delta, -2)*sqrt(delta/2) \
            * (beta1*delta + beta2*delta**2 + beta3*delta**3) / power(self.veff.a4(T), 1.5) / 81 / T)

    def rate(self, T):
        return np.real(T**4 * power(abs(self.bounce_action(T)) / (2*pi), 3/2) * np.exp(-abs(self.bounce_action(T))))
        #return T**4 * np.exp(-self.bounce_action(T))

    def hubble2(self, T):
        return pi**2 * GSTAR_SM * T**4 / 90 / M_PL**2  + abs((self.veff.Veff0Min(T)*3/M_PL**2))
    
    def dP(self, T):
        return self.rate(T) / self.hubble2(T)**2 / T
    
    def get_Tstar(self):
        # Scan from T0 up to find where Gamma / H^4 = 1
        def bubble_rate(T):
            clips =  np.heaviside(T - self.veff.T0, 0.0) * np.heaviside(self.veff.Tc - T, 0.0)
            return (clips*(self.rate(T) / self.hubble2(T)**2) - 1.0)
        
        res = fsolve(bubble_rate, [(self.veff.Tc + self.veff.T0)/2])
        return res[0]

    def alpha(self):
        # Latent heat
        Tstar = self.Tstar
        deltaT = 0.001*Tstar

        prefactor = 30 / pi**2 / (GSTAR_SM) / Tstar**4
        phi_plus = self.veff.phi_plus(Tstar)

        deltaV = -self.veff(phi_plus, Tstar)
        dVdT = (self.veff(phi_plus, Tstar+deltaT) - self.veff(phi_plus, Tstar))/(deltaT)
        return prefactor * (deltaV + Tstar * dVdT / 4)

    def betaByHstar(self):
        Tstar = self.Tstar
        deltaT = 0.00000001*Tstar
        
        # Get the derivative of S3/T
        dSdT = (self.bounce_action(Tstar+deltaT) - self.bounce_action(Tstar))/(deltaT)
        return Tstar * dSdT

    def vw(self):
        alpha = self.alpha()
        return (1/sqrt(3) + sqrt(alpha**2 + 2*alpha/3))/(1+alpha)
    
    def kappa(self):
        alpha = self.alpha()
        vw = self.vw()

        if (alpha > 10.0) or (alpha < 1.0e-3):
            warnings.warn('alpha = {} and exceeds bounds for kappa (kinetic energy coefficient) precision.'.format(alpha), DeprecationWarning)
        
        # TODO: check if subsonic deflagration


        # TODO: check if detonation
    
    ### Gravitational Wave Spectra Params
    def f_peak(self):
        # returns peak frequency in Hz
        return (1.9e-5 / self.vw()) * self.betaByHstar() * (self.Tstar / 1.0e5) * power(GSTAR_SM/100, 1/6)

    def sw(self, f):
        return power(f/self.f_peak(), 3) * power(7/(4 + 3*(f/self.f_peak())), 7/2)

    def omega(self):
        pass


