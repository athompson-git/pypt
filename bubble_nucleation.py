# Solves for bubble nucleation dynamic params based on bounce action and effective potential

from .constants import *
from .pt_math import *

from .ftpot import *

class BubbleNucleation:
    def __init__(self, veff: VFT):
        self.veff = veff
        self.Tstar = self.get_Tstar()

    def fx_bounce_action(self, x):
        return 1 + 0.25*x * (1 + 2.4/(1-x) + 0.26 / (1-x)**2)

    def bounce_action(self, T):
        # Returns S3/T given the parameters in Veff in thin-wall approx
        return 13.72 * power(T / self.veff.a3(T), 2)  * power(abs(self.veff.a2(T)) / T**2, 3/2) \
            * self.fx_bounce_action(4*self.veff.a4(T) * self.veff.a2(T) / self.veff.a3(T)**2)
        #delta = 8*self.veff.a4(T) * self.veff.a2(T) / self.veff.a3(T)**2
        #beta1 = 8.2938
        #beta2 = -5.5330
        #beta3 = 0.8180
        #return -pi * self.veff.a3(T) * 8*sqrt(2)*power(2 - delta, -2)*sqrt(delta/2) \
        #    * (beta1*delta + beta2*delta**2 + beta3*delta**3) / power(self.veff.a4(T), 1.5) / 81 / T

    def rate(self, T):
        return np.real(T**4 * power(abs(self.bounce_action(T)) / (2*pi), 3/2) * np.exp(-self.bounce_action(T))) \
            * np.heaviside(T - self.veff.T0, 0.0) * np.heaviside(self.veff.Tc - T, 0.0)
        #return T**4 * np.exp(-self.bounce_action(T))

    def hubble2(self, T):
        return pi**2 * GSTAR_SM * T**4 / 90 / M_PL**2
    
    def dP(self, T):
        return self.rate(T) / self.hubble2(T)**2 / T
    
    def get_Tstar(self):
        # Scan from T0 up to find where Gamma / H^4 = 1
        def bubble_rate(T):
            return ((self.rate(T) / self.hubble2(T)**2) - 1.0)
        
        res = fsolve(bubble_rate, [(self.veff.Tc + self.veff.T0)/2])
        return res[0]

    def alpha(self):
        # Latent heat
        Tstar = self.Tstar

        prefactor = 30 / pi**2 / GSTAR_SM / Tstar**4
        phi_plus = self.veff.phi_plus(Tstar)

        deltaV = -self.veff(phi_plus, Tstar)
        dVdT = (self.veff(phi_plus, 1.001*Tstar) - self.veff(phi_plus, 0.999*Tstar))/(1.001 * Tstar - 0.999 * Tstar)
        
        return -prefactor * (abs(deltaV) - abs(Tstar * dVdT))

    def betaByHstar(self):
        Tstar = self.Tstar
        
        # Get the derivative of S3/T
        dSdT = (self.bounce_action(1.001*Tstar) - self.bounce_action(0.999 * Tstar))/(1.001 * Tstar - 0.999 * Tstar)
        return Tstar * dSdT

    def vw(self):
        alpha = self.alpha()
        return (1/sqrt(3) + sqrt(alpha**2 + 2*alpha/3))/(1+alpha)



