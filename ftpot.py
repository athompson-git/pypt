# Class for storing a finite temp effective potential

from .constants import *
from .pt_math import *


PT_CONST_AB = 16 * pi**2 * exp(1.5 - 2*GAMMA_EULER)
PT_CONST_AF = pi**2 * exp(1.5 - 2*GAMMA_EULER)
PT_CONST_EXPAB = exp(log(PT_CONST_AB) - 1.5)
PT_CONST_EXPAF = exp(log(PT_CONST_AB) - 1.5)


def jf(m, T):
    pass




class VFT:
    def __init__(self):
        self.T0 = None
        self.Tc = None

    def a2(self, T):
        return 1.0

    def a3(self, T):
        return 1.0

    def a4(self, T):
        return 1.0

    def phi_plus(self, T):
        return np.real(-3*self.a3(T) + sqrt(9*self.a3(T)**2 - 32*self.a4(T)*self.a2(T)))/(8*self.a4(T))

    def get_T0(self, T_range=[1.0, 100.0]):
        res = fsolve(self.a2, T_range)
        return abs(res[0])
    
    def get_Tc(self):
        res = fsolve(self.Veff0Min, [self.T0])
        return abs(res[0])
    
    def Veff0Min(self, T):
        return self.a2(T) * self.phi_plus(T)**2 + self.a3(T) * self.phi_plus(T)**3 + self.a4(T) * self.phi_plus(T)**4

    def __call__(self, phi, T):
        pass





class VEffSM(VFT):
    def __init__(self):
        self.Tc = 150.0
        
        self.D0 = (2*M_W**2 + M_Z**2 + 2*M_T**2) / 8 / VEV_H**2
        self.D1 = (2 * M_W**3 + M_Z**3) / (4*pi*VEV_H**3)
        self.D2 = 3 * (2*M_W**4 + M_Z**4 - 4*M_T**4) / (64*pi**2 * VEV_H**4)
        self.T0 = sqrt((M_H**2 - 8*self.D2*VEV_H**2)/(4*self.D0))


    def lamT(self, T):
        return 0.5*power(M_H/VEV_H, 2) - 3*(2*M_W**4 * log(M_W**2 / PT_CONST_EXPAB / T**2) \
            + M_Z**4 * log(M_Z**2 / PT_CONST_EXPAB / T**2) \
            - 4*M_T**4 * log(M_T**2 / PT_CONST_EXPAF / T**2))/(16*pi**2 * VEV_H**4)

    def __call__(self, phi, T):
        return np.real(self.D0*(self.Tc**2 - T**2)*phi**2 - self.D1*T*phi**3 + self.lamT(T)*phi**4)




class VEffMarfatia(VFT):
    def __init__(self, gchi=1.0, mchi=1.0, mu=100.0, lam=0.1, c=0.1, Lambda=1000.0):
        super().__init__()
        self.gchi = gchi
        self.mchi = mchi
        self.mu = mu
        self.lam = lam
        self.c = c
        self.Lambda = Lambda
    
    def set_params(self, gchi=1.0, mchi=1.0, mu=100.0, lam=0.1, c=0.1, Lambda=1000.0):
        self.gchi = gchi
        self.mchi = mchi
        self.mu = mu
        self.lam = lam
        self.c = c
        self.Lambda = Lambda
    
    def a2(self, T):
        beta = 1/T
        return 1/(1536*pi**2*beta**4*self.mu**8)*(-36*self.gchi**4*beta**4*self.mu**8+64*pi**2*beta**4*self.lam*self.mu**8-9*beta**4*self.lam**2*self.mu**8-3*self.c**4*pi*(beta**2*self.mu**2)**(3/2)-12*pi*self.lam**2*self.mu**4*(beta**2*self.mu**2)**(3/2)-24*self.gchi**4*beta**4*self.mu**8*log((self.mchi**2*beta**2)/PT_CONST_AF)+24*self.gchi**4*beta**4*self.mu**8*log(self.mchi**2/self.Lambda**2)-6*beta**4*self.lam**2*self.mu**8*log((beta**2*self.mu**2)/PT_CONST_AB)+6*beta**4*self.lam**2*self.mu**8*log(self.mu**2/self.Lambda**2))

    def a3(self, T):
        beta = 1/T
        return -(1/(192*pi**2*beta**4*self.mu**6))*(18*self.gchi**3*self.mchi*beta**4*self.mu**6-32*self.c*pi**2*beta**4*self.mu**6-c**3*pi*(beta**2*self.mu**2)**(3/2)+12*self.gchi**3*self.mchi*beta**4*self.mu**6*log((self.mchi**2*beta**2)/PT_CONST_AF)-12*self.gchi**3*self.mchi*beta**4*self.mu**6*log(self.mchi**2/self.Lambda**2))

    def a4(self, T):
        beta = 1/T
        return (1/(384*pi**2*beta**4*self.mu**4))*(-8*self.gchi**2*pi**2*beta**2*self.mu**4-9*self.c**2*beta**4*self.mu**4-54*self.gchi**2*self.mchi**2*beta**4*self.mu**4+8*pi**2*beta**2*self.lam*self.mu**4+192*pi**2*beta**4*self.mu**6-9*beta**4*self.lam*self.mu**6-12*c**2*pi*(beta**2*self.mu**2)**(3/2)-24*pi*self.lam*self.mu**2*(beta**2*self.mu**2)**(3/2)-36*self.gchi**2*self.mchi**2*beta**4*self.mu**4*((self.mchi**2*beta**2)/PT_CONST_AF)+36*self.gchi**2*self.mchi**2*beta**4*self.mu**4*(self.mchi**2/self.Lambda**2)-6*c**2*beta**4*self.mu**4*((beta**2*self.mu**2)/PT_CONST_AB)-6*beta**4*self.lam*self.mu**6*((beta**2*self.mu**2)/PT_CONST_AB)+6*c**2*beta**4*self.mu**4*(self.mu**2/self.Lambda**2)+6*beta**4*self.lam*self.mu**6*(self.mu**2/self.Lambda**2))
        
    def __call__(self, phi, T):
        return np.real(self.a2(1/T)*phi**2 + self.a3(1/T)*phi**3 + self.a4(1/T)*phi**4)




class VEffMarfatia2(VFT):
    def __init__(self, a=0.1, lam=0.061, c=0.249, d=0.596, b=75.0**4):
        super().__init__()
        self.a = a
        self.b = b
        self.lam = lam
        self.c = c
        self.d = d
        self.T0 = self.get_T0()
        self.Tc = self.get_Tc()
    
    def phi_plus0(self, T0):
        return (3*self.c + sqrt(9*self.c**2 + 8*self.lam*self.d*T0**2))/(2*self.lam)

    def get_T0(self):
        def root_func(T0):
            return self.b - 0.5 * self.phi_plus0(T0)**2 * (self.d*T0**2 + 0.5*self.c*self.phi_plus0(T0))
        
        res = fsolve(root_func, [1.0])
        return res[0]
    
    def a2(self, T):
        return self.d * (T**2 - self.T0**2)
    
    def a3(self, T):
        return -(self.a*T + self.c)

    def a4(self, T):
        return 0.25*self.lam
    
    def set_params(self, a=0.1, lam=0.061, c=0.249, d=0.596):
        self.a = a
        self.lam = lam
        self.c = c
        self.d = d
        self.T0 = self.get_T0()
        self.Tc = self.get_Tc()

    def __call__(self, phi, T):
        return np.real(self.d * (T**2 - self.T0**2)*phi**2 - (self.a*T + self.c)*phi**3 + 0.25*self.lam*phi**4)