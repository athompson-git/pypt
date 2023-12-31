# Class for storing a finite temp effective potential

from .constants import *
from .cosmology_functions import *

import pkg_resources


PT_CONST_AB = 16 * pi**2 * exp(1.5 - 2*GAMMA_EULER)
PT_CONST_AF = pi**2 * exp(1.5 - 2*GAMMA_EULER)
PT_CONST_EXPAB = exp(log(PT_CONST_AB) - 1.5)
PT_CONST_EXPAF = exp(log(PT_CONST_AB) - 1.5)


def thermal_boson_integrand(x, m2beta2):
    if m2beta2 < 0.0:
        if x**2 < abs(m2beta2):
            return x**2 * log(1 - cos(sqrt(x**2 - m2beta2)))
        else:
            return x**2 * log(1 - exp(-sqrt(x**2 + m2beta2)))
    return x**2 * log(1 - exp(-sqrt(x**2 + m2beta2)))

def thermal_fermion_integrand(x, m2beta2):
    if m2beta2 < 0.0:
        if x**2 < abs(m2beta2):
            return x**2 * log(1 + cos(sqrt(x**2 - m2beta2)))
        else:
            return x**2 * log(1 + exp(-sqrt(x**2 + m2beta2)))
    return x**2 * log(1 + exp(-sqrt(x**2 + m2beta2)))

def JF(m2beta2):
    # takes in m^2(phi)*beta^2 as argument
    return quad(thermal_fermion_integrand, 0, np.inf, args=(m2beta2,))[0]


def JB(m2beta2):
    # takes in m^2(phi)*beta^2 as argument
    return quad(thermal_boson_integrand, 0, np.inf, args=(m2beta2,))[0]




# Import thermal integral data
jb_fpath = pkg_resources.resource_filename(__name__, "data/boson_thermal_integral.txt")
jf_fpath = pkg_resources.resource_filename(__name__, "data/fermion_thermal_integral.txt")

jb_integral_data = np.genfromtxt(jb_fpath)
jf_integral_data = np.genfromtxt(jf_fpath)


def JBInterp(m2beta2):
    return np.interp(m2beta2, jb_integral_data[:,0], jb_integral_data[:,1])


def JFInterp(m2beta2):
    return np.interp(m2beta2, jf_integral_data[:,0], jf_integral_data[:,1])




# Generic Finite Temperature Effective Potential Class
class VFT:
    def __init__(self, renorm_mass=1.0e6):
        self.renorm_mass_scale = renorm_mass
        self.T0 = None
        self.Tc = self.get_Tc()

    def a2(self, T):
        return 1.0

    def a3(self, T):
        return 1.0

    def a4(self, T):
        return 1.0

    def phi_plus(self, T):
        # TODO: add exception handling if FOPT is not found
        return np.real(-3*self.a3(T) + sqrt(9*self.a3(T)**2 - 32*self.a4(T)*self.a2(T)))/(8*self.a4(T))
    
    def get_mins(self, T, bounds):
        # returns minima of potential at a given T
        test_phis = np.linspace(-self.renorm_mass_scale, self.renorm_mass_scale, 500)
        test_veffs = np.array([self.__call__(phi, T) for phi in test_phis])

        #min_result = brute(self.__call__, args=(T,), ranges=[(-self.renorm_mass_scale, self.renorm_mass_scale)])
        
        min_ids = argrelmin(test_veffs)
        
        return test_phis[min_ids]
    
    def get_Tc(self, verbose=False):
        T_high = self.renorm_mass_scale
        
        # check that we contain the crossover between these two extrema
        mins_high = self.get_mins(T_high, bounds=[(-T_high, T_high)])
        mins_low = self.get_mins(1.0e-4, bounds=[(-T_high, T_high)])

        if verbose:
            print("mins T=0:", mins_low, "mins T_high = ", mins_high)

        if len(mins_low) < 1:
            print("Starting with potential that has no VEV!")
            return None

        while len(mins_high) > 0.0:
            T_high = 2 * T_high
            mins_high = self.get_mins(T_high, bounds=[(-T_high, T_high)])

        # begin binary search between 1 MeV and 5 * renorm mass scale
        T_arr = np.logspace(-3+np.log10(T_high), np.log10(T_high), 500)

        low, high = 0, len(T_arr) - 1
        target = 0.0  # target derivative dV/dphi(phi+)

        while low <= high:
            mid = (low + high) // 2
            T_mid_value = T_arr[mid]
            if verbose:
                print("Checking T = {}".format(T_mid_value))

            # Compute phi+, phi-
            mins = self.get_mins(T_mid_value, bounds=[(-T_high, T_high)])
            if verbose:
                print("Minima at T={} are phi={}".format(T_mid_value, mins))
            if len(mins) >1:
                phi_plus = max(mins)
                phi_minus = min(mins)
            elif len(mins) == 1:
                phi_plus = max(abs(mins))  # TODO fix sign
            else:
                print("no mins found!")
                return None

            veff_at_min = self.__call__(phi_plus, T_mid_value)
            if abs(veff_at_min) == np.inf:
                print("Found bad Veff -> infinity!")
                return None

            if round(veff_at_min) == target:  # round to nearest GeV^4
                self.Tc = T_mid_value
                return T_mid_value  # Target found, return its index
            elif veff_at_min < target:
                low = mid + 1  # Disregard the left half
            else:
                high = mid - 1  # Disregard the right half
        
        self.Tc = T_mid_value

    def get_dVdphi(self, phi, T):
        return 
    
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




class VEffRealScalarYukawa(VFT):
    def __init__(self, gchi=1.0, mchi=1.0, mu=100.0, lam=0.1, c=0.1, Lambda=1000.0, msign=-1.0):
        self.gchi = gchi  # yukawa coupling
        self.mchi = mchi  # fermion mass
        self.mu = mu  # scalar mass
        self.lam = lam  # quartic coupling
        self.c = c  # cubic coupling
        self.msign = msign
        self.renorm_mass_scale = Lambda
        self.Lambda = Lambda

        super().__init__(renorm_mass=Lambda)
    
    def set_params(self, gchi=1.0, mchi=1.0, mu=100.0, lam=0.1, c=0.1, Lambda=1000.0, msign=-1.0):
        self.gchi = gchi
        self.mchi = mchi
        self.mu = mu
        self.lam = lam
        self.c = c
        self.renorm_mass_scale = Lambda
        self.Lambda = Lambda
        self.msign = msign
        self.Tc = self.get_Tc()

    def a1(self, T):
        return np.power(192*pi**2 * self.mu**2,-1)*(36*sqrt(2)*self.gchi*self.mchi**3*self.mu**2+8*self.c*pi**2*T**2*self.mu**2 \
                                                    +16*sqrt(2)*self.gchi*self.mchi*pi**2*T**2*self.mu**2-9*self.c*self.mu**4 \
                                                    -24*self.c*pi*T**4*(self.mu**2/T**2)**(3/2) \
                                                    +24*sqrt(2)*self.gchi*self.mchi**3*self.mu**2*log(self.mchi**2/(PT_CONST_AF*T**2)) \
                                                    -24*sqrt(2)*self.gchi*self.mchi**3*self.mu**2*log(self.mchi**2/self.Lambda**2) \
                                                    -6*self.c*self.mu**4*log(self.mu**2/(PT_CONST_AB*T**2))+6*self.c*self.mu**4*log(self.mu**2/self.Lambda**2))

    def a2(self, T):
        return np.power(384*pi**2 *self.mu**4, -1)*(-9*self.c**2*self.mu**4+108*self.gchi**2*self.mchi**2*self.mu**4 \
                                                    +16*self.gchi**2*pi**2*T**2*self.mu**4+8*pi**2*T**2*self.lam*self.mu**4 \
                                                    +192*pi**2*self.mu**6-9*self.lam*self.mu**6-12*self.c**2*pi*T**4*(self.mu**2/T**2)**(3/2) \
                                                    -24*pi*T**4*self.lam*self.mu**2*(self.mu**2/T**2)**(3/2) \
                                                    +72*self.gchi**2*self.mchi**2*self.mu**4*log(self.mchi**2/(PT_CONST_AF*T**2)) \
                                                    -72*self.gchi**2*self.mchi**2*self.mu**4*log(self.mchi**2/self.Lambda**2) \
                                                    -6*self.c**2*self.mu**4*log[self.mu**2/(PT_CONST_AB*T**2)] \
                                                    -6*self.lam*self.mu**6*log[self.mu**2/(PT_CONST_AB*T**2)] \
                                                    +6*self.c**2*self.mu**4*log(self.mu**2/self.Lambda**2) \
                                                    +6*self.lam*self.mu**6*log(self.mu**2/self.Lambda**2))
    
    def a3(self, T):
        return self.c / 6
    
    def a4(self, T):
        return self.lam / 24
    
    def mu2(self, phi, T):
        # with thermal mass correction
        return np.real((self.msign*self.mu**2 + self.c * phi + self.lam * phi**2 / 2)) \
            + (self.lam**2 / 24 + self.gchi**2 / 4) * T**2 + (self.c**2 * T / (8*pi*self.mu))

    def mchi2(self, phi):
        return ((self.mchi + self.gchi * phi))**2
    
    def cw(self, m2):
        return power(m2 / 8 / pi, 2) * (log(abs(m2) / self.Lambda**2) - 1.5)
    
    def vtree(self, phi):
        return (self.msign * self.mu**2 / 2) * phi**2 + (self.c / 6) * phi**3 + (self.lam / 24) * phi**4
    
    def vacuum_energy_corr(self):
        return (12*self.mchi**4 - 3*self.mu**4 - 8*self.mchi**4 * log(self.mchi**2 / self.Lambda**2) \
                + 2 *self.mu**4 * log(self.mu**2 / self.Lambda**2))/(128 * pi**2)
    
    def v_thermal(self, phi, T):
        return T**4 * JBInterp(self.mu2(phi, T)/T**2) / (2*pi**2) - 2*T**4 * JFInterp(self.mchi2(phi)/T**2) / pi**2

    def vtot(self, phi, T):
        return np.real(self.vtree(phi) + self.cw(self.mu2(phi, T)) - 4*self.cw(self.mchi2(phi)) \
            + T**4 * JBInterp(self.mu2(phi, T)/T**2) / (2*pi**2) - 2*T**4 * JFInterp(self.mchi2(phi)/T**2) / pi**2) \
            - self.vacuum_energy_corr()
    
    def phi_minima(self):
        return (2/self.lam) * (-self.c + sqrt(self.c**2 - 3*self.lam*self.msign*self.mu**2)), \
            (2/self.lam) * (-self.c - sqrt(self.c**2 - 3*self.lam*self.msign*self.mu**2))

    def __call__(self, phi, T):
        # return the V(0,T) subtracted potential
        return self.vtot(phi, T) - self.vtot(0.0, T)



class VEffMarfatia2(VFT):
    def __init__(self, a=0.1, lam=0.061, c=0.249, d=0.596, b=75.0**4):
        super().__init__()
        self.a = a
        self.b = b
        self.lam = lam
        self.c = c
        self.d = d
        self.T0 = self.get_T0_from_B()
        self.Tc = self.get_Tc()
    
    def phi_plus_marf(self, T0):
        return (3*self.c + sqrt(9*self.c**2 + 8*self.lam*self.d*T0**2))/(2*self.lam)

    def get_T0_from_B(self):
        def root_func(T0):
            return self.b + (-self.d*T0**2 * self.phi_plus_marf(T0)**2 - self.c*self.phi_plus_marf(T0)**3 + self.lam*self.phi_plus_marf(T0)**4 / 4)
        
        res = fsolve(root_func, [1.0])
        return res[0]
    
    def a2(self, T):
        return self.d * (T**2 - self.T0**2)
    
    def a3(self, T):
        return -(self.a*T + self.c)

    def a4(self, T):
        return 0.25*self.lam
    
    def set_params(self, a=0.1, lam=0.061, c=0.249, d=0.596, b=75.0**4):
        self.a = a
        self.lam = lam
        self.c = c
        self.d = d
        self.b = b
        self.T0 = self.get_T0_from_B()
        self.Tc = self.get_Tc()

    def __call__(self, phi, T):
        return np.real(self.d * (T**2 - self.T0**2)*phi**2 - (self.a*T + self.c)*phi**3 + 0.25*self.lam*phi**4)