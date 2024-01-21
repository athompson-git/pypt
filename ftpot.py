# Class for storing a finite temp effective potential

from .constants import *
from .cosmology_functions import *

import pkg_resources
from scipy.special import zeta, gamma, factorial

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

# High-T expansion of thermal integrals
l_list = np.arange(1, 20, 1)
def JB_highT(x):
    return -(pi**4 / 45) + (pi**2 / 12)*x - (pi/6)*power(abs(x), 3/2) \
        - (1/32)*x**2 * log(abs(x) / PT_CONST_AB)

def JF_highT(x):
    return (7*pi**4 / 360) - (pi**2 / 24)*x - (1/32)*x**2 * log(abs(x) / PT_CONST_AF)



# Generic Finite Temperature Effective Potential Class
class VFT:
    def __init__(self, renorm_mass=1.0e6, verbose=False):
        self.renorm_mass_scale = renorm_mass
        self.T0 = None
        self.get_Tc(verbose=verbose)

    def a2(self, T):
        return 1.0

    def a3(self, T):
        return 1.0

    def a4(self, T):
        return 1.0

    def phi_plus(self, T):
        # TODO: add exception handling if FOPT is not found
        return np.real(-3*self.a3(T) + sqrt(9*self.a3(T)**2 - 32*self.a4(T)*self.a2(T)))/(8*self.a4(T))
    
    def get_mins(self, T):
        # returns minima of potential at a given T
        test_phis = np.linspace(-10*self.renorm_mass_scale, 10*self.renorm_mass_scale, 10000)
        test_veffs = np.array([self.__call__(phi, T) for phi in test_phis])

        idx_extrema = argrelmin(test_veffs, axis=0)

        minima_candidates = test_phis[idx_extrema[0]]

        # Once minima are found, refine scan for each.
        #refined_minima = []
        #for min in minima_candidates:
        #    refine_phis = np.linspace(0.9*min, 1.1*min, 100)
        #    refine_veffs = np.array([self.__call__(phi, T) for phi in refine_phis])
            
        #    refined_minima.extend(refine_phis[argrelmin(refine_veffs)])

        return minima_candidates
    
    def get_Tc(self, verbose=False):
        self.Tc = None

        T_high = 10*self.renorm_mass_scale
        
        # check that we contain the crossover between these two extrema
        mins_high = self.get_mins(T_high)
        mins_low = self.get_mins(1.0e-4)  # choose T_low at 0.1 MeV

        if verbose:
            print("mins T=0:", mins_low, "mins T_high = ", mins_high)
            print("shape of mins_high = {}".format(mins_high.shape))

        if len(mins_low) < 1 or len(mins_high) < 1:
            print("Starting with potential that has no VEV (high or low)!")
            return None

        # begin binary search between 1 MeV and 5 * renorm mass scale
        # search for where V(phi) > 0 for all phi
        test_phis = np.linspace(0.001, 10*self.renorm_mass_scale, 1000)
        tol = 0.001*self.renorm_mass_scale  # 1% tolerance of the renorm. mass scale
        T_low = 1e-6

        low, high = T_low, T_high
        target = 0.0 # target value of potential: use T_low as ref.
        if verbose:
            print("Searching over range ", abs(low - high), tol)

        while abs(low - high) > tol:
            T_mid_value = (low + high) / 2

            # Compute phi+, phi-
            mins = self.get_mins(T_mid_value)
            if verbose:
                print("--- Minima at T={} are phi={}".format(T_mid_value, mins))
            
            # Pick the nontrivial VEV
            if len(mins) < 1:
                print("--- no mins found!")
                return None

            # Check how many points are above zero
            test_veffs = self.__call__(test_phis, T_mid_value)
            is_stable = np.any(test_veffs < 0.0)

            if abs(low - high) < tol and len(mins)>1:  # round to nearest GeV^4
                if verbose:
                    print("**** FOUND T CRITICAL AT T = {} ****".format(T_mid_value))
                self.Tc = T_mid_value
                return T_mid_value  # Target found, return its index
            elif is_stable:
                low = T_mid_value  # Disregard the left half
            else:
                high = T_mid_value  # Disregard the right half
        
        self.Tc = T_mid_value
        return T_mid_value

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
    def __init__(self, gchi=1.0, mchi=1.0, mu=100.0, lam=0.1, c=0.1, Lambda=1000.0,
                 msign=-1.0, verbose=False):
        self.gchi = gchi  # yukawa coupling
        self.mchi = mchi  # fermion mass
        self.mu = mu  # scalar mass
        self.lam = lam  # quartic coupling
        self.c = c  # cubic coupling
        self.msign = msign
        self.renorm_mass_scale = Lambda
        self.Lambda = Lambda

        super().__init__(renorm_mass=Lambda, verbose=verbose)
    
    def set_params(self, gchi=1.0, mchi=1.0, mu=100.0, lam=0.1, c=0.1, Lambda=1000.0, msign=-1.0, verbose=False):
        self.gchi = gchi
        self.mchi = mchi
        self.mu = mu
        self.lam = lam
        self.c = c
        self.renorm_mass_scale = Lambda
        self.Lambda = Lambda
        self.msign = msign
        self.Tc = self.get_Tc(verbose=verbose)

    def a1(self, T):
        return 0.0
    
    def a2(self, T):
        return self.msign*self.mu**2
    
    def a3(self, T):
        return self.c / 6
    
    def a4(self, T):
        return self.lam / 24
    
    def daisy(self, T):
        # return daisy corrections to the thermal mass
        return (self.lam**2 / 24 + self.gchi**2 / 4)*T**2 - (self.c**2 * T / (8*pi*self.mu))
        
    def mu2(self, phi, T):
        # with thermal mass correction
        return self.msign*self.mu**2 + self.c * phi + self.lam * phi**2 / 2 #- self.daisy(T)

    def mchi2(self, phi):
        return (self.mchi + self.gchi * phi)**2
    
    def cw(self, m2):
        return power(m2 / 8 / pi, 2) * (log((m2)**2)/2 - 1.5)  # taking Log(x^2)/2 = re[Log[x]]
    
    def vtree(self, phi):
        return (self.msign * self.mu**2 / 2) * phi**2 + (self.c / 6) * phi**3 + (self.lam / 24) * phi**4
    
    def vct(self, phi):
        deltaOmega = (-12*self.mchi**4 + 3*self.mu**4 + 8*self.mchi**4 * log(self.mchi**2) \
                        + 2*self.mu**4 * log(self.mu**2))/(128 * pi**2)
        
        deltaP = (4*sqrt(2)*self.gchi*self.mchi**3 * (log(power(self.mchi, 2))-1) \
                 + self.msign*self.c*self.mu**2 * (1-2*log(power(self.mu, 2))))/(32*pi**2)
        
        deltaMu2 = self.msign*self.mu**2 + (-4*power(self.gchi*self.mchi, 2) + self.msign*self.lam*self.mu**2 \
                   + 12*power(self.gchi*self.mchi,2)*log(power(self.mchi, 2)) \
                    - (self.c**2 + self.msign*self.lam*self.mu**2)*log(power(self.mu,2)))/(32*pi**2)
        
        deltaC = self.c-(1/(32*pi**2))*(-8*sqrt(2)*self.mchi*self.gchi**3 - 8*self.Lambda*self.gchi**4 \
                           + (2*power(self.c+self.lam*self.Lambda, 3))/(2*self.c*self.Lambda + self.lam*self.Lambda**2 + 2*self.msign*self.mu**2)\
                             -12*self.gchi**3 * (sqrt(2)*self.mchi + self.gchi*self.Lambda) * log(power(self.mchi + self.gchi*self.Lambda/sqrt(2),2)) \
                                + 3*self.lam*(self.c + self.lam*self.Lambda)*log(abs(self.c*self.Lambda + 0.5*self.lam*self.Lambda**2 + self.msign*self.mu**2)))
        
        deltaLambda = self.lam + (1/(32*pi**2))*(32*self.gchi**4 + 12*self.gchi**4 * log((self.mchi + self.gchi*self.Lambda/sqrt(2))**2) \
                                    - 3*self.lam**2 * log(abs(self.c*self.Lambda + 0.5*self.lam*self.Lambda**2 + self.msign*self.mu**2)) \
                                    + 4*(self.c + self.lam*self.Lambda)**2 \
                                    * (self.c**2 - 4*self.c*self.lam*self.Lambda - 2*power(self.lam*self.Lambda,2) - 6*self.lam*self.msign*self.mu**2) \
                                        / (2*self.c*self.Lambda + self.lam*self.Lambda**2 + 2*self.msign*self.mu**2)**2)
        
        return deltaOmega + deltaP*phi + (deltaMu2/2)*phi**2 + (deltaC/6)*phi**3 + (deltaLambda/24)*phi**4

    def v_thermal(self, phi, T):
        return T**4 * JBInterp(self.mu2(phi, T)/T**2) / (2*pi**2) - 2*T**4 * JFInterp(self.mchi2(phi)/T**2) / pi**2

    def v_zero_temp(self, phi):
        return np.real(self.vtree(phi) + self.cw(self.mu2(phi, 0.0)) - 4*self.cw(self.mchi2(phi)) + self.vct(phi))

    def vtot(self, phi, T):
        return np.real( self.vtree(phi) + self.cw(self.mu2(phi, T)) - 4*self.cw(self.mchi2(phi)) \
            + self.v_thermal(phi, T) + self.vct(phi) )
    
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