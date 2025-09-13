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
    def __init__(self, renorm_mass=1.0e6, verbose=False, Tc=None, is_real=False):
        self.is_real = is_real
        self.renorm_mass_scale = renorm_mass
        if Tc is None:
            self.get_Tc(verbose=verbose)
        else:
            self.Tc = Tc

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
        if self.is_real:
            test_phis = np.linspace(-30*self.renorm_mass_scale, 30*self.renorm_mass_scale, 10000)
            test_veffs = np.array([self.__call__(phi, T) for phi in test_phis])
        else:
            test_phis = np.linspace(-0.0001, 30*self.renorm_mass_scale, 10000)
            test_veffs = np.array([self.__call__(phi, T) for phi in test_phis])
            test_veffs[0] = test_veffs[1] + 0.0000001  # overwrite to ensure minimum at phi=0

        idx_extrema = argrelmin(test_veffs, axis=0)
        minima_candidates = test_phis[idx_extrema[0]]

        return minima_candidates
    
    def get_Tc(self, verbose=False):
        self.Tc = None

        T_high = 100*self.renorm_mass_scale
        T_low = 1e-6
        
        # check that we contain the crossover between these two extrema
        mins_high = self.get_mins(T_high)
        mins_low = self.get_mins(T_low)  # choose T_low at 1 keV

        if verbose:
            print("mins T=0:", mins_low, "mins T_high = ", mins_high)
            print("shape of mins_high = {}".format(mins_high.shape))

        if len(mins_low) < 1:
            print("Starting with potential that has no T=0 VEV (high or low)!")
            return None

        # begin binary search between 1 MeV and 5 * renorm mass scale
        # search for where V(phi) > 0 for all phi
        test_phis = np.linspace(0.0001, 100*self.renorm_mass_scale, 1000)
        tol = 0.0001*self.renorm_mass_scale  # 1% tolerance of the renorm. mass scale
        

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
                 msign=-1.0, Tc=None, verbose=False, is_real=False):
        self.gchi = gchi  # yukawa coupling
        self.mchi = mchi  # fermion mass
        self.mu = mu  # scalar mass
        self.lam = lam  # quartic coupling
        self.c = c  # cubic coupling
        self.msign = msign
        self.renorm_mass_scale = Lambda
        self.Lambda = Lambda

        super().__init__(renorm_mass=Lambda, Tc=Tc, verbose=verbose, is_real=is_real)
    
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
        return self.msign*self.mu**2 + self.c * phi + self.lam * phi**2 / 2 - self.daisy(T)

    def mchi2(self, phi):
        return (self.mchi + self.gchi * phi)**2
    
    def cw(self, m2):
        return power(m2 / 8 / pi, 2) * (log((m2)**2)/2 - 1.5)  # taking Log(x^2)/2 = re[Log[x]]
    
    def vtree(self, phi):
        return (self.msign * self.mu**2 / 2) * phi**2 + (self.c / 6) * phi**3 + (self.lam / 24) * phi**4
    
    def vct(self, phi):
        deltaOmega = (-12*self.mchi**4 + 3*self.mu**4 + 8*self.mchi**4 * log(self.mchi**2) \
                        - 2*self.mu**4 * log(self.mu**2))/(128 * pi**2)
        
        deltaP = (8*self.gchi*self.mchi**3 * (log(power(self.mchi, 2))-1) \
                 + self.msign*self.c*self.mu**2 * (1-log(power(self.mu, 2))))/(32*pi**2)
        
        deltaMu2 = (-8*power(self.gchi*self.mchi, 2) + self.msign*self.lam*self.mu**2 \
                   + 24*power(self.gchi*self.mchi,2)*log(power(self.mchi, 2)) \
                    - 2*(self.c**2 + self.msign*self.lam*self.mu**2)*log(power(self.mu,2)))/(32*pi**2)
        
        xi1 = abs(self.c*self.Lambda + 0.5*self.lam*self.Lambda**2 + self.msign*self.mu**2)
        xi2 = self.c+self.lam*self.Lambda
        mchi_shift = power(self.mchi + self.gchi*self.Lambda,2)

        deltaC = -self.lam*self.Lambda - (1/(32*pi**2))*(-32*self.mchi*self.gchi**3 + 96*self.Lambda*self.gchi**4 \
                            + (4*self.Lambda*power(xi2, 4))/power(2*xi1,2) \
                            + (2*(self.c-5*self.lam*self.Lambda)*power(xi2,2))/(2*xi1) \
                            -48*self.gchi**3 * self.mchi * log(mchi_shift) \
                            + 3*self.lam*self.c*log(xi1))
        
        deltaLambda = (1/(32*pi**2))*(self.gchi**4 * (128 + 48*log(mchi_shift)) \
                                    - 3*self.lam**2 * log(xi1) \
                                    + 4*xi2**2 * (self.c**2 - 4*self.c*self.lam*self.Lambda - 2*power(self.lam*self.Lambda,2) - 6*self.lam*self.msign*self.mu**2) / power(2*xi1, 2))
        
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



class VEffGeneric(VFT):
    def __init__(self, a=0.1, lam=0.061, c=0.249, d=0.596,
                 vev=None, b=75.0**4, verbose=False) -> None:
        self.verbose = verbose
        self.a = a
        self.b = b
        self.lam = lam
        self.c = c
        self.d = d
        if vev is not None:            
            self.T0sq = self.get_T0sq_from_vev(vev)
            self.vev = vev
        elif b is not None:
            self.T0sq = self.get_T0sq_from_B()
            self.vev = self.phi_plus_from_T0(self.T0sq)
        else:
            raise Exception("either the VEV or b must be set!")
        
        if verbose:
            print("VEV = {}, T0^2={}".format(self.vev, self.T0sq))
        Tc = self.get_Tc()

        bad_Tc = (np.isnan(Tc)) or (Tc < 0)
        if bad_Tc:
            print("Bad Tc found! Either imaginary or negative.")

        super().__init__(renorm_mass=self.vev, verbose=verbose, is_real=False, Tc=Tc)
    
    def set_params(self, a=0.1, lam=0.061, c=0.249, d=0.596,
                   vev=None, b=75.0**4) -> None:
        self.a = a
        self.b = b
        self.lam = lam
        self.c = c
        self.d = d

        if vev is not None:
            self.T0sq = self.get_T0sq_from_vev(vev)
            self.vev = vev
        elif b is not None:
            self.T0sq = self.get_T0sq_from_B()
        else:
            raise Exception("either the VEV or b must be set!")
        
        self.vev = self.phi_plus_from_T0(self.T0sq)
        Tc = self.get_Tc()

        bad_Tc = (np.isnan(Tc)) or (Tc < 0)
        if bad_Tc:
            print("Bad Tc found! Either imaginary or negative.")
        
        self.renorm_mass_scale = vev
        self.Tc = Tc
    
    def get_vev(self, T) -> float:
        phi1 = (3*(self.c + self.a*T) - sqrt(9*(self.c + self.a*T)**2 - 8*self.d*self.lam*(T**2 - self.T0sq)))/(2*self.lam)
        phi2 = (3*(self.c + self.a*T) + sqrt(9*(self.c + self.a*T)**2 - 8*self.d*self.lam*(T**2 - self.T0sq)))/(2*self.lam)

        # both vev and its 2nd derivative must be positive
        vev_candidates = [0.0]
        if self.d2Vdphi(phi2, T) > 0. and phi2 > 0.:
            vev_candidates.append(phi2)
        elif self.d2Vdphi(phi1, T) > 0. and phi1 > 0.:
            vev_candidates.append(phi1)
        
        pots = [self.__call__(x, T) for x in vev_candidates]
        vev_id = np.argmin(pots)

        return vev_candidates[vev_id]

    def phi_plus_from_T0(self, T0sq) -> float:
        return (3*self.c + sqrt(9*self.c**2 + 8*self.lam*self.d*T0sq))/(2*self.lam)
    
    def phi_critical(self) -> float:
        return self.Tc * (2*(self.a + self.c/self.Tc)/self.lam)
    
    def wall_tension(self) -> float:
        # from thin wall approx
        return power(self.phi_critical(), 3) * power(self.lam/2, 0.5) / 6

    def get_T0sq_from_B(self) -> float:
        def root_func(T0sq):
            return self.b + (-self.d*T0sq * self.phi_plus_from_T0(T0sq)**2 - self.c*self.phi_plus_from_T0(T0sq)**3 \
                             + self.lam*self.phi_plus_from_T0(T0sq)**4 / 4)
        
        res = fsolve(root_func, [1.0])
        return res[0]
    
    def get_T0sq_from_vev(self, vev) -> float:
        return (self.lam * vev**2 - 3*self.c*vev)/(2*self.d)
    
    def get_Tc(self) -> float:
        # from FKS
        return (self.c*self.a + np.sqrt(self.lam*self.d*(self.c**2 + (self.lam*self.d - self.a**2)*self.T0sq)))/(self.lam*self.d - self.a**2)

    def a2(self, T) -> float:
        return self.d * (T**2 - self.T0sq)
    
    def a3(self, T) -> float:
        return -(self.a*T + self.c)

    def a4(self, T) -> float:
        return 0.25*self.lam
    
    def dVdT(self, phi, T) -> float:
        # first derivative of the potential with respect to temperature
        return 2*self.d*T*phi**2 - self.a*phi**3
    
    def d2VdT2(self, phi) -> float:
        # second derivative of the potential with respect to temperature
        return 2*self.d*phi**2
    
    def d2Vdphi(self, phi, T) -> float:
        return 2*self.d*(T**2-self.T0sq) - 6*(self.c + self.a*T)*phi + 3. * phi**2 / self.lam

    def __call__(self, phi, T) -> float:
        return np.real(self.d * (T**2 - self.T0sq)*phi**2 - (self.a*T + self.c)*phi**3 + 0.25*self.lam*phi**4)