"""
The methods in this code are meant to reflect the solution to the equation of motion derived in
Blau, Guendelman, Guth (BGG) and Flores, Kusenko Sasaki (FKS) from the Israel junction conditions.
The solution to the EoM is for a false vacuum patch collapsing with the expansion of true vacuum bubbles
into a primordial black hole.

References:
Dynamics of false-vacuum bubbles, Blau, Guendelman, Guth (1987)
Revisiting formation of primordial black holes in a supercooled first-order phase
transition, Flores, Kusenko, Sasaki (2024) [2402.13341]

"""

from scipy.integrate import solve_ivp
from scipy.constants import pi

from .constants import *
from .cosmology_functions import *

# Constants
M_PL_FKS = M_PL * np.power(8*pi, -0.5)
G_NEWTON_FKS = np.power(8*pi, 0.5) / M_PL**2  # Newton's constant


class FKSCollapse:
    def __init__(self, deltaV, sigma, vw=1.0):
        """
        Class that takes in the potntial parameters and the initial radial size of a
        false vacuum patch and calculates the trajectory and collapse condition
        for PBH formation.
        deltaV : the potential difference in GeV^4
        sigma: the wall tension in GeV^3
        """
        self.rhoV = deltaV  # potential difference at the FV formation time
        self.sigma = sigma  # wall tension
        self.vw = vw

        # Get potential parameter constants
        self.eta = self.get_eta()
        self.Hsigma2 = self.get_Hsigma2()
        self.HV2 = self.get_HV2()
        self.gamma = self.get_gamma()

    # Helper functions to define constants; these execute first in init
    def get_HV2(self):
        return self.rhoV / (3 * M_PL_FKS**2)

    def get_Hsigma2(self):
        return (self.sigma / (2 * M_PL_FKS**2))**2

    def get_eta(self):
        return np.sqrt(self.get_Hsigma2() / self.get_HV2())
    
    def get_gamma(self):
        return (2 * self.get_eta()) / np.sqrt(1 + self.get_eta()**2)
    
    # Collapse potential functions
    def Uz(self, z):
        return -((1 - z**3) / z**2)**2 - (self.gamma**2 / z)

    def E0(self, M):
        return (-4 * self.eta**2) / ((2 * G_NEWTON_FKS * M)**(2/3) * self.HV2**(1/3) * (1 + self.eta**2)**(4/3))

    def zs(self, E0):
        return self.gamma**2 / abs(E0)

    def z(self, r, M):
        return ((self.HV2 + self.Hsigma2) / (2 * G_NEWTON_FKS * M))**(1/3) * r

    def rFromZ(self, z, M):
        return z/((self.HV2 + self.Hsigma2) / (2 * G_NEWTON_FKS * M))**(1/3)

    def dz(self, dr, M):
        return ((self.HV2 + self.Hsigma2) / (2 * G_NEWTON_FKS * M))**(1/3) * dr

    def M0(self, r):
        # Todo: divide vwall by scale factor
        return (4 * pi / 3) * self.rhoV * r**3 - 8 * pi**2 * G_NEWTON_FKS * self.sigma**2 * r**3 \
            + 4 * pi * self.sigma * r**2 \
                * np.sqrt( np.clip(1 - ((8 * pi * G_NEWTON_FKS / 3) * self.rhoV * r**2 + self.vw**2),
                                   a_min=0.0, a_max=2.0) )

    def H2(self, T, gstar):
        return (8 * pi**3 * gstar * T**4) / (90 * M_PL_FKS**2)

    # Gradients
    def dU_dz(self, z):
        # Calculate dU/dz as derived
        return 6 * (1 - z**3) / z**2 + 4*(1-z**3)**2 / z**5 \
                + 4 * self.eta**2 / (z**2 * (1 + self.eta**2)) 

    def dz_dr(self, rhoV, sigma, M):
        # Calculate dz/dr
        return ((self.HV2(rhoV) + self.Hsigma2(sigma)) / (2 * G_NEWTON_FKS * M))**(1/3)

    def dU_dr(self, r, M):        
        # Calculate z as a function of r
        z_val = self.z(r, M)
        
        # Calculate dU/dz
        dUdz_val = self.dU_dz(z_val)
        
        # Calculate dz/dr
        dzdr_val = self.dz_dr(M)
        
        # Chain rule: dU/dr = dU/dz * dz/dr
        return dUdz_val * dzdr_val

    def tau(self, tau_prime, rhoV, sigma):
        #Convert from tau_prime to tau
        numerator = 2 * np.sqrt(self.Hsigma2(sigma)) * tau_prime
        denominator = self.HV2(rhoV) + self.Hsigma2(sigma)
        return numerator / denominator
    
    # COLLAPSE CONDITION
    def does_pbh_form(self, r_fv):
        """
        Check the mass and formation time of the PBH in coordinate z
        Compare that with U(z): does the false vacuum patch form smaller than z_TP?
        In other words, does it form on the other side of the formation potential barrier U(z)
        Takes in false vacuum patch radius r_fv in GeV^-1
        """
        # Get Schwarzchild radius
        m_pbh = self.M0(r_fv)
        E_pbh = self.E0(m_pbh)
        z_sc = self.gamma**2 / abs(E_pbh)

        # Determine z_m (location of potential maximum)
        z_m = np.power(0.5*sqrt(8 + (1 - 0.5*self.gamma**2)**2) - 0.5*(1 - 0.5*self.gamma**2), 1/3)

        # If we already above the barrier, collapse
        if E_pbh > self.Uz(z_m):
            return True
        
        # Determine z_TP otherwise and check
        test_z = np.linspace(0.0, z_m, 10000)  # only check values up to the max to get the left crossing point
        z_TP = test_z[np.argmin(abs(self.Uz(test_z) - E_pbh))]

        z_fv = self.z(r_fv, m_pbh)

        if z_fv <= z_sc:
            print("False vacuum patch radius already smaller than Schwarzchild radius...")
        
        if z_fv <= z_TP:
            return True
        
        return False
    
    def get_crit_mass(self):
        mbar = 4*pi*M_PL_FKS**2 / sqrt(self.HV2)
        z_m = np.power(0.5*sqrt(8 + (1 - 0.5*self.gamma**2)**2) - 0.5*(1 - 0.5*self.gamma**2), 1/3)

        return mbar * power(self.gamma * z_m**2, 2) * sqrt(1 - self.gamma**2 / 4) \
            / (3*sqrt(3) * power(power(z_m, 6) - 1, 3/2))
    
    def get_M_at_zTP(self):
        """
        Find the mass in GeV that would intersect with z_TP
        """
        z0 = power((self.HV2 + self.Hsigma2) \
                   /((4*pi/3)*self.rhoV - 8*pi**2 * G_NEWTON_FKS*self.sigma**2)/(2*G_NEWTON_FKS), 1/3)
        
        m_tp_prefact = (-4 * self.eta**2) / ((2 * G_NEWTON_FKS)**(2/3) * self.HV2**(1/3) * (1 + self.eta**2)**(4/3))

        return power(m_tp_prefact / self.Uz(z0), 3/2)

    def get_collapse_time(self, r_fv):
        does_collapse = self.does_pbh_form(r_fv)
        if not does_collapse:
            return np.inf

        m_pbh = self.M0(r_fv)
        E_pbh = self.E0(m_pbh)
        z_sc = self.gamma**2 / abs(E_pbh)  # Schwarzchild radius
        z0 = self.z(r_fv, m_pbh)

        if z0 <= z_sc:
            return 0.0  # collapse already happens

        def integrand(z):
            return 1 / sqrt(E_pbh - self.Uz(z))
        
        tau = quad(integrand, z_sc, z0)  # dimensionless time

        return tau * (2 * sqrt(self.Hsigma2)) / (self.HV2 + self.Hsigma2)  # convert back to GeV^-1
    
    # Simulation functions
    def simulate_z(self, r0, dr0, time_span=(0, 1e10)):
        m_pbh = self.M0(r0)
        E0 = self.E0(m_pbh)

        def dz_dt(t, y):
            z, dzdt = y
            
            #dzdt = np.array(np.sqrt(Energy - Uz(z, eta_BP1)))  # Acceleration term from potential
            d2zdt2 = np.array(-0.5*self.dU_dz(z) / np.sqrt(E0 - self.Uz(z)))
            return [dzdt, d2zdt2]
    
        z0 = self.z(r0, m_pbh)
        dz0 = self.dz(dr0, m_pbh) * 2*np.sqrt(self.Hsigma2) / (self.HV2 + self.Hsigma2)

        initial_conditions_z = [z0, dz0]
        solution_z = solve_ivp(dz_dt, time_span, initial_conditions_z,
                            method='RK45', dense_output=True, max_step=0.00000001)
        t_values = solution_z.t
        z_values = solution_z.y[0]
        zp_values = solution_z.y[1]

        return t_values, z_values, zp_values


