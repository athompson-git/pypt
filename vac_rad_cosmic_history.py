from .ftpot import *
from .bubble_nucleation import *
from .cosmology_functions import *

from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

def time_to_temp(t, gstar=GSTAR_SM):
    return sqrt(sqrt(90/pi**3/gstar/8) * M_PL / t)

class CosmicHistoryVacuumRadiation:
    def __init__(self, deltaV, sigma, vw, veff: VEffGeneric):
        self.dV = deltaV
        self.bn = BubbleNucleationQuartic(veff)
        self.vw = vw
        self.sigma = sigma

        # get the equality time

        Heq2, Teq = self.get_equality_quantities()
        self.Heq2 = Heq2
        self.Teq = Teq

        self.Tau_crit = temp_to_time(veff.Tc) * sqrt(Heq2) / sqrt(2)  # dimensionless

    # Solve for time of equality
    def get_equality_quantities(self):
        def f(x):
            return (2*pi**2 / 90) * gstar_sm(x) * x**4 - (2/3)*self.dV
        
        root = fsolve(f, [0])
        Teq = max(root)

        return (2/3)*self.dV/(M_PL**2), Teq


    # scale factor
    def da_dTau(self, tau, y):
        # a, rhoR, v0, v1, v2, v3, r = y[0], y[1], y[2], y[3], y[4], y[5], y[6]
        return y[0] * power(y[1] + self.rhoV(tau, y), 0.5) 

    # radiation density
    def dRhoR_dTau(self, tau, y):
        # TODO: add drhoV/dtau
        return - (4*y[1]*self.da_dTau(tau, y)/y[0] + 0)

    # Radius
    def dr_dTau(self, tau, y):
        return heaviside(tau - self.Tau_crit, 0.0) * self.vw / y[0]

    # dv_i / d_tau
    def dv0_dTau(self, tau, y):
        # a, rhoR, v0, v1, v2, v3, r = y[0], y[1], y[2], y[3], y[4], y[5], y[6]
        return heaviside(tau - self.Tau_crit, 0.0) \
            * self.decay_rate(tau) * power(y[0], 3)

    def dv1_dTau(self, tau, y):
        return heaviside(tau - self.Tau_crit, 0.0) \
            * self.decay_rate(tau) * power(y[0], 3) * y[6]

    def dv2_dTau(self, tau, y):
        return heaviside(tau - self.Tau_crit, 0.0) \
            * self.decay_rate(tau) * power(y[0], 3) * power(y[6], 2)

    def dv3_dTau(self, tau, y):
        return heaviside(tau - self.Tau_crit, 0.0) \
            * self.decay_rate(tau) * power(y[0], 3) * power(y[6], 3)

    # Helper functions
    # Big Gamma
    def decay_rate(self, tau):
        # TODO(AT): go away from rad domination time-to-temperature conversion
        # returns dimensionless decay rate
        # convert temp to time
        T = time_to_temp(sqrt(2) * tau / sqrt(self.Heq2))  # convert from dimensionless time
        dimensionless_rate = 4 * self.bn.rate(T) / power(self.Heq2, 2)
        return dimensionless_rate

    # Vacuum energy fraction
    def I_frac(self, tau, r, v0, v1, v2, v3):
        return (4*pi/3) * (power(r, 3)*v0 - 3*power(r, 2)*v1 + 3*r*v2 - v3)

    # Vacuum energy density
    def rhoV(self, tau, y):
        return (2*self.dV / (3*self.Heq2*power(M_PL, 2))) * np.exp(-self.I_frac(tau, y[6], y[2], y[3], y[4], y[5]))

    def solve_system(self, max_time=20):
        # Initial conditions
        y0 = [1, 1, 0, 0, 0, 0, 0]  # Initial values for y1, y2, ..., y7

        def system_of_odes(tau, y):
            dydt = np.zeros(7)
            dydt[0] = self.da_dTau(tau, y)
            dydt[1] = self.dRhoR_dTau(tau, y)
            dydt[2] = self.dv0_dTau(tau, y)
            dydt[3] = self.dv1_dTau(tau, y)
            dydt[4] = self.dv2_dTau(tau, y)
            dydt[5] = self.dv3_dTau(tau, y)
            dydt[6] = self.dr_dTau(tau, y)
            return dydt
        
        # Time span for the solution
        t_span = (self.Tau_crit, max_time*self.Tau_crit)  # From t=0 to t=10
        t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Points to evaluate the solution


        return solve_ivp(system_of_odes, t_span, y0, t_eval=t_eval, method='RK45')


"""
# example of solve_ivp:

    solves dy/dt = f(t,y)

    def func(x, y):
        return -1/x**2*sigmav(epsilon, ma,1/x, alphad, mf, mchi_ratio) * (y**2-(4*0.192*mp*m*x**1.5*np.exp(-x))**2)
    return solve_ivp(func, (xinit, xfin), [y0], method='BDF'), y0

"""