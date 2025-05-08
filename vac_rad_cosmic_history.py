from .ftpot import *
from .cosmology_functions import *

from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.misc import derivative

M_PL_FKS = M_PL * np.power(8*pi, -0.5)


def time_to_temp(t, gstar=GSTAR_SM):
    return sqrt(sqrt(90/pi**3/gstar/8) * M_PL / t)




class CosmicHistoryVacuumRadiation:
    def __init__(self, veff: VEffGeneric, vw):
        self.vw = vw
        self.veff = veff

        self.dV = veff.vev**4 * ((1/8) - 0.5*veff.c/veff.vev) / 4
        self.sigma = veff.wall_tension()

        # get the equality time

        Heq2, Teq, teq = self.get_equality_quantities()
        self.Heq2 = Heq2
        self.Teq = Teq
        self.teq = teq

        self.Tau_crit = temp_to_time(veff.Tc) * sqrt(Heq2) / sqrt(2)  # dimensionless

    # Solve for time of equality
    def get_equality_quantities(self):
        # Put temperatures on a grid
        temps = np.linspace(self.veff.Tc/50, self.veff.Tc*50.0, 1000)

        # Find where we have the most equality between rho_R and rho_V
        rho_R = (pi**2 / 30) * gstar_sm(temps) * temps**4

        if not np.any(rho_R < self.dV):
            return None, None, None

        id_equality = np.argmin(abs(self.dV - rho_R))
        Teq = temps[id_equality]
        teq = temp_to_time(Teq)

        # Return Hubble^2, T_eq and t_eq (temp and time in natural units)
        return (2/3)*self.dV/(M_PL_FKS**2), Teq, teq
    
    def get_equality_quantities_tperc(self):
        # Put temperatures on a grid
        temps = np.linspace(self.veff.Tc/50, self.veff.Tc, 200)

        # Get the VEVs at each T
        phi_vevs_T = np.array([self.veff.get_vev(T) for T in temps])

        # Check the potential difference between the VEV and 0 at each T
        dV_by_T = np.array([np.clip(self.veff(0.0, T)-self.veff(phi_vevs_T[i], T),
                                    a_min=0.0, a_max=np.inf) for i,T in enumerate(temps)])
        
        # Find where we have the most equality between rho_R and rho_V
        rho_R = (pi**2 / 30) * gstar_sm(temps) * temps**4

        if not np.any(rho_R < dV_by_T):
            return None, None, None

        id_equality = np.argmin(abs(dV_by_T - rho_R))
        Teq = temps[id_equality]
        teq = temp_to_time(Teq)
        rhoV_eq = dV_by_T[id_equality]

        # Return Hubble^2, T_eq and t_eq (temp and time in natural units)
        return (2/3)*rhoV_eq/(M_PL_FKS**2), Teq, teq

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
    def bounce_action(self, T):
        # Returns S3/T given the parameters in Veff in thin-wall approx
        delta = 8*self.veff.a4(T) * self.veff.a2(T) / self.veff.a3(T)**2
        beta1 = 8.2938
        beta2 = -5.5330
        beta3 = 0.8180
        return np.clip((-pi * self.veff.a3(T) * 8*sqrt(2)*power(2 - delta, -2) \
                        *sqrt(abs(delta)/2) \
            * (beta1*delta + beta2*delta**2 + beta3*delta**3) \
                / power(self.veff.a4(T), 1.5) / 81 / T), a_min=0.0, a_max=np.inf)  # Testing factor of 10 reduction for FKS repro
    
    def kappa_func(self, T):
        return self.veff.lam * 2 * self.veff.d * (T**2 - self.veff.T0sq) / power(3 * (self.veff.a*T + self.veff.c), 2)

    def b3bar(self, kappa):
        return (16/243) * (1 - 38.23*(kappa - 2/9) + 115.26*(kappa - 2/9)**2 + 58.07*sqrt(kappa)*(kappa - 2/9)**2 + 229.07*kappa*(kappa - 2/9)**2)

    def bounce_action_fks(self, T):
        prefactor = power(2 * self.veff.d * (T**2 - self.veff.T0sq), 3/2) / power(3 * (self.veff.a*T + self.veff.c), 2)

        kappa = self.kappa_func(T)
        kappa_gtr_zero = kappa > 0

        kappa_c = 0.52696

        return np.nan_to_num(np.clip(kappa_gtr_zero * (prefactor*(2*pi/(3*(kappa - kappa_c)**2)) * self.b3bar(kappa) / T) \
                + (1 - kappa_gtr_zero) * (prefactor*(27*pi/2) * (1 + np.exp(-power(abs(kappa), -0.5))) / (1 + abs(kappa)/kappa_c) / T),
                        a_min=0.0, a_max=np.inf))

    def rate(self, T):
        return np.real(T**4 * power(abs(self.bounce_action(T)) / (2*pi), 3/2) * np.exp(-abs(self.bounce_action(T))))
    
    def decay_rate(self, tau):
        # TODO(AT): go away from rad domination time-to-temperature conversion
        # returns dimensionless decay rate
        # convert temp to time
        T = time_to_temp(sqrt(2) * tau / sqrt(self.Heq2))  # convert from dimensionless time
        dimensionless_rate = 4 * self.rate(T) / power(self.Heq2, 2)
        return dimensionless_rate

    # Vacuum energy fraction
    def I_frac(self, tau, r, v0, v1, v2, v3):
        return (4*pi/3) * (power(r, 3)*v0 - 3*power(r, 2)*v1 + 3*r*v2 - v3)

    # Vacuum energy density
    def rhoV(self, tau, y):
        return (2*self.dV / (3*self.Heq2*power(M_PL_FKS, 2))) * np.exp(-self.I_frac(tau, y[6], y[2], y[3], y[4], y[5]))

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
        t_eval = np.linspace(t_span[0], t_span[1], 200)  # Points to evaluate the solution


        return solve_ivp(system_of_odes, t_span, y0, t_eval=t_eval, method='RK45')


"""
# example of solve_ivp:

    solves dy/dt = f(t,y)

    def func(x, y):
        return -1/x**2*sigmav(epsilon, ma,1/x, alphad, mf, mchi_ratio) * (y**2-(4*0.192*mp*m*x**1.5*np.exp(-x))**2)
    return solve_ivp(func, (xinit, xfin), [y0], method='BDF'), y0

"""


class CosmicHistoryVacuumRadiation_dT_dt:
    def __init__(self, Veff, vw):
        self.vw = vw
        self.Veff = Veff

        if type(Veff) == VEffGeneric:
            self.dV0 = Veff.vev**4 * ((1/8) - 0.5*Veff.c/Veff.vev) / 4
            self.sigma = Veff.wall_tension()
            self.Tc = Veff.Tc
            self.VEV = Veff.get_vev

        # troubleshooting
        self.data = [[], [], []]

        # get the equality time
        Heq2, Teq, teq = self.get_equality_quantities()
        self.Heq2 = Heq2
        self.Teq = Teq
        self.teq = teq
        
        self.c = self.c()
        self.rho_R_eq = (pi**2 / 30) * gstar_sm(self.Teq) * self.Teq**4

        self.tau_eq = self.temperature_to_tau(Teq)
        self.tau_crit = self.temperature_to_tau(self.Tc)

        # recursion help...value of dT_dt will always be one time-step behind
        self.previous_dT_dt = sqrt(self.c) * self.temperature_to_tau(1) * (-sqrt(hubble2_rad(self.Tc))*self.Tc) # start with radiation domination



    # adimensional temperature...note that plugging in 1 for these (linear) functions gives dT/dτ and dτ/dT
    def tau_to_temperature(self, tau):
        # return power(2 * self.Heq2, 0.5) * tau
        # return power(self.rho_R_eq, 0.25) * tau
        return self.Teq * tau
    def temperature_to_tau(self, T):
        # return T / power(2 * self.Heq2, 0.5)
        # return T / power(self.rho_R_eq, 0.25)
        return T / self.Teq

    # adimensional energy density coefficient (rho_hat = c/(3*M_Pl**2) * rho)
    def c(self):
        return 2 / self.Heq2

    def dV(self, phi, T):
        if type(self.Veff) == VEffGeneric:
            return self.Veff(0.0, T)-self.Veff(phi, T)
    
    # Solve for time of equality
    def get_equality_quantities(self):
        # Put temperatures on a grid
        temps = np.linspace(self.Tc/50, self.Tc, 200)

        # Get the VEVs at each T
        phi_vevs_T = np.array([self.VEV(T) for T in temps])

        # Check the potential difference between the VEV and 0 at each T
        dV_by_T = np.array([np.clip(self.dV(phi_vevs_T[i], T),
                                    a_min=0.0, a_max=np.inf) for i,T in enumerate(temps)])
        
        # Find where we have the most equality between rho_R and rho_V
        rho_R = (pi**2 / 30) * gstar_sm(temps) * temps**4

        if not np.any(rho_R < dV_by_T):
            return None, None, None

        id_equality = np.argmin(abs(dV_by_T - rho_R))
        Teq = temps[id_equality]
        teq = temp_to_time(Teq)
        rhoV_eq = dV_by_T[id_equality]
        # self.data[2].append((rhoV_eq, rho_R[id_equality]))

        # Return Hubble^2, T_eq and t_eq (temp and time in natural units)
        return (2/3)*rhoV_eq/(M_PL**2), Teq, teq

    # adimensional radiation energy density
    def rhoR(self, tau):
        T = self.tau_to_temperature(tau)
        rho = pi**2 / 30 * gstar_sm(T) * T**4
        # rho = pi**2 / 30 * gstar_sm(self.Tc) * T**4 # checking against radiation-domination approximations
        return (self.c/3/M_PL**2) * rho
    def drhoR_dtau(self, tau):
        T = self.tau_to_temperature(tau)
        dgr_dT = derivative(gstar_sm, T)
        drhor_dtau = self.tau_to_temperature(1) * (pi**2/30) * (dgr_dT*T**4 + 4*gstar_sm(T)*T**3)
        # drhor_dtau = self.tau_to_temperature(1) * (pi**2/30) * 4*gstar_sm(self.Tc)*T**3 # checking against radiation-domination approximations
        return (self.c/3/M_PL**2) * drhor_dtau
    
    # scale factor
    def da_dtau(self, tau, y, dT_dt=None):
        if dT_dt is None:
            dT_dt = self.previous_dT_dt
        # a, v0, v1, v2, v3, r, dT/dt = y[0], y[1], y[2], y[3], y[4], y[5], y[6]
        return -y[0]/4/self.rhoR(tau) * (self.drhoR_dtau(tau) + self.drhoV_dtau(tau, y)) # fluid eq.
        # return y[0] / y[6] * power(self.rhoR(tau) + self.rhoV(tau, y), 0.5) # first-order Friedmann eq.
        # return y[0] / dT_dt * power(self.rhoR(tau) + self.rhoV(tau, y), 0.5)
    
    # adimensional √(c) (dτ/dT) (dT/dt)
    def dT_dt(self, tau, y):
        # return y[0] / self.da_dtau(tau, y) * power(self.rhoR(tau) + self.rhoV(tau, y), 0.5)
        rhoR, drhoR = self.rhoR(tau), self.drhoR_dtau(tau)
        rhoV, drhoV = self.rhoV(tau, y), self.drhoV_dtau(tau, y)
        return -4*rhoR * power(rhoR + rhoV, 0.5) / (drhoR + drhoV)
    # def dTdt_dtau(self, tau, y):
    #     term1 = (self.drhoR_dtau(tau) + self.drhoV_dtau(tau, y)) / 2 / power(self.rhoR(tau) + self.rhoV(tau, y), 0.5)
    #     term2 = 2 * self.rhoR(tau) / y[6]
    #     return y[0] / self.da_dtau(tau, y) * (term1 + term2)

    # adimensional comoving radius
    def dr_dtau(self, tau, y):
        return heaviside(self.tau_crit - tau, 0.0) * self.vw / y[0] / self.previous_dT_dt
        # return heaviside(self.tau_crit - tau, 0.0) * self.vw / y[0] / y[6]

    # adimensional dv_i / dτ
    def dv0_dtau(self, tau, y):
        # a, v0, v1, v2, v3, r, dT/dt = y[0], y[1], y[2], y[3], y[4], y[5], y[6]
        return heaviside(self.tau_crit - tau, 0.0) * self.decay_rate(tau) * power(y[0], 3) / self.previous_dT_dt
        # return heaviside(self.tau_crit - tau, 0.0) * self.decay_rate(tau) * power(y[0], 3) / y[6]

    def dv1_dtau(self, tau, y):
        return heaviside(self.tau_crit - tau, 0.0) * self.decay_rate(tau) * power(y[0], 3) * y[5] / self.previous_dT_dt
        # return heaviside(self.tau_crit - tau, 0.0) * self.decay_rate(tau) * power(y[0], 3) * y[5] / y[6]

    def dv2_dtau(self, tau, y):
        return heaviside(self.tau_crit - tau, 0.0) * self.decay_rate(tau) * power(y[0], 3) * power(y[5], 2) / self.previous_dT_dt
        # return heaviside(self.tau_crit - tau, 0.0) * self.decay_rate(tau) * power(y[0], 3) * power(y[5], 2) / y[6]

    def dv3_dtau(self, tau, y):
        return heaviside(self.tau_crit - tau, 0.0) * self.decay_rate(tau) * power(y[0], 3) * power(y[5], 3) / self.previous_dT_dt
        # return heaviside(self.tau_crit - tau, 0.0) * self.decay_rate(tau) * power(y[0], 3) * power(y[5], 3) / y[6]

    # Helper functions
    # Big Gamma
    def bounce_action(self, T):
        # Returns S3/T given the parameters in Veff in thin-wall approx
        delta = 8*self.Veff.a4(T) * self.Veff.a2(T) / self.Veff.a3(T)**2
        beta1 = 8.2938
        beta2 = -5.5330
        beta3 = 0.8180
        return np.clip((-pi * self.Veff.a3(T) * 8*sqrt(2)*power(2 - delta, -2) \
                        *sqrt(abs(delta)/2) \
            * (beta1*delta + beta2*delta**2 + beta3*delta**3) \
                / power(self.Veff.a4(T), 1.5) / 81 / T), a_min=0.0, a_max=np.inf)  # Testing factor of 10 reduction for FKS repro
    
    def kappa_func(self, T):
        return self.Veff.lam * 2 * self.Veff.d * (T**2 - self.Veff.T0sq) / power(3 * (self.Veff.a*T + self.Veff.c), 2)

    def b3bar(self, kappa):
        return (16/243) * (1 - 38.23*(kappa - 2/9) + 115.26*(kappa - 2/9)**2 + 58.07*sqrt(kappa)*(kappa - 2/9)**2 + 229.07*kappa*(kappa - 2/9)**2)

    def bounce_action_fks(self, T):
        prefactor = power(2 * self.Veff.d * (T**2 - self.Veff.T0sq), 3/2) / power(3 * (self.Veff.a*T + self.Veff.c), 2)

        kappa = self.kappa_func(T)
        kappa_gtr_zero = kappa > 0

        kappa_c = 0.52696

        return np.nan_to_num(np.clip(kappa_gtr_zero * (prefactor*(2*pi/(3*(kappa - kappa_c)**2)) * self.b3bar(kappa) / T) \
                + (1 - kappa_gtr_zero) * (prefactor*(27*pi/2) * (1 + np.exp(-power(abs(kappa), -0.5))) / (1 + abs(kappa)/kappa_c) / T),
                        a_min=0.0, a_max=np.inf))

    def bounce_action_cosmoTransitions(self, T):
        # returns S3 / T
        phi_absMin = self.Veff.find_minimum_phi(T)
        phi_metaMin = self.Veff.phi_min # cosmoTransitions is sensitive to this value

        SFI = SingleFieldInstanton(phi_absMin, phi_metaMin, V=lambda phi: self.Veff(phi, T), dV=lambda phi: self.Veff.dVeffBL(phi, T), d2V=lambda phi: self.Veff.d2VeffBL(phi, T))

        phitol = 1e-5 * (T / self.Tc)
        # the default phitol=1e-4 is fine for high temperatures, but needs to be lowered for low temperatures
        profile = SFI.findProfile(phitol=phitol)

        # r, phi, dphi = profile.R, profile.Phi, profile.dPhi
        # integrand = 4*pi * r**2 * ( dphi**2 / 2 + self.Veff(phi, T) )
        # action = simpson(integrand, r)
        action = SFI.findAction(profile)

        return action / T

    def rate(self, T):
        if type(self.Veff) == VEffGeneric:
            S3_T = abs(self.bounce_action(T))
        return np.real(T**4 * power(abs(S3_T) / (2*pi), 3/2) * np.exp(-S3_T))
        # see Eq. (4.7) in https://arxiv.org/pdf/2210.07075.pdf
        
    # adimensional true-vacuum nucleation rate (false-vacuum decay rate)
    def decay_rate(self, tau):
        T = self.tau_to_temperature(tau)
        dimensionless_rate = self.rate(T) * power(self.c, 2)
        return dimensionless_rate

    # Vacuum energy fraction
    def I_frac(self, r, v0, v1, v2, v3):
        return (4*pi/3) * (power(r, 3)*v0 - 3*power(r, 2)*v1 + 3*r*v2 - v3)

    # Vacuum energy density
    def rhoV(self, tau, y):
        #return heaviside(self.tau_crit - tau, 0.0) \
        #     * (self.c*self.dV0 / (3*power(M_PL, 2))) * np.exp(-self.I_frac(y[5], y[1], y[2], y[3], y[4]))
        return (self.c*self.dV0 / (3*power(M_PL, 2))) * np.exp(-self.I_frac(y[5], y[1], y[2], y[3], y[4]))

    def drhoV_dtau(self, tau, y):
        v = (y[1], y[2], y[3], y[4])
        dv_dtau = (self.dv0_dtau(tau, y), self.dv1_dtau(tau, y), self.dv2_dtau(tau, y), self.dv3_dtau(tau, y))
        r, dr_dtau = y[5], self.dr_dtau(tau, y)
        
        drhoV_dtau = -self.rhoV(tau, y) * 4*pi/3 * (-3*v[2]*dr_dtau + r**3 * dv_dtau[0] + 3*r**2 *(v[0]*dr_dtau - dv_dtau[1]) - 3*r*(2*v[1]*dr_dtau + dv_dtau[2]) - dv_dtau[3])
        return drhoV_dtau


    def solve_system(self, tau_min=1e-2):
        # initial conditions at τ = tau_crit
        y0 = [1, 0, 0, 0, 0, 0,
            #   sqrt(self.c) * self.temperature_to_tau(1) * (-H_rad(self.Tc)*self.Tc)
        ]
        self.previous_dT_dt = sqrt(self.c) * self.temperature_to_tau(1) * (-sqrt(hubble2_rad(self.Tc))*self.Tc)

        def system_of_odes(tau, y):
            self.data[0].append((tau, y))
            dydt = np.zeros(6)
            dydt[0] = self.da_dtau(tau, y)
            dydt[1] = self.dv0_dtau(tau, y)
            dydt[2] = self.dv1_dtau(tau, y)
            dydt[3] = self.dv2_dtau(tau, y)
            dydt[4] = self.dv3_dtau(tau, y)
            dydt[5] = self.dr_dtau(tau, y)
            # dydt[6] = self.dTdt_dtau(tau, y)

            # once all values are computed, update stored dT_dt value for next step
            self.previous_dT_dt = self.dT_dt(tau, y)
            print(self.previous_dT_dt)

            return dydt
        
        # time span for the solution
        t_span = (self.tau_crit*1, tau_min*self.tau_crit)
        t_eval = np.linspace(t_span[0], t_span[1], 200)  # points to evaluate the solution


        return solve_ivp(system_of_odes, t_span, y0, t_eval=t_eval, method='RK45'
                        #  rtol=1e-8, atol=1e-8,
                        #  max_step=0.001,
                        #  jac=self.Jacobian
                         )
    


    def Jacobian(self, tau, y):
        # OUT OF DATE
        # supposedly helps with "stiff" systems

        a, v0, v1, v2, v3, r, dT_dt = y
        rhoR, drhoR_dtau = self.rhoR(tau), self.drhoR_dtau(tau)
        rhoV, drhoV_dtau = self.rhoV(tau, y), self.drhoV_dtau(tau, y)
        dT_dt = self.dT_dt(tau, y)
        Gamma = self.decay_rate(tau)

        J = np.zeros((6, 6))
        J[0, 0] = -(drhoR_dtau + drhoV_dtau)/4/rhoR
        J[1, 0] = 3 * Gamma * a**2 / dT_dt
        J[2, 0] = 3 * Gamma * a**2 * r / dT_dt
        J[3, 0] = 3 * Gamma * a**2 * r**2 / dT_dt
        J[4, 0] = 3 * Gamma * a**2 * r**3 / dT_dt
        J[5, 0] = -self.vw/a**2 / dT_dt

        J[2, 5] = Gamma * a**3 / dT_dt
        J[3, 5] = 2 * Gamma * a**3 * r / dT_dt
        J[4, 5] = 3 * Gamma * a**3 * r**2 / dT_dt

        return J
    # OUTPUT: give numpy arrays of a(T), dT/dt(T), ρ(T)