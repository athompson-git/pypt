# Unit tests for PBH / bubble nucleation calculations
#
# Copyright (c) 2025 Adrian Thompson via MIT License

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from pypt.ftpot import VEffGeneric
from pypt.bubble_nucleation import BubbleNucleationQuartic


def test_bubble_nucleation(a=0.1, lam=0.061, c=0.249, d=0.596, vev=100.0, verbose=True):
    """
    Test the bubble nucleation calculation with user-specified parameters.
    
    Parameters
    ----------
    a : float
        Cubic temperature coefficient
    lam : float
        Quartic coupling
    c : float
        Cubic coupling (temperature-independent part)
    d : float
        Quadratic temperature coefficient
    vev : float
        Vacuum expectation value at T=0
    verbose : bool
        Print detailed output
    """
    print("=" * 60)
    print("BUBBLE NUCLEATION TEST")
    print("=" * 60)
    print(f"\nInput parameters:")
    print(f"  a   = {a}")
    print(f"  lam = {lam}")
    print(f"  c   = {c}")
    print(f"  d   = {d}")
    print(f"  vev = {vev}")
    print()

    # Step 1: Create the generic quartic potential
    print("-" * 40)
    print("Step 1: Creating VEffGeneric potential...")
    print("-" * 40)
    
    try:
        veff = VEffGeneric(a=a, lam=lam, c=c, d=d, vev=vev, verbose=verbose)
        print(f"  T0^2 = {veff.T0sq:.4e}")
        print(f"  VEV  = {veff.vev:.4e}")
        print(f"  Tc   = {veff.Tc:.4e}")
        
        # Check if FOPT is possible
        fopt_possible = veff.Tc is not None and veff.Tc > 0 and np.isfinite(veff.Tc)
        print(f"\n  FOPT possible: {fopt_possible}")
        
        if not fopt_possible:
            print("  ERROR: No first-order phase transition found!")
            return None
            
    except Exception as e:
        print(f"  ERROR creating potential: {e}")
        return None

    # Step 2: Create BubbleNucleationQuartic
    print("\n" + "-" * 40)
    print("Step 2: Creating BubbleNucleationQuartic...")
    print("-" * 40)
    
    try:
        bn = BubbleNucleationQuartic(veff, verbose=verbose, assume_rad_dom=True, gstar_D=1.0)
        
        print(f"\n  Tstar = {bn.Tstar:.4e}")
        print(f"  Tperc = {bn.Tperc:.4e}")
        print(f"  tperc = {bn.tperc:.4e} s")
        print(f"  S3/T at Tstar = {bn.bounce_action(bn.Tstar):.4f}")
        print(f"  S3/T at Tperc = {bn.bounce_action(bn.Tperc):.4f}")
        
        # Compute derived quantities
        print(f"\n  alpha (latent heat) = {bn.alpha():.4e}")
        print(f"  beta/H* = {bn.betaByHstar():.4e}")
        print(f"  vw (wall velocity) = {bn.vw():.4f}")
        
    except ValueError as e:
        print(f"  ERROR: {e}")
        print("\n  Proceeding to plot fv_exponent to diagnose...")
        bn = None
    except Exception as e:
        print(f"  ERROR creating BubbleNucleationQuartic: {e}")
        return None

    # Step 3: Plot fv_exponent
    print("\n" + "-" * 40)
    print("Step 3: Plotting fv_exponent...")
    print("-" * 40)
    
    plot_fv_exponent(veff, bn, verbose=verbose)
    
    return bn


def plot_fv_exponent(veff, bn=None, n_points=100, verbose=True):
    """
    Plot the false vacuum exponent I(T) and check where exp(-I(T)) = 0.7.
    
    Parameters
    ----------
    veff : VEffGeneric
        The effective potential
    bn : BubbleNucleationQuartic or None
        The bubble nucleation object (if available)
    n_points : int
        Number of temperature points to sample
    verbose : bool
        Print detailed output
    """
    # Create a temporary BubbleNucleationQuartic just for computing fv_exponent
    # if bn is None (e.g., if Tperc wasn't found)
    if bn is None:
        # Create a minimal object to compute fv_exponent
        # We need to set up the required attributes manually
        class TempBN:
            def __init__(self, veff):
                self.veff = veff
                self.Tc = veff.Tc
                self.verbose = verbose
                self.a = veff.a
                self.c = veff.c
                self.d = veff.d
                self.lam = veff.lam
                self.T0sq = veff.T0sq
                self.vev = veff.vev
                self.gstar_D = 1.0
                self.use_fks_action = False
                
                # Compute Tstar first
                self.Tstar = self._get_Tstar()
                
                # Compute vw and alpha at Tstar
                self._phi_plus_star = self.veff.get_vev(self.Tstar)
                self._alpha_star = self._compute_alpha(self.Tstar, self._phi_plus_star)
                self._vw_star = self._compute_vw(self.Tstar, self._phi_plus_star, self._alpha_star)
            
            def _get_Tstar(self):
                from pypt.cosmology_functions import hubble2_rad, gstar_sm
                T_grid = np.linspace(np.sqrt(abs(self.T0sq)), self.Tc, 100000)
                GammaByHstar = np.nan_to_num([self.rate(T)/np.power(self.hubble_rate_sq(T),2) for T in T_grid])
                star_id = np.argmin(abs(GammaByHstar - 1.0))
                return T_grid[star_id]
            
            def bounce_action(self, T):
                delta = 8*self.veff.a4(T) * self.veff.a2(T) / self.veff.a3(T)**2
                beta1, beta2, beta3 = 8.2938, -5.5330, 0.8180
                return np.clip((-np.pi * self.veff.a3(T) * 8*np.sqrt(2)*np.power(2 - delta, -2) \
                    *np.sqrt(abs(delta)/2) \
                    * (beta1*delta + beta2*delta**2 + beta3*delta**3) \
                    / np.power(self.veff.a4(T), 1.5) / 81 / T), a_min=0.0, a_max=np.inf)
            
            def rate(self, T):
                return np.real(T**4 * np.power(abs(self.bounce_action(T)) / (2*np.pi), 3/2) \
                    * np.exp(-abs(self.bounce_action(T))))
            
            def hubble_rate_sq(self, T):
                from pypt.cosmology_functions import hubble2_rad, gstar_sm
                from pypt.constants import M_PL
                h2_rad = hubble2_rad(T, gstar=gstar_sm(T)+self.gstar_D)
                phic = self.veff.get_vev(T)
                h2_vac = (1/3/M_PL**2) * (-self.veff(phic, T))
                h2_total = h2_rad + h2_vac
                return abs(h2_total) if h2_total < 0 else h2_total
            
            def _compute_alpha(self, T, phi_plus):
                from pypt.constants import GSTAR_SM
                prefactor = 30 / np.pi**2 / GSTAR_SM / T**4
                deltaV = -self.veff(phi_plus, T)
                dVdT_val = 2*self.d*T*phi_plus**2 - self.a*phi_plus**3
                return prefactor * (deltaV + T * dVdT_val / 4)
            
            def _compute_vw(self, T, phi_plus, alpha_val):
                from pypt.constants import GSTAR_SM
                if not np.isfinite(alpha_val) or alpha_val <= 0:
                    return 1.0
                deltaV = -self.veff(phi_plus, T)
                vJ = (np.sqrt(2*alpha_val/3 + alpha_val**2) + np.sqrt(1/3))/(1+alpha_val)
                rho_r = np.pi**2 * GSTAR_SM * T**4 / 30
                denom = alpha_val * rho_r
                if denom <= 0 or deltaV < 0:
                    return 1.0
                v_candidate = np.sqrt(deltaV / denom)
                return v_candidate if v_candidate < vJ else 1.0
            
            def R_bubble_temperature(self, Tprime, T, n_samples=100):
                T_vals = np.linspace(T, Tprime, n_samples)
                dT = T_vals[1] - T_vals[0]
                gamma = 1.0
                hubble_sq_vals = np.array([self.hubble_rate_sq(T_) for T_ in T_vals])
                hubble_sq_vals = np.maximum(hubble_sq_vals, 1e-100)
                hubble_vals = np.sqrt(hubble_sq_vals)
                integrands = self._vw_star * np.power(self.Tc / T_vals, -1/gamma) / (T_vals * hubble_vals * gamma) * dT
                return np.sum(integrands)
            
            def fv_exponent(self, T, n_samples=100):
                if T <= 0 or not np.isfinite(T):
                    return np.inf
                Tprime_vals = np.linspace(T, self.Tc, n_samples)
                dT = Tprime_vals[1] - Tprime_vals[0]
                gamma = 1.0
                r_bubble = np.array([self.R_bubble_temperature(Tprime, T) for Tprime in Tprime_vals])
                rates = self.rate(Tprime_vals)
                hubble_sq_vals = np.array([self.hubble_rate_sq(Tprime) for Tprime in Tprime_vals])
                hubble_sq_vals = np.maximum(hubble_sq_vals, 1e-100)
                hubble_vals = np.sqrt(hubble_sq_vals)
                integrands = (4*np.pi/3) * rates * np.power(r_bubble, 3) \
                    * np.power(self.Tc / T, 3/gamma) / (Tprime_vals * gamma * hubble_vals) * dT
                result = np.sum(integrands)
                return np.inf if not np.isfinite(result) else result
        
        bn_temp = TempBN(veff)
    else:
        bn_temp = bn
    
    # Temperature range: from Tc down to T0 (or 1% of Tc)
    T_min = max(1e-4 * veff.Tc, np.sqrt(abs(veff.T0sq)) * 1.01) if veff.T0sq > 0 else 1e-4 * veff.Tc
    T_max = veff.Tc * 0.999  # slightly below Tc
    
    T_vals = np.linspace(T_min, T_max, n_points)
    
    print(f"  Temperature range: [{T_min:.4e}, {T_max:.4e}]")
    print(f"  Computing fv_exponent at {n_points} points...")
    
    fv_exp_vals = []
    p_fv_vals = []
    
    for i, T in enumerate(T_vals):
        try:
            fv_exp = bn_temp.fv_exponent(T)
            fv_exp_vals.append(fv_exp)
            p_fv = np.exp(-fv_exp) if np.isfinite(fv_exp) else 0.0
            p_fv_vals.append(p_fv)
        except Exception as e:
            if verbose:
                print(f"  Warning at T={T:.4e}: {e}")
            fv_exp_vals.append(np.nan)
            p_fv_vals.append(np.nan)
    
    fv_exp_vals = np.array(fv_exp_vals)
    p_fv_vals = np.array(p_fv_vals)
    
    # Find where p_fv crosses 0.7
    crossing_indices = np.where(np.diff(np.sign(p_fv_vals - 0.7)))[0]
    
    print(f"\n  Results:")
    print(f"  p_fv at T_max (near Tc): {p_fv_vals[-1]:.4f}")
    print(f"  p_fv at T_min: {p_fv_vals[0]:.4f}")
    
    if len(crossing_indices) > 0:
        for idx in crossing_indices:
            T_cross = 0.5 * (T_vals[idx] + T_vals[idx+1])
            print(f"  p_fv = 0.7 crossing at T ~ {T_cross:.4e}")
    else:
        if p_fv_vals[-1] < 0.7:
            print("  WARNING: p_fv never reaches 0.7 - percolation completes before Tc")
        elif p_fv_vals[0] > 0.7:
            print("  WARNING: p_fv always > 0.7 - percolation never completes in range")
        else:
            print("  No clear crossing of p_fv = 0.7 found")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot 1: fv_exponent vs T
    ax1 = axes[0, 0]
    valid_mask = np.isfinite(fv_exp_vals)
    ax1.loglog(T_vals[valid_mask], fv_exp_vals[valid_mask], 'b-', linewidth=2)
    ax1.axhline(y=-np.log(0.7), color='r', linestyle='--', label='I(T) for p_fv=0.7')
    ax1.set_xlabel('Temperature T')
    ax1.set_ylabel('I(T) = fv_exponent')
    ax1.set_title('False Vacuum Exponent I(T)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: p_fv = exp(-I) vs T
    ax2 = axes[0, 1]
    ax2.loglog(T_vals, p_fv_vals, 'b-', linewidth=2)
    ax2.axhline(y=0.7, color='r', linestyle='--', label='p_fv = 0.7 (percolation)')
    ax2.set_xlabel('Temperature T')
    ax2.set_ylabel('p_fv = exp(-I(T))')
    ax2.set_title('False Vacuum Fraction')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Mark Tstar if available
    if bn_temp is not None and hasattr(bn_temp, 'Tstar'):
        ax2.axvline(x=bn_temp.Tstar, color='g', linestyle=':', label=f'Tstar={bn_temp.Tstar:.2e}')
        ax2.legend()
    
    # Plot 3: Bounce action S3/T vs T
    ax3 = axes[1, 0]
    S3_T_vals = [bn_temp.bounce_action(T) for T in T_vals]
    ax3.semilogy(T_vals, S3_T_vals, 'b-', linewidth=2)
    ax3.axhline(y=140, color='r', linestyle='--', label='S3/T = 140')
    ax3.set_xlabel('Temperature T')
    ax3.set_ylabel('S3/T')
    ax3.set_title('Bounce Action')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Nucleation rate vs T
    ax4 = axes[1, 1]
    rate_vals = [bn_temp.rate(T) for T in T_vals]
    ax4.semilogy(T_vals, rate_vals, 'b-', linewidth=2)
    ax4.set_xlabel('Temperature T')
    ax4.set_ylabel('Γ(T)')
    ax4.set_title('Nucleation Rate')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return T_vals, fv_exp_vals, p_fv_vals


if __name__ == "__main__":
    # Default test parameters - modify these as needed
    params = {
        'a': 0.0626,
        'lam': 0.275,
        'c': 0.1052,
        'd': 2.725,
        'vev': 1.0,
    }
    
    print("\nRunning bubble nucleation test with default parameters...")
    print("Modify the 'params' dictionary in __main__ to test different values.\n")
    
    bn = test_bubble_nucleation(**params, verbose=True)
