# Unit tests for PBH formation from bubble nucleation
#
# Copyright (c) 2025 Adrian Thompson via MIT License

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from pypt.ftpot import VEffGeneric
from pypt.bubble_nucleation import BubbleNucleationQuartic
from pypt.bgg_fks_collapse import FKSCollapse, get_pbh_abundance
from pypt.cosmology_functions import hubble2_rad, gstar_sm, temp_to_time, a_ratio_rad
from pypt.constants import *


def test_pbh_formation(a=1/16, lam=0.061, c=0.1052, d=2.725, vev=1.0, 
                       gstar_D=1.0, verbose=True):
    """
    Test PBH formation calculation with user-specified potential parameters.
    
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
    gstar_D : float
        Dark sector degrees of freedom
    verbose : bool
        Print detailed output
    """
    print("=" * 70)
    print("PBH FORMATION TEST")
    print("=" * 70)
    print(f"\nInput parameters:")
    print(f"  a      = {a}")
    print(f"  lam    = {lam}")
    print(f"  c      = {c}")
    print(f"  d      = {d}")
    print(f"  vev    = {vev}")
    print(f"  gstar_D = {gstar_D}")
    print()

    # =========================================================================
    # Step 1: Create VEffGeneric potential
    # =========================================================================
    print("-" * 50)
    print("Step 1: Creating VEffGeneric potential...")
    print("-" * 50)
    
    try:
        veff = VEffGeneric(a=a, lam=lam, c=c, d=d, vev=vev, verbose=verbose)
        print(f"  T0^2 = {veff.T0sq:.6e}")
        print(f"  VEV  = {veff.vev:.6e}")
        print(f"  Tc   = {veff.Tc:.6e}")
        
        fopt_possible = veff.Tc is not None and veff.Tc > 0 and np.isfinite(veff.Tc)
        print(f"\n  FOPT possible: {fopt_possible}")
        
        if not fopt_possible:
            print("  ERROR: No first-order phase transition found!")
            return None, None, None
            
    except Exception as e:
        print(f"  ERROR creating potential: {e}")
        return None, None, None

    # =========================================================================
    # Step 2: Create BubbleNucleationQuartic and extract parameters
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 2: Creating BubbleNucleationQuartic...")
    print("-" * 50)
    
    try:
        bn = BubbleNucleationQuartic(veff, verbose=verbose, assume_rad_dom=True, gstar_D=gstar_D)
        
        Tstar = bn.Tstar
        Tperc = bn.Tperc
        tperc = bn.tperc
        alpha = bn.alpha()
        betaByHstar = bn.betaByHstar()
        vw = bn.vw()
        
        # Hubble rate at Tstar
        H_star = np.sqrt(bn.hubble_rate_sq(Tstar))
        
        print(f"\n  Phase Transition Parameters:")
        print(f"    Tstar       = {Tstar:.6e} GeV")
        print(f"    Tperc       = {Tperc:.6e} GeV")
        print(f"    tperc       = {tperc:.6e} GeV^-1")
        print(f"    alpha       = {alpha:.6e}")
        print(f"    beta/H*     = {betaByHstar:.6e}")
        print(f"    vw          = {vw:.6f}")
        print(f"    H(Tstar)    = {H_star:.6e} GeV")
        print(f"    S3/T(Tstar) = {bn.bounce_action(Tstar):.4f}")
        print(f"    S3/T(Tperc) = {bn.bounce_action(Tperc):.4f}")
        
    except Exception as e:
        print(f"  ERROR creating BubbleNucleationQuartic: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

    # =========================================================================
    # Step 3: Create FKSCollapse object
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 3: Creating FKSCollapse object...")
    print("-" * 50)
    
    # Compute deltaV and wall tension from the potential
    phi_plus = bn.phi_plus
    deltaV = -veff(phi_plus, Tperc)  # Potential difference (positive)
    sigma = veff.wall_tension()      # Wall tension
    
    print(f"\n  Collapse Parameters:")
    print(f"    phi_+     = {phi_plus:.6e} GeV")
    print(f"    deltaV    = {deltaV:.6e} GeV^4")
    print(f"    sigma     = {sigma:.6e} GeV^3")
    
    try:
        fks = FKSCollapse(deltaV=deltaV, sigma=sigma, vw=vw)
        
        print(f"\n  FKS Derived Parameters:")
        print(f"    eta       = {fks.eta:.6e}")
        print(f"    gamma     = {fks.gamma:.6e}")
        print(f"    H_V^2     = {fks.HV2:.6e} GeV^2")
        print(f"    H_sigma^2 = {fks.Hsigma2:.6e} GeV^2")
        
    except Exception as e:
        print(f"  ERROR creating FKSCollapse: {e}")
        return bn, None, None

    # =========================================================================
    # Step 4: Define false vacuum radius ansatz
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 4: False vacuum radius ansatz...")
    print("-" * 50)
    
    # Ansatz: r_fv = vw / (H* * beta/H*)
    r_fv = vw / (H_star * betaByHstar)
    
    print(f"\n  r_fv = vw / (H* * beta/H*)")
    print(f"       = {vw:.4f} / ({H_star:.4e} * {betaByHstar:.4e})")
    print(f"       = {r_fv:.6e} GeV^-1")
    
    # Check if PBH forms at this radius
    does_form = fks.does_pbh_form(r_fv)
    print(f"\n  Does PBH form at r_fv? {does_form}")
    
    if does_form:
        m_pbh = fks.M0(r_fv)
        print(f"  M_PBH(r_fv) = {m_pbh:.6e} GeV")

    # =========================================================================
    # Step 5: Plot p_surv_false_vacuum vs r
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 5: Plotting p_surv_false_vacuum(r)...")
    print("-" * 50)
    
    r_min = 0.001 * r_fv
    r_max = 2.0 * r_fv
    n_points = 50
    
    r_vals = np.linspace(r_min, r_max, n_points)
    p_surv_vals = []
    
    print(f"\n  Computing p_surv from r = {r_min:.4e} to {r_max:.4e} GeV^-1...")
    
    for r in r_vals:
        try:
            p_surv = bn.p_surv_false_vacuum(r)
            p_surv_vals.append(p_surv)
        except Exception as e:
            if verbose:
                print(f"    Warning at r={r:.4e}: {e}")
            p_surv_vals.append(np.nan)
    
    p_surv_vals = np.array(p_surv_vals)
    
    # Get p_surv at r_fv
    p_surv_at_r_fv = bn.p_surv_false_vacuum(r_fv)
    print(f"\n  p_surv(r_fv) = {p_surv_at_r_fv:.6e}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    valid_mask = np.isfinite(p_surv_vals) & (p_surv_vals > 0)
    if np.any(valid_mask):
        ax.semilogy(r_vals[valid_mask] / r_fv, p_surv_vals[valid_mask], 'b-', linewidth=2)
    else:
        ax.plot(r_vals / r_fv, p_surv_vals, 'b-', linewidth=2)
    
    ax.axvline(x=1.0, color='r', linestyle='--', label=f'r_fv = {r_fv:.2e} GeV^-1')
    ax.axhline(y=p_surv_at_r_fv, color='g', linestyle=':', alpha=0.7, 
               label=f'p_surv(r_fv) = {p_surv_at_r_fv:.2e}')
    
    ax.set_xlabel(r'$r / r_{fv}$', fontsize=12)
    ax.set_ylabel(r'$P_{surv}(r)$', fontsize=12)
    ax.set_title('False Vacuum Survival Probability vs Patch Radius', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # =========================================================================
    # Step 6: Report individual terms in get_pbh_abundance
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 6: PBH Abundance Calculation Breakdown...")
    print("-" * 50)
    
    tc = bn.tc
    
    # Compute each term individually
    print("\n  Computing individual terms in get_pbh_abundance:")
    
    # 1. Normalization term
    normalization = np.power(HUBBLE, 3) / (4 * np.pi * RHO_CRIT_GEV4 * OMEGA_DM / 3)
    print(f"\n  1. Normalization term:")
    print(f"     HUBBLE = {HUBBLE:.6e} GeV")
    print(f"     RHO_CRIT = {RHO_CRIT_GEV4:.6e} GeV^4")
    print(f"     OMEGA_DM = {OMEGA_DM:.6f}")
    print(f"     normalization = H^3 / (4*pi*rho_crit*Omega_DM/3)")
    print(f"                   = {normalization:.6e}")
    
    # 2. p_surv_false_vacuum(r_fv)
    p_surv = p_surv_at_r_fv
    print(f"\n  2. p_surv_false_vacuum(r_fv):")
    print(f"     p_surv = {p_surv:.6e}")
    
    # 3. xi term
    H2_Tperc = bn.hubble_rate_sq(Tperc)
    a_ratio = a_ratio_rad(tc, tperc)
    xi = r_fv * H2_Tperc * a_ratio
    print(f"\n  3. xi term:")
    print(f"     H^2(Tperc) = {H2_Tperc:.6e} GeV^2")
    print(f"     a(tc)/a(tperc) = {a_ratio:.6e}")
    print(f"     xi = r_fv * H^2(Tperc) * a_ratio")
    print(f"        = {r_fv:.4e} * {H2_Tperc:.4e} * {a_ratio:.4e}")
    print(f"        = {xi:.6e}")
    
    # 4. N_patches term
    gstar_ratio = gstar_sm(0.0) / gstar_sm(Tperc)
    H2_ratio = bn.hubble_rate_sq(Tperc) / HUBBLE
    T_ratio = T0_SM / Tperc
    
    Npatches = np.power(xi, -3) * gstar_ratio * np.power(T_ratio * H2_ratio, 3)
    print(f"\n  4. N_patches term:")
    print(f"     xi^-3 = {np.power(xi, -3):.6e}")
    print(f"     g*(T0)/g*(Tstar) = {gstar_ratio:.6f}")
    print(f"     T0_SM = {T0_SM:.6e} GeV")
    print(f"     H^2(Tperc)/H_today = {H2_ratio:.6e}")
    print(f"     N_patches = xi^-3 * (g* ratio) * (T0*H^2(Tperc)/(Tstar*H))^3")
    print(f"               = {Npatches:.6e}")
    
    # 5. PBH mass
    if does_form:
        m_pbh = fks.M0(r_fv)
        print(f"\n  5. PBH mass at r_fv:")
        print(f"     M_PBH = {m_pbh:.6e} GeV")
        print(f"           = {m_pbh * 1.783e-24:.6e} grams")  # GeV to grams
        
        # Final abundance
        abundance = normalization * m_pbh * p_surv * Npatches
        print(f"\n  6. Final PBH abundance:")
        print(f"     Omega_PBH = normalization * M_PBH * p_surv * N_patches")
        print(f"              = {normalization:.4e} * {m_pbh:.4e} * {p_surv:.4e} * {Npatches:.4e}")
        print(f"              = {abundance:.6e}")
        
        # Also compute using the function directly
        abundance_func = get_pbh_abundance(m_pbh, r_fv, bn)
        print(f"\n     get_pbh_abundance() = {abundance_func:.6e}")
    else:
        print(f"\n  5. PBH mass: N/A (no collapse at r_fv)")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    return bn, fks, r_fv


if __name__ == "__main__":
    # Default test parameters - modify these as needed
    params = {
        'a': 0.0626,
        'lam': 0.275,
        'c': 0.1052,
        'd': 2.725,
        'vev': 1.0,
        'gstar_D': 1.0,
    }
    
    print("\nRunning PBH formation test with default parameters...")
    print("Modify the 'params' dictionary in __main__ to test different values.\n")
    
    bn, fks, r_fv = test_pbh_formation(**params, verbose=True)
