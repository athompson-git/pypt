# plotting macros for visualizing parameter scans
# 
# Copyright (c) 2025 Adrian Thompson via MIT License

import json
import matplotlib.pyplot as plt
import numpy as np
from .constants import *

from matplotlib.colors import LogNorm

from .gw import *

"""
Make a 6-plot 'cornerplot' over the parameters for a quartic potential as scatterplots
color-coded by a parameter string of choice.
"""
def corner_plots_quartic_potential(json_filepath, parameter_to_colorcode="Tc"):

    fig = plt.figure(constrained_layout=True, figsize=[10.0, 10.0])
    spec = fig.add_gridspec(3, 3)


    with open(json_filepath, "r") as file:
        param_json = json.load(file)

    a_list = []
    d_list = []
    c_list = []
    lam_list = []
    color_param_list = []

    for i in range(len(param_json)):
        p = param_json[i]
        d_param = p["d"]
        c_param = p["c"]
        a_param = p["a"]
        lam_param = p["lambda"]

        Tc = p["Tc"]
        alpha = p["alpha"]
        betaByHstar = p["betaByHstar"]
        vw = p["v_wall"]
        f_peak = p["f_peak"]
        mpbh = p["MPBH"]
        
        if Tc is None:
            continue
        if alpha < 0:
            continue
        """
        if alpha > 1.0:
            continue
        """
        if vw is None:
            continue
        if betaByHstar is None or betaByHstar == 0.0:
            continue
        if f_peak is None:
            continue
        """
        if betaByHstar > 1e8:
            continue
        """
        if mpbh is None:
            continue
        """
        if fpbh <= 0.0:
            continue
        if fpbh >= 1.0:
            continue
        """

        color_param = p[parameter_to_colorcode]

        a_list.append(a_param)
        d_list.append(d_param)
        c_list.append(c_param)
        lam_list.append(lam_param)
        color_param_list.append(color_param)

    param_choice = np.array(color_param_list)
    param_min = (min(param_choice))
    param_max = (max(param_choice))

    def get_color_log(alpha):
        ln_alpha = np.log10(alpha)
        return (ln_alpha - np.log10(param_min))/(np.log10(param_max) - np.log10(param_min))
    
    def get_color(alpha):
        return (alpha - param_min)/(param_max - param_min)
    
    color_ids = get_color_log(param_choice)
    colors = plt.cm.viridis(color_ids)

    ax1 = fig.add_subplot(spec[0, 0])
    ax1.scatter(a_list, d_list, marker=".", c=colors, alpha=0.8)
    ax1.set_ylabel(r"$D$", fontsize=14)
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax2 = fig.add_subplot(spec[1, 0])
    ax2.scatter(a_list, c_list, marker=".", c=colors, alpha=0.8)
    ax2.set_ylabel(r"$C/\langle \phi \rangle$", fontsize=14)
    ax2.set_yscale('log')
    ax2.set_xscale('log')

    ax3 = fig.add_subplot(spec[1, 1])
    ax3.scatter(d_list, c_list, marker=".", c=colors, alpha=0.8)
    ax3.set_yscale('log')
    ax3.set_xscale('log')

    ax4 = fig.add_subplot(spec[2, 0])
    ax4.scatter(a_list, lam_list, marker=".", c=colors, alpha=0.8)
    ax4.set_ylabel(r"$\lambda$", fontsize=14)
    ax4.set_xlabel(r"$A$", fontsize=14)
    ax4.set_yscale('log')
    ax4.set_xscale('log')

    ax5 = fig.add_subplot(spec[2, 1])
    ax5.scatter(d_list, lam_list, marker=".", c=colors, alpha=0.8)
    ax5.set_xlabel(r"$D$", fontsize=14)
    ax5.set_yscale('log')
    ax5.set_xscale('log')

    ax6 = fig.add_subplot(spec[2, 2])
    ax6.scatter(c_list, lam_list, marker=".", c=colors, alpha=0.8)
    ax6.set_xlabel(r"$C/\langle \phi \rangle$", fontsize=14)
    ax6.set_yscale('log')
    ax6.set_xscale('log')

    cbar_ax = fig.add_axes([0.8, 0.4, 0.03, 0.6])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_clim(vmin=param_min, vmax=param_max)
    cbar = fig.colorbar(sm, cax=cbar_ax)

    plt.show()




"""
Plot a 1D histogram from a json using a parameter string.
"""
def plot_pbh_hist1d(json_filepath, varstr="MPBH", label=r"$M_{PBH}$ [g]"):

    with open(json_filepath, "r") as file:
        param_json = json.load(file)

    var_list = []


    for i in range(len(param_json)):
        p = param_json[i]

        var = p[varstr]
        if var is None:
            continue
        if var <= 0.0:
            continue
        
        var_list.append(var)


    min_mass = min(var_list)
    max_mass = max(var_list)
    mass_bins = np.logspace(np.log10(min_mass), np.log10(max_mass), 50)
    plt.hist(var_list, bins=mass_bins, histtype='step')
    plt.xscale('log')
    plt.ylabel(r"Density of Model Points", fontsize=16)
    plt.xlabel(label, fontsize=16)
    plt.tight_layout()
    plt.show()




"""
Pass two known parameter strings from the json of interest and plot a 2D scatterplot
color coded by a third parameter string choice color_param.

If passing MPBH, automatically rescales to grams.
"""
def plot_2d(json_filepath, varstr1="MPBH", varstr2 = "fBPH",
            xlabel=r"$M_{PBH}$ [g]", ylabel=r"$f_{PBH}$",
            ylim=None, xlim=None, color_param="v_wall", color_label="$v_w$",
            color_log=False):

    with open(json_filepath, "r") as file:
        param_json = json.load(file)

    var1_list = []
    var2_list = []
    colvar_list = []

    gw = GravitationalWave()

    for i in range(len(param_json)):
        p = param_json[i]

        var1 = p[varstr1]
        var2 = p[varstr2]
        colvar = p[color_param]

        if var1 is None:
            continue
        if var2 is None:
            continue
        if var1 <= 0.0:
            continue
        if var2 <= 0.0:
            continue
        if colvar is None or colvar <= 0.0:
            continue
        
        if varstr1 == "MPBH":
            var1 *= 1/GEV_PER_G
        if varstr2 == "MPBH":
            var2 *= 1/GEV_PER_G
        
        if (varstr2 == "f_peak") or (varstr1 == "f_peak"):
            gw.alpha = p["alpha"]
            gw.betaByHstar = p["betaByHstar"]
            gw.vw = p["v_wall"]
            gw.Tstar = p["Tstar"]
            var2 = gw.f_peak_col()

        var1_list.append(var1)
        var2_list.append(var2)
        colvar_list.append(colvar)

    param_choice = np.array(colvar_list)
    param_min = (min(param_choice))
    param_max = (max(param_choice))

    def get_color(alpha):
        if color_log:
            return (np.log10(alpha) - np.log10(param_min))/(np.log10(param_max) - np.log10(param_min))
        return (alpha - param_min)/(param_max - param_min)
    
    color_ids = get_color(param_choice)
    colors = plt.cm.viridis(color_ids)

    if color_log:
        plt.scatter(var1_list, var2_list, c=colors, norm=LogNorm())
    else:
        plt.scatter(var1_list, var2_list, c=colors, norm=LogNorm())

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_clim(vmin=param_min, vmax=param_max)
    cbar = plt.colorbar(sm)
    cbar.set_label(color_label)

    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.show()


