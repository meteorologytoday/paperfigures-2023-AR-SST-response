import xarray as xr
import traceback
from pathlib import Path
import argparse
from datetime import (timedelta, datetime, timezone)
from scipy.stats import ttest_ind_from_stats
import numpy as np
import pandas as pd
import tool_fig_config

parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--input-dir', type=str, help='Input file', required=True)
parser.add_argument('--lat-rng', type=float, nargs=2, help='Latitude range', required=True)
parser.add_argument('--lon-rng', type=float, nargs=2, help='Longitude range', required=True)
parser.add_argument('--output', type=str, help='Output file', default="")
parser.add_argument('--AR-algo', type=str, help='Algorithm of making AR object', default="ANOM_LEN")
parser.add_argument('--title', type=str, help='Output title', default="")
parser.add_argument('--title-style', type=str, help='Output title', default="folder", choices=["folder", "latlon"])
parser.add_argument('--varnames', type=str, help='varnames you want to do stats. X and Y.', nargs=2)
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()
print(args)


print("Loading data...")

engine='netcdf4'
ds_anom = xr.open_dataset("%s/anom.nc" % (args.input_dir,), engine=engine)
ds_ttl  = xr.open_dataset("%s/ttl.nc"  % (args.input_dir,), engine=engine)
   
args.lon_rng = np.array(args.lon_rng) % 360.0

print("Selecting data range : lat = [%.2f , %.2f], lon = [%.2f, %.2f]" % (*args.lat_rng, *args.lon_rng))
print(ds_anom.coords["lat"].to_numpy())
print(ds_anom.coords["lon"].to_numpy())
latlon_sel = (
    (ds_anom.coords["lat"] >= args.lat_rng[0])
    & (ds_anom.coords["lat"] <= args.lat_rng[1])
    & (ds_anom.coords["lon"] >= args.lon_rng[0])
    & (ds_anom.coords["lon"] <= args.lon_rng[1])
)

AR_sel = ds_ttl['map_%s' % (args.AR_algo,)] > 0
#extra_sel = (ds_anom.time < pd.Timestamp('2016-05-01')) & (ds_anom.time > pd.Timestamp('2014-09-01')) 

total_sel = latlon_sel & AR_sel #& extra_sel

print("Number of selected data points: %d " % ( np.sum(total_sel), ) )



factor_X = 1e-6
factor_Y = 1e-6

#varname_x = "dMLTdt"
#varname_y = "MLG_frc"

#varname_x = "MLG_frc"
#varname_y = "MLG_nonfrc"

varname_x = args.varnames[0]
varname_y = args.varnames[1]

print("Varname X: ", varname_x)
print("Varname Y: ", varname_y)

ds_anom = ds_anom.where(total_sel)#.where(np.abs(ds_anom[varname_x]) < 2e-6)

if varname_y == "dMLDdt":
    factor_Y = 1e-4

data_x = ds_anom[varname_x].to_numpy().flatten() / factor_X
data_y = ds_anom[varname_y].to_numpy().flatten() / factor_Y

valid_idx = np.isfinite(data_x) & np.isfinite(data_y)
data_x = data_x[valid_idx]
data_y = data_y[valid_idx]

corr_coe = np.corrcoef(data_x, data_y)
coe = np.polyfit(data_x, data_y, deg=1)

print("corr_coe = ", corr_coe)
print("coe = ", coe)

print("===== Error propagation begin =====")
# Use error propagation to estimate the 
# uncertainty of slope

cnt_valid_points = np.sum(valid_idx)
print("cnt_valid_points = %d" % (cnt_valid_points,))

# First estimate the standard deviation of y
estimator_y = coe[0] * data_x + coe[1]
var_y = np.var(data_y - estimator_y, ddof=1)


# Compute the uncertainty
var_x = np.var(data_x, ddof=0)
std_slope = (var_y / cnt_valid_points / var_x)**0.5
std_inter = (var_y * np.sum(data_x * data_x) / cnt_valid_points**2 / var_x)**0.5

print("std(y-y_est) = ", var_y**0.5)
print("Best fit slope = ", coe[0])
print("std(slope) = ", std_slope)
print("std(inter) = ", std_inter)


print("===== Error propagation end =====")






#fit_slope, fit_const = np.linalg.lstsq(A, y, rcond=None)[0]


print(data_x.shape)

edges_x = np.arange(-1.5, 1.52, 0.02)#np.linspace(-1, 1, 101) * 1.5
edges_y = np.arange(-1.5, 1.52, 0.02)#np.linspace(-1, 1, 101) * 1.5

hist, edges_x, edges_y = np.histogram2d(data_x, data_y, bins=[edges_x, edges_y], density=True)

mid_x = ( edges_x[:-1] + edges_x[1:] ) / 2
mid_y = ( edges_y[:-1] + edges_y[1:] ) / 2

hist_std = np.std(hist)
hist_color_lev_max = 1.5 #hist_std * 4
print("Maximum of hist : ", np.amax(hist))
print("Std dev hist : ", hist_std)
print("Decide the max = ", hist_color_lev_max)

# compute mass centers
mass_center = np.zeros_like(mid_x)
uncertainty_lower = np.zeros_like(mid_x)
uncertainty_upper = np.zeros_like(mid_x)
for i in range(len(mid_x)):
    wgt = hist[i, :]**1
    if np.all(wgt == 0):
        mass_center[i] = np.nan
        uncertainty_upper[i] = np.nan
        uncertainty_lower[i] = np.nan

    else:

        #mass_center[i] = np.average(mid_y, weights=hist[i, :]**1)


        idx = (edges_x[i] < data_x) & ( data_x <= edges_x[i+1] )
        q = np.quantile(data_y[idx], [0.25, 0.50, 0.75])
        std = np.std(data_y[idx])

        #mass_center[i] = q[1]
        #mass_center[i] = np.average(mid_y, weights=hist[i, :]**1)
        mass_center[i] = np.mean(data_y[idx])#, weights=hist[i, :]**1)
        #uncertainty_lower[i] = q[0]
        #uncertainty_upper[i] = q[2]

        uncertainty_lower[i] = mass_center[i] - std
        uncertainty_upper[i] = mass_center[i] + std




"""
if args.breakdown == "atmocn":
    print("Anomalous AR forcing: ")
    for m, (month_name, idx) in enumerate(plot_months): 
        ratio = ds_stat["AR"]["MLG_nonfrc"][idx, 0] /  ds_stat["AR"]["MLG_frc"][idx, 0] 
        print("[month=%s, idx=%d] The ratio MLG_nonfrc / MLG_frc = %.2f" % (month_name, idx, ratio) )
"""

# =========================== Plotting Codes below ====================================

shared_levels = np.linspace(-1, 1, 11) * 0.5
plot_infos = {
    
    "dMLTdt" : {
        "levels": shared_levels,
        "label" : "$ \\dot{\\overline{\\Theta}}_{\mathrm{ttl}} $",
        "color" : "gray",
        "hatch" : '///',
    },

    "MLG_frc" : {
        "levels": shared_levels,
        "label" : "$ \\dot{\\overline{\\Theta}}_{\mathrm{sfc}} $",
        "color" : "orangered",
        "hatch" : '///',
    }, 

    "MLG_nonfrc" : {
        "levels": shared_levels,
        "label" : "$ \\dot{\\overline{\\Theta}}_{\mathrm{ocn}} $",
        "color" : "dodgerblue",
        "hatch" : '///',
    }, 

    "MLG_adv" : {
        "levels": shared_levels,
        "label" : "$ G_{\mathrm{adv}} $",
    }, 

    "MLG_vdiff" : {
        "levels": shared_levels,
        "label" : "$ G_{\mathrm{vdiff}} $",
    }, 

    "MLG_ent" : {
        "levels": shared_levels,
        "label" : "$ G_{\mathrm{ent}} $",
    }, 

    "MLG_hdiff" : {
        "levels": shared_levels,
        "label" : "$ G_{\mathrm{hdiff}} $",
    }, 

    "MLG_res2" : {
        "levels": shared_levels,
        "label" : "$ G_{\mathrm{res}} $",
    }, 

    "MLG_frc_sw" : {
        "levels": shared_levels,
        "label" : "$ G_{\mathrm{sw}} $",
    }, 

    "MLG_frc_lw" : {
        "levels": shared_levels,
        "label" : "$ G_{\mathrm{lw}} $",
    }, 

    "MLG_frc_lh" : {
        "levels": shared_levels,
        "label" : "$ G_{\mathrm{lh}} $",
    }, 

    "MLG_frc_sh" : {
        "levels": shared_levels,
        "label" : "$ G_{\mathrm{sh}} $",
    }, 

    "MLG_frc_fwf" : {
        "levels": shared_levels,
        "label" : "$ G_{\mathrm{fwf}} $",
    }, 

    "dMLDdt" : {
        "levels": np.linspace(-1, 1, 11) * 0.5,
        "label" : "$ w_e = \partial h / \partial t $",
    }, 


}

# Plot data
print("Loading Matplotlib...")
import matplotlib as mpl
if args.no_display is False:
    mpl.use('TkAgg')
else:
    mpl.use('Agg')
    mpl.rc('font', size=15)
    mpl.rc('axes', labelsize=15)
     
 
  
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from matplotlib.dates import DateFormatter
from scipy.stats import linregress

print("done")

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 3,
    h = 3,
    wspace = 1.0,
    hspace = 1.0,
    w_left = 1.0,
    w_right = 1.0,
    h_bottom = 1.0,
    h_top = 0.5,
    ncol = 1,
    nrow = 1,
)


subplot_kw = {
    'aspect' : 1,
}

fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw = subplot_kw, gridspec_kw = gridspec_kw)


def pretty_latlon(lat, lon):

    lon %= 360
    
    if lon > 180:
        lon = 360 - lon
        lon_EW = "W"
    else:
        lon_EW = "E"

    if lat > 0:
        lat_NS = "N"
    elif lat == 0:
        lat_NS = "E"
    else:
        lat_NS = "S"

    lat = abs(lat)

    if lat % 1 == 0 and lon % 1 == 0:
        return "%d%s" % (lat, lat_NS), "%d%s" % (lon, lon_EW)

    else:
        return "%.2f%s" % (lat, lat_NS), "%.2f%s" % (lon, lon_EW)


hist[hist == 0] = np.nan

mappable = ax.contourf(mid_x, mid_y, hist.transpose(), np.linspace(0, hist_color_lev_max, 11), cmap='bone_r', extend="max", zorder=1)
#mappable = ax.imshow(hist.transpose(), cmap='bone_r', extend="max")


ax.text(0.10, 0.95, '$ y = \\left(%.2f\\pm%.4f\\right)x %+.2f$, $R = %.2f$' % (coe[0], std_slope, coe[1], corr_coe[1, 0],), transform=ax.transAxes, color='red', ha='left', va='top', size=10)

x = np.linspace(-1, 1, 100) * .7
#y = coe[0] * x**2 + coe[1] * x**1 + coe[2]
y = coe[0] * x + coe[1]
ax.plot(x, y, 'r--')
ax.plot(mid_x, mass_center, linestyle=':', color="dodgerblue", zorder=20)

# plot uncertainty
ax.fill_between(mid_x, uncertainty_lower, uncertainty_upper, zorder=2, alpha=0.2, facecolor="#888888", edgecolor="#000000")

plot_info_x = plot_infos[varname_x]
plot_info_y = plot_infos[varname_y]

ax.set_xlabel("%s [$ 1 \\times 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $]" % (plot_info_x['label'],))
ax.set_ylabel("%s [$ 1 \\times 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $]" % (plot_info_y['label'],))


ax.set_xlim(np.array([-1, 1]) * 1.5)
ax.set_ylim(np.array([-1, 1]) * 1.5)

ax.set_yticks([-1, 0, 1])
ax.set_xticks([-1, 0, 1])

cax = tool_fig_config.addAxesNextToAxes(fig, ax, "right", thickness=0.03, spacing=0.05)
cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
cb.set_label('PDF [$ \\left( 1 \\times 10^{-6} \\, \\mathrm{K} / \\mathrm{s}\\right)^{-2} $]')
#cb.set_ticks([])
ax.set_title(args.title)

ax.grid(True, alpha=0.3, which="major")

if args.output != "":
   
    print("Output filename: %s" % (args.output,))
    fig.savefig(args.output, dpi=200)


if not args.no_display:
    print("Show figure")
    plt.show()

