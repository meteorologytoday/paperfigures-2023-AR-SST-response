#import fmon_tools, watertime_tools
#import ARstat_tool
import xarray as xr
import traceback
from pathlib import Path
import argparse
from datetime import (timedelta, datetime, timezone)
from scipy.stats import ttest_ind_from_stats
import numpy as np

parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--input-dir', type=str, help='Input file', required=True)
parser.add_argument('--lat-rng', type=float, nargs=2, help='Latitude range', required=True)
parser.add_argument('--lon-rng', type=float, nargs=2, help='Longitude range', required=True)
parser.add_argument('--output', type=str, help='Output file', default="")
parser.add_argument('--title', type=str, help='Output title', default="")
parser.add_argument('--title-style', type=str, help='Output title', default="folder", choices=["folder", "latlon"])
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()
print(args)


print("Loading data...")
ds_stat = {}
for k in ["AR",]:
    ds_stat[k] = xr.open_dataset("%s/stat_%s.nc" % (args.input_dir, k)).sel(time="Oct-Mar", stat="mean")
   
    ds = ds_stat[k]
    
    MLG_res2 = (ds['dMLTdt'] - (
          ds['MLG_frc']
        + ds['MLG_nonfrc']
        + ds['MLG_rescale']
    )).rename('MLG_res2')

    print("RESIDUE: ", np.amax(np.abs(ds['MLG_residue'])))

    ds = xr.merge(
        [
            ds,
            MLG_res2,
        ]
    )

    ds_stat[k] = ds

    ds = None 

    

args.lon_rng = np.array(args.lon_rng) % 360.0

print("Selecting data range : lat = [%.2f , %.2f], lon = [%.2f, %.2f]" % (*args.lat_rng, *args.lon_rng))

for k, _var in ds_stat.items():
    latlon_sel = (
        (_var.coords["lat"] >= args.lat_rng[0])
        & (_var.coords["lat"] <= args.lat_rng[1])
        & (_var.coords["lon"] >= args.lon_rng[0])
        & (_var.coords["lon"] <= args.lon_rng[1])
    )

    ds_stat[k] = _var.where(latlon_sel)


factor = 0.000001
ds = ds_stat["AR"]
data_x = ds["MLG_frc"].to_numpy() / factor
data_y = ds["MLG_nonfrc"].to_numpy() / factor

print(data_x.shape)

edges_x = np.linspace(-1, 1, 51)
edges_y = np.linspace(-1, 1, 52)

hist, edges_x, edges_y = np.histogram2d(data_x, data_y, bins=[edges_x, edges_y])

mid_x = ( edges_x[:-1] + edges_x[1:] ) / 2
mid_y = ( edges_y[:-1] + edges_y[1:] ) / 2

print("Maximum of hist : ", np.amax(hist))

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
        "label" : "$ G_{\mathrm{ttl}} $",
        "color" : "gray",
        "hatch" : '///',
    },

    "MLG_frc" : {
        "levels": shared_levels,
        "label" : "$ G_{\mathrm{frc}} $",
        "color" : "orangered",
        "hatch" : '///',
    }, 

    "MLG_nonfrc" : {
        "levels": shared_levels,
        "label" : "$ G_{\mathrm{nfrc}} $",
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


}

# Plot data
print("Loading Matplotlib...")
import matplotlib as mpl
if args.no_display is False:
    mpl.use('TkAgg')
else:
    mpl.use('Agg')
    #mpl.rc('font', size=20)
    #mpl.rc('axes', labelsize=15)
     
 
  
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from matplotlib.dates import DateFormatter
from scipy.stats import linregress

print("done")

fig, ax = plt.subplots(1, 1, figsize=(6, 8), constrained_layout=True)# gridspec_kw = dict(hspace=0.3, wspace=0.4))


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


mappable = ax.contourf(mid_x, mid_y, hist, 20, cmap='hot_r', extend="both")

ax.set_ylabel("[ $ 1 \\times 10^{-6} \\mathrm{K} \\, / \\, \\mathrm{s} $ ]")
ax.set_xlabel("[ $ 1 \\times 10^{-6} \\mathrm{K} \\, / \\, \\mathrm{s} $ ]")


if args.output != "":
   
    print("Output filename: %s" % (args.output,))
    fig.savefig(args.output, dpi=200)


if not args.no_display:
    print("Show figure")
    plt.show()

