import numpy as np
import xarray as xr
import pandas as pd

import traceback
from pathlib import Path
import argparse
from datetime import (timedelta, datetime, timezone)

def correlate(x1, x2):
    
    if len(x1) != len(x2):
        raise Exception("Unequal input of arrays.")


    c = np.zeros((len(x1),))

    _x1 = np.array(x1)
    _x2 = np.array(x2)

    _x1 /= np.sum(_x1**2)**0.50
    _x2 /= np.sum(_x2**2)**0.50

    for i in range(len(c)-1):
        __x1 = _x1[:len(c)-i]
        __x2 = _x2[i:]

        c[i] = np.sum(__x1 * __x2)

    return c





parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--input', type=str, help='Input file', required=True)
parser.add_argument('--input-NINO', type=str, help='Input NINO index file', default="")
parser.add_argument('--input-PDO',  type=str, help='Input NINO index file', default="")
parser.add_argument('--output-EOF', type=str, help='Input file', default="")
parser.add_argument('--nEOF', type=int, help='Input file', default=2)
parser.add_argument('--output-timeseries', type=str, help='Input file', default="")
parser.add_argument('--title', type=str, help='Output title', default="")
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--markers', action="store_true")
args = parser.parse_args()
print(args)



ds = xr.open_dataset(args.input)


climidx_names = ["NINO", "PDO"]
corr = {}
climidx = {}


_args = vars(args)
for climidx_name in climidx_names:

    argname = "input_%s" % (climidx_name,)

    if _args[argname] != "":

        corr[climidx_name] = []

        ds_climidx = xr.open_dataset(_args[argname])
        _climidx = np.zeros((len(ds.coords["time"]), ))

        for i in range(len(_climidx)):
            date_selected = pd.date_range("%04d-01-01" % (ds.time.dt.year[i],), freq="MS", periods=6) - pd.DateOffset(months=3)
            _climidx[i] = ds_climidx["anom"].sel(time=date_selected).mean(dim="time")
           
            print("[%s] Year %04d: %.2f" % (climidx_name, ds.time.dt.year[i], _climidx[i])) 

        print(_climidx)

        for i in range(args.nEOF):
            corr[climidx_name].append(correlate(_climidx, ds["amps_normalized"].sel(EOF=i)))


        climidx[climidx_name] = _climidx

for climidx_name, _corr in corr.items():
    for i in range(len(_corr)):
        print(climidx_name, " corr with EOF%d: " % (i+1), _corr[i])

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
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


print("done")

# First figure : EOFs

cent_lon = 180.0

plot_lon_l = 100.0
plot_lon_r = 260.0
plot_lat_b = 10.0
plot_lat_t = 60.0

proj = ccrs.PlateCarree(central_longitude=cent_lon)
proj_norm = ccrs.PlateCarree()

fig_EOF, ax = plt.subplots(
    2, 1,
    figsize=(8, 8),
    subplot_kw=dict(projection=proj),
    gridspec_kw=dict(hspace=0.2, wspace=0.2),
    constrained_layout=False,
)

for i, _ax in enumerate(ax):

    _ax.set_title("(%s) AR freqency EOF%d (explained variance = %d %%)" % ("abcdefg"[i+1], i+1, np.floor(ds["explained_variance_ratio"].sel(EOF=i)*100)))

    coords = ds.coords

    mappable = _ax.contourf(coords["lon"], coords["lat"], ds["count_EOF"].sel(EOF=i), levels=np.linspace(-1, 1, 21) * 0.1, cmap="bwr", extend="both", transform=proj_norm)

    _ax.plot([160, 360-160], [30, 35], color="lime", linestyle="dashed", transform=proj_norm)
    _ax.plot([360-150, 360-130], [30, 40], color="lime", linestyle="dashed", transform=proj_norm)

    _ax.set_global()
    _ax.coastlines()
    _ax.set_extent([plot_lon_l, plot_lon_r, plot_lat_b, plot_lat_t], crs=proj_norm)

    gl = _ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')

    gl.xlabels_top   = False
    gl.ylabels_right = False

    #gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    gl.xlocator = mticker.FixedLocator([120, 150, 180, -150, -120])#np.arange(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator([10, 20, 30, 40, 50])

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

        
    #cb = plt.colorbar(mappable, ax=_ax, orientation="vertical", pad=0.01, shrink=0.5)
    #cb.ax.set_ylabel("AR days per wateryear")


# Second figure: timeseries

fig_timeseries, ax = plt.subplots(1, 1, figsize=(6, 4))


for i in range(args.nEOF): #len(ds.coords["EOF"])):

    line_prop = [
        dict(ls="solid",  color="k", ),
        dict(ls="dashed", color="k", ),
    ][i]

    label = "EOF%d" % (i+1,)
   
    #if len(corr) != 0:
    #    label = "%s, $R^2=%.2f$" % (corr[i][0],)
   
    ax.plot(ds.coords["time"].dt.year, ds["amps_normalized"].sel(EOF=i), **line_prop, label=label)

for climidx_name, climidx in climidx.items():
 
    prop = dict(
        NINO = dict(ls="solid", color="orangered",  label="NINO3.4"),
        PDO  = dict(ls="solid", color="dodgerblue", label="PDO"),
    )[climidx_name]

    corr_str = ["%.2f" % corr[climidx_name][i][0] for i in range(args.nEOF)]
    prop["label"] = "%s (%s)" % (prop["label"], ", ".join(corr_str))
        
    ax.plot(ds.coords["time"].dt.year, climidx / np.std(climidx), **prop)

ax.set_xlabel("Time")
ax.set_ylabel("Normalized index")

ax.legend()
ax.grid()
#ax.set_title("Timeseries of EOF indices")

"""
# Figure 3: cross-correlation
fig_corr, ax = plt.subplots(1, 1, figsize=(6, 4))

for i in range(len(corr)):
    ax.plot(list(range(len(corr[i]))), corr[i], label="EOF%d" % (i,))

ax.legend()
"""

if not args.no_display:
    plt.show()

if args.output_EOF != "":
    fig_EOF.savefig(args.output_EOF, dpi=200)

if args.output_timeseries != "":
    fig_timeseries.savefig(args.output_timeseries, dpi=200)


