import numpy as np
import xarray as xr

import traceback
from pathlib import Path
import argparse
from datetime import (timedelta, datetime, timezone)
import tool_fig_config


parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--input', type=str, help='Input file', required=True)
parser.add_argument('--output', type=str, help='Input file', default="")
parser.add_argument('--title', type=str, help='Output title', default="")
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--markers', action="store_true")
args = parser.parse_args()
print(args)


ds = xr.open_dataset(args.input)

AR_mean= ds["count"].mean(dim=("time",))
AR_std = ds["count"].std(dim=("time",))


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

cent_lon = 180.0

plot_lon_l = 100.0
plot_lon_r = 260.0
plot_lat_b = 10.0
plot_lat_t = 60.0

proj = ccrs.PlateCarree(central_longitude=cent_lon)
proj_norm = ccrs.PlateCarree()

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 4.8,
    h = 2.0,
    wspace = 1.0,
    hspace = 0.5,
    w_left = 1.0,
    w_right = 1.0,
    h_bottom = 0.25,
    h_top = 1.0,
    ncol = 1,
    nrow = 1,
)


fig, ax = plt.subplots(
    1, 1,
    figsize=figsize,
    subplot_kw=dict(projection=proj, aspect="auto"),
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
)

ax.set_title("(a) AR days per year (shading)\nand variability (contour)")

coords = ds.coords
cmap = cm.get_cmap("GnBu")

#cmap.set_over("yellow")
#cmap.set_under("red")

#mean_levels = np.linspace(0, 70, 15)
mean_levels = np.linspace(0, 30, 6)
std_levels = np.linspace(0, 20, 11)

mappable = ax.contourf(coords["lon"], coords["lat"], AR_mean, levels=mean_levels, cmap=cmap, extend="max", transform=proj_norm)
cs = ax.contour(coords["lon"], coords["lat"], AR_std, levels=std_levels, colors="k", transform=proj_norm, linewidths=1)

ax.clabel(cs, fmt="%d")

ax.plot([160, 360-160], [30, 35], color="lime", linestyle="dashed", transform=proj_norm, zorder=100)
ax.plot([360-150, 360-130], [30, 40], color="lime", linestyle="dashed", transform=proj_norm, zorder=100)

# Add western and eastern boxes over AR active region

ax.scatter([151, 360-155], [31, 35], s=20, marker='x', edgecolors='face', c="red", transform=proj_norm, zorder=99)

"""
ax.add_patch(
    Rectangle(
        (150, 30), 10, 10,
        linewidth=2, edgecolor="red", facecolor='none', transform=proj_norm,
    zorder=99)
)

ax.add_patch(
    Rectangle(
        (200, 30), 10, 10,
        linewidth=2, edgecolor="red", facecolor='none', transform=proj_norm,
    zorder=99)
)
"""


ax.set_global()
ax.coastlines()
ax.set_extent([plot_lon_l, plot_lon_r, plot_lat_b, plot_lat_t], crs=proj_norm)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
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

 
cax = tool_fig_config.addAxesNextToAxes(fig, ax, "right", thickness=0.02, spacing=0.02)
cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)#, #ticks=[-1, 0, 1])
cb.ax.set_ylabel("AR days per wateryear")

if not args.no_display:
    plt.show()

if args.output != "":
    
    fig.savefig(args.output, dpi=200)

