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

parser.add_argument('--input-dir', type=str, help='Input file', required=True)
parser.add_argument('--output', type=str, help='Output file', default="")
parser.add_argument('--varnames', type=str, nargs="+", help='Output file', default=["dMLTdt", "MLG_frc", "MLG_nonfrc"])
parser.add_argument('--watermonths', type=int, nargs="+", help='Output file', default=[1, 2, 3, 4, 5, 6])
parser.add_argument('--title', type=str, help='Output title', default="")
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()
print(args)

t_months = np.array(args.watermonths)

ds_stat = {}
for k in ["clim", "AR", "ARf", "AR-ARf", "AR+ARf"]:
    ds_stat[k] = xr.open_dataset("%s/stat_%s.nc" % (args.input_dir, k))



# generate AR freq
#print(np.array([31, 30, 31, 31, 28, 31])[:, None, None]) 
#print(ds_stat["AR"]["IVT"][:, :, :, 3].to_numpy().shape)
ARfreq = ds_stat["AR"]["IVT"][:, :, :, 3].to_numpy() / ds_stat["AR+ARf"]["IVT"][:, :, :, 3].to_numpy()
# / ( np.array([31, 30, 31, 31, 28, 31])[:, None, None] ) / 25


plot_infos_scnario = {

    "clim" : {
        "title" : "All",
    },

    "AR" : {
        "title" : "AR",
    },

    "ARf" : {
        "title" : "AR free",
    },

    "AR-ARf" : {
        "title" : "AR minus AR free",
    }

}


plot_infos = {
    
    "dMLTdt" : {
        "levels": np.linspace(-1, 1, 11) * 0.5,
        "label" : "$ G_{\mathrm{ttl}} $",
        "factor" : 1e-6,
    },

    "MLG_frc" : {
        "levels": np.linspace(-1, 1, 11) * 0.5,
        "label" : "$ G_{\mathrm{frc}} $",
        "factor" : 1e-6,
    }, 

    "MLG_nonfrc" : {
        "levels": np.linspace(-1, 1, 11) * 0.5,
        "label" : "$ G_{\mathrm{nfrc}} $",
        "factor" : 1e-6,
    }, 

    "MLG_vdiff" : {
        "levels": np.linspace(-1, 1, 11) * 0.2,
        "label" : "$ G_{\mathrm{vdiff}} $",
        "factor" : 1e-6,
    }, 

    "MLG_hdiff" : {
        "levels": np.linspace(-1, 1, 11) * 0.2,
        "label" : "$ G_{\mathrm{hdiff}} $",
        "factor" : 1e-6,
    }, 


    "MLG_ent" : {
        "levels": np.linspace(-1, 1, 11) * 0.2,
        "label" : "$ G_{\mathrm{ent}} $",
        "factor" : 1e-6,
    }, 

    "MLG_adv" : {
        "levels": np.linspace(-1, 1, 11) * 0.2,
        "label" : "$ G_{\mathrm{adv}} $",
        "factor" : 1e-6,
    }, 

    "MXLDEPTH" : {
        "levels": np.linspace(-1, 1, 11) * 20,
        "label" : "MXLDEPTH",
        "factor" : 1.0,
    }, 
 
    "MLD" : {
        "levels": np.linspace(-1, 1, 11) * 20,
        "label" : "MLD",
        "factor" : 1.0,
    }, 
 
    "MLG_frc_sw" : {
        "levels": np.linspace(-1, 1, 11) * 0.5,
        "label" : "$ G_{\mathrm{sw}} $",
        "factor" : 1e-6,
    }, 

    "MLG_frc_lw" : {
        "levels": np.linspace(-1, 1, 11) * 0.5,
        "label" : "$ G_{\mathrm{lw}} $",
        "factor" : 1e-6,
    }, 

    "MLG_frc_sh" : {
        "levels": np.linspace(-1, 1, 11) * 0.5,
        "label" : "$ G_{\mathrm{sen}} $",
        "factor" : 1e-6,
    }, 

    "MLG_frc_lh" : {
        "levels": np.linspace(-1, 1, 11) * 0.5,
        "label" : "$ G_{\mathrm{lat}} $",
        "factor" : 1e-6,
    }, 

    "MLG_frc_fwf" : {
        "levels": np.linspace(-1, 1, 11) * 0.5,
        "label" : "$ G_{\mathrm{fwf}} $",
        "factor" : 1e-6,
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
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


print("done")

from scipy.stats import ttest_ind_from_stats

def student_t_test(mean1, std1, nobs1, mean2, std2, nobs2):

    shp = mean1.shape
    
    pvals = np.zeros(shp, dtype=np.float64)
    pvals[:] = np.nan
    
    tstats = pvals.copy()

    for j in range(shp[0]):
        for i in range(shp[1]):
            
            m1 = mean1[j, i]
            m2 = mean2[j, i]
            
            s1 = std1[j, i]
            s2 = std2[j, i]

            n1 = nobs1[j, i]
            n2 = nobs2[j, i]

            if np.all(np.isfinite([m1, m2, s1, s2, n1, n2])) and n1 >= 30 and n2 >= 30:
                
                result = ttest_ind_from_stats(
                    m1, s1, n1,
                    m2, s2, n2,
                    equal_var=False,
                    alternative='two-sided',
                )

                tstats[j, i] = result[0]
                pvals[j, i] = result[1]



    return tstats, pvals

cent_lon = 180.0

plot_lon_l = 120.0
plot_lon_r = 240.0
plot_lat_b = 10.0
plot_lat_t = 60.0

proj = ccrs.PlateCarree(central_longitude=cent_lon)
proj_norm = ccrs.PlateCarree()

varnames = args.varnames

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 4.8,
    h = 2.0,
    wspace = 1.0,
    hspace = 0.5,
    w_left = 1.0,
    w_right = 1.5,
    h_bottom = 1.0,
    h_top = 1.0,
    ncol = len(t_months),
    nrow = len(varnames),
)


fig, ax = plt.subplots(
    len(varnames), len(t_months),
    figsize=figsize,
    subplot_kw=dict(projection=proj, aspect="auto"),
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
    squeeze=False,
)

coords = ds_stat["clim"].coords
cmap = cm.get_cmap("bwr")

cmap.set_over("green")
cmap.set_under("yellow")

mappables = [ None for i in range(len(varnames)) ]
for i, mon in enumerate(t_months):

    _ax = ax[:, i]
  
    m = mon - 1


    # Set title for different month except for the whole average mon==7
    if mon != 7:
        _ax[0].set_title([
            "Oct",
            "Nov",
            "Dec",
            "Jan",
            "Feb",
            "Mar",
            "Oct-Mar",
        ][m], size=20)

    for i, varname in enumerate(varnames):

        _mean1 = ds_stat["AR"][varname][m, :, :, 0].to_numpy()
        _mean2 = ds_stat["ARf"][varname][m, :, :, 0].to_numpy()
 
        _std1 = ds_stat["AR"][varname][m, :, :, 1].to_numpy()
        _std2 = ds_stat["ARf"][varname][m, :, :, 1].to_numpy()
 
        _nobs1 = ds_stat["AR"][varname][m, :, :, 3].to_numpy()
        _nobs2 = ds_stat["ARf"][varname][m, :, :, 3].to_numpy()

        _, pvals = student_t_test(_mean1, _std1, _nobs1, _mean2, _std2, _nobs2)        
        
        _diff = ds_stat["AR"][varname][m, :, :, 0].to_numpy()
        #_diff = ((ds_stat["AR"][varname][m, :, :, 0] - ds_stat["clim"][varname][m, :, :, 0])).to_numpy()

        _dot = _diff * 0 
        _significant_idx =  (pvals <= 0.05) 

        _dot[ _significant_idx                 ] = 0.75
        _dot[ np.logical_not(_significant_idx) ] = 0.25

        # Remove insignificant data
        #_diff[np.logical_not(_significant_idx)] = np.nan

        plot_info = plot_infos[varname]

        mappables[i] = _ax[i].contourf(coords["lon"], coords["lat"], _diff / plot_info["factor"], levels=plot_info["levels"], cmap=cmap, extend="both", transform=proj_norm)

        cs = _ax[i].contourf(coords["lon"], coords["lat"], _dot, colors='none', levels=[0, 0.5, 1], hatches=[None, ".."], transform=proj_norm)
        

        for _, collection in enumerate(cs.collections):
            collection.set_edgecolor((.2, .2, .2))
            collection.set_linewidth(0.)

        _ARfreq = ARfreq[m, :, :]
        #_ARfreq[_ARfreq >= 0.3] = 0.75
        #_ARfreq[_ARfreq <  0.3] = 0.25
        #_ax[i].contourf(coords["lon"], coords["lat"], ARfreq[m, :, :], colors='none', levels=[0, 0.5, 1], hatches=[None, ".."], transform=proj_norm)
        _ax[i].contour(coords["lon"], coords["lat"], ARfreq[m, :, :], levels=[0.3, ], colors="k", linestyles='--',  linewidths=1, transform=proj_norm, alpha=0.8, zorder=10)
        _ax[i].contour(coords["lon"], coords["lat"], ARfreq[m, :, :], levels=[0.4, ], colors="k", linestyles='-',linewidths=1, transform=proj_norm, alpha=0.8, zorder=10)
       
    for __ax in _ax: 

        __ax.set_global()
        #__ax.gridlines()
        __ax.coastlines(color='gray')
        __ax.set_extent([plot_lon_l, plot_lon_r, plot_lat_b, plot_lat_t], crs=proj_norm)

        gl = __ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
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



for i in range(len(mappables)):
    cax = tool_fig_config.addAxesNextToAxes(fig, ax[i, -1], "right", thickness=0.03, spacing=0.05)
    cb = plt.colorbar(mappables[i], cax=cax, orientation="vertical", pad=0.00)
    cb.ax.set_ylabel(" %s [ $ 1 \\times 10^{-6} \\, \\mathrm{K} \\, / \\, \\mathrm{s} $ ]" % (plot_infos[varnames[i]]["label"],))


if not args.no_display:
    plt.show()

if args.output != "":

    print("Saving output: ", args.output)    
    fig.savefig(args.output, dpi=200)


print("Finished.")


