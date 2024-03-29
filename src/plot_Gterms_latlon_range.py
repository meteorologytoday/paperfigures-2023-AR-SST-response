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

parser.add_argument('--number-of-years', type=int, required=True)
parser.add_argument('--input-dir', type=str, help='Input file', required=True)
parser.add_argument('--lat-rng', type=float, nargs=2, help='Latitude range', required=True)
parser.add_argument('--lon-rng', type=float, nargs=2, help='Longitude range', required=True)
parser.add_argument('--output', type=str, help='Output file', default="")
parser.add_argument('--title', type=str, help='Output title', default="")
parser.add_argument('--title-style', type=str, help='Output title', default="folder", choices=["folder", "latlon"])
parser.add_argument('--breakdown', type=str, help='Output title', default="atmocn", choices=["atmocn", "atm", "ocn"])
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()
print(args)

plotted_varnames = {
    "atmocn" : ["dMLTdt", "MLG_frc", "MLG_nonfrc"],
    "atm" : ["MLG_frc", "MLG_frc_sw", "MLG_frc_lw", "MLG_frc_sh", "MLG_frc_lh", "MLG_frc_fwf"],
    "ocn" : ["MLG_nonfrc", "MLG_adv", "MLG_vdiff", "MLG_ent", "MLG_hdiff"],
}[args.breakdown]

print(plotted_varnames)

factor = 1e-6
plot_months = [
    ( "Oct-Mar", 6 ),
    ( "Oct", 0 ),
    ( "Nov", 1 ),
    ( "Dec", 2 ),
    ( "Jan", 3 ),
    ( "Feb", 4 ),
    ( "Mar", 5 ),
]

plot_months = [
    ( "Oct-Mar", 6 ),
#    ( "Oct-Dec", 7 ),
#    ( "Jan-Mar", 8 ),
    ( "Oct-Nov", 9 ),
    ( "Dec-Jan",10 ),
    ( "Feb-Mar",11 ),
]


ds_stat = {}
for k in ["clim", "AR", ]:
    ds_stat[k] = xr.open_dataset("%s/stat_%s.nc" % (args.input_dir, k))
   
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

print(ds_stat["AR"].coords["lat"])
print(ds_stat["AR"].coords["lon"])
    

args.lon_rng = np.array(args.lon_rng) % 360.0

print("Selecting data range : lat = [%.2f , %.2f], lon = [%.2f, %.2f]" % (*args.lat_rng, *args.lon_rng))

for k, _var in ds_stat.items():
    latlon_sel = (
        (_var.coords["lat"] >= args.lat_rng[0])
        & (_var.coords["lat"] <= args.lat_rng[1])
        & (_var.coords["lon"] >= args.lon_rng[0])
        & (_var.coords["lon"] <= args.lon_rng[1])
    )

    ds_stat[k] = _var.where(latlon_sel).mean(dim=['lat', 'lon'])




if args.breakdown == "atmocn":
    print("Anomalous AR forcing: ")
    for m, (month_name, idx) in enumerate(plot_months): 
        ratio = ds_stat["AR"]["MLG_nonfrc"][idx, 0] /  ds_stat["AR"]["MLG_frc"][idx, 0] 
        print("[month=%s, idx=%d] The ratio MLG_nonfrc / MLG_frc = %.2f" % (month_name, idx, ratio) )


# =========================== Plotting Codes below ====================================







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


shared_levels = np.linspace(-1, 1, 11) * 0.5
plot_infos = {
    
    "dMLTdt" : {
        "levels": shared_levels,
        "label" : "$ \\dot{\\overline{\\Theta}}_{\mathrm{ttl}} $",
        "color" : "gray",
        "hatch" : None,#'///',
    },

    "MLG_frc" : {
        "levels": shared_levels,
        "label" : "$ \\dot{\\overline{\\Theta}}_{\mathrm{sfc}} $",
        "color" : "orangered",
        "hatch" : None,#'///',
    }, 

    "MLG_nonfrc" : {
        "levels": shared_levels,
        "label" : "$ \\dot{\\overline{\\Theta}}_{\mathrm{ocn}} $",
        "color" : "dodgerblue",
        "hatch" : None,#'///',
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


plot_ylim = {

    "atmocn" : {
        "mean" : [-1.2, 0.2],
        "anom" : [-0.6, 0.7],
    },

    "atm" : {
        "mean" : [-1.2, 0.8],
        "anom" : [-0.3, 0.7],
    },

    "ocn" : {
        "mean" : [-0.6, 0.3],
        #"anom" : [-0.06, 0.01],
        "anom" : [-0.6, 0.3],
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

fig, ax = plt.subplots(2, 1, figsize=(6, 8), squeeze=False, constrained_layout=True)# gridspec_kw = dict(hspace=0.3, wspace=0.4))


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



if args.title == "":

    if args.title_style == "folder":
        fig.suptitle(args.input_dir)
    elif args.title_style == "latlon":
        title = ""#("%s %s : " % pretty_latlon(final_lat, final_lon)) + args.breakdown
        fig.suptitle(title)


else:
    fig.suptitle(args.title)


full_width = 0.8
bar_width = full_width / len(plotted_varnames) #0.15


# Plot different decomposition
for s, sname in enumerate(["clim", "AR"]):

    ds = ds_stat[sname]
   
    _ax = ax[s, 0]

    for i, varname in enumerate(plotted_varnames):
        _data = ds[varname].to_numpy()

        #print("data shape: ", _data.shape)

        kwargs = {}
        if 'color' in plot_infos[varname]:
            kwargs['color'] = plot_infos[varname]['color']

        if 'hatch' in plot_infos[varname]:
            kwargs['hatch'] = plot_infos[varname]['hatch']


        #if sname == "AR":
        #    _data -= ds_stat["clim"][varname].to_numpy() 

        x_pos = np.arange(len(plot_months))
        selected_idx = [ plot_month[1] for plot_month in plot_months ]

        print("selected_idx = ", selected_idx)

        # 4 = annual_mean = the mean of annual means 
        _ax.bar(x_pos + i*bar_width, _data[selected_idx, 4] / factor, bar_width, label=plot_infos[varname]['label'], **kwargs)

        #print(ds)
        #_anom_ARpARf_data = ds_stat["AR+ARf"][varname].to_numpy()[selected_idx, :]  / factor
        _anom_AR_data = ds_stat["AR"][varname].to_numpy()[selected_idx, :] / factor
        #_anom_ARf_data = ds_stat["ARf"][varname].to_numpy()[selected_idx, :] / factor
        _mean_data = ds_stat["clim"][varname].to_numpy()[selected_idx, :] / factor

        # error bar

        #N = args.number_of_years
        N = _data[selected_idx, 8]
        print("N = ", N) 
        _error_bar_lower = - _anom_AR_data[:, 5] / (N**0.5)
        _error_bar_upper =   _anom_AR_data[:, 5] / (N**0.5)

        _offset = _data[selected_idx, 4] / factor



        # Plot error bars to different time
        text_transform = transforms.blended_transform_factory(_ax.transData, _ax.transAxes)
        for m, idx in enumerate(selected_idx): 
            _ax.plot(
                [x_pos[m] + i*bar_width] * 2,
                np.array([_error_bar_lower[m], _error_bar_upper[m]]) + _offset[m],
                color="black",
                zorder=99,
            )

            _ax.text(x_pos[m] + full_width/2 - bar_width/2, 0.95, "(%.1f)" % (N[m], ), transform=text_transform, va="top", ha="center")

            """
            result = ttest_ind_from_stats(
                _anom_AR_data[m, 0], _anom_AR_data[m, 1], _anom_AR_data[m, 3],
                _anom_ARf_data[m, 0], _anom_ARf_data[m, 1], _anom_ARf_data[m, 3],
                equal_var=False,
                alternative='two-sided',
            )

            print("[m=%d, idx=%d] Result of T-test: " % (m, idx), result)
            """

    _ax.set_xticks(x_pos)
    _ax.set_xticklabels([ plot_month[0] for plot_month in plot_months])

    _ax.set_xlabel("Month")
    _ax.set_ylabel("[ $ 1 \\times 10^{-6} \\mathrm{K} \\, / \\, \\mathrm{s} $ ]")

    _ax.set_title("(%s) %s " % (
        "abcdefghijklmn"[s],
        plot_infos_scnario[sname]['title'],
    ))

    
    _ax.set_xlim([-0.5, len(plot_months) + 1.0])

    _ax.legend(loc="center right", borderpad=0.4, labelspacing=0.1)

    if sname == "clim":
        ylim = plot_ylim[args.breakdown]['mean']

    elif sname == "AR": 
        ylim = plot_ylim[args.breakdown]['anom']

        
    yticks = np.arange( np.ceil(ylim[0] / 0.2) * 0.2 , ylim[1], 0.2)
    _ax.set_ylim(ylim)
    _ax.set_yticks(yticks)

    _ax.grid(True)
    _ax.set_axisbelow(True)

        
#ax[0, 0].set_ylim([-1.2, 1.2])
#ax[1, 0].set_ylim([-.8, .8])


if args.output != "":
   
    print("Output filename: %s" % (args.output,))
    fig.savefig(args.output, dpi=200)


if not args.no_display:
    print("Show figure")
    plt.show()

