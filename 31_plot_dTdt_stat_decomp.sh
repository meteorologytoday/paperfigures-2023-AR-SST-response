#!/bin/bash

# This is the program to generate the scatter plot of dT/dt

source 00_setup.sh

nproc=2

box_params=(
    "AR_ACTIVE"  30 40 150 210
    "ALL_NPAC"  10 60 100 240
    "EXTRATROPICAL_NPAC"  30 60 100 240
)
    
#"OPEN_NPAC"  0 90 100 260

nparams=5
for (( i=0 ; i < $(( ${#box_params[@]} / $nparams )) ; i++ )); do

    box_name="${box_params[$(( i * $nparams + 0 ))]}"
    lat_s="${box_params[$(( i * $nparams + 1 ))]}"
    lat_n="${box_params[$(( i * $nparams + 2 ))]}"
    lon_w="${box_params[$(( i * $nparams + 3 ))]}"
    lon_e="${box_params[$(( i * $nparams + 4 ))]}"


    input_dir=${diagdata_dir}/climanom_${yrng_str}
    
    
    eval "python3 $src_dir/plot_dTdt_stat_decomp.py \\
        --input-dir $input_dir \\
        --lat-rng $lat_s $lat_n \\
        --lon-rng $lon_w $lon_e \\
        --output $fig_dir/dTdt_stat_${box_name}.svg \\
        --AR-algo $AR_algo \\
        --title \"\" \\
        --no-display \\
        " &

    wait


done
