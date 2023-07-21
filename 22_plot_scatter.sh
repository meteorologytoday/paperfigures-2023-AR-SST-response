#!/bin/bash

source 00_setup.sh

AR_algo=ANOM_LEN

nproc=2

box_params=(
    "OPEN_NPAC"  0 90 100 260
)

nparams=5
for (( i=0 ; i < $(( ${#box_params[@]} / $nparams )) ; i++ )); do

    box_name="${box_params[$(( i * $nparams + 0 ))]}"
    lat_s="${box_params[$(( i * $nparams + 1 ))]}"
    lat_n="${box_params[$(( i * $nparams + 2 ))]}"
    lon_w="${box_params[$(( i * $nparams + 3 ))]}"
    lon_e="${box_params[$(( i * $nparams + 4 ))]}"


    for suffix in "" ; do

        input_dir=${diagdata_dir}${suffix}/climanom_${yrng_str}/${AR_algo}
        
        eval "python3 $src_dir/plot_dTdt_scatter_by_ARday_frc_nonfrc.py \\
            --input-dir $input_dir \\
            --lat-rng $lat_s $lat_n \\
            --lon-rng $lon_w $lon_e \\
            --output $fig_dir/dTdt_scatter.png \\
            --title $box_name \\
            " &

        wait

    done


done
