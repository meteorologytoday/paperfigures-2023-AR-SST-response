#!/bin/bash

source 00_setup.sh

AR_algo=ANOM_LEN

box_params=(
    "AR_REGION"    30 40 150 210
    "AR_REGION_W"  30 40 150 160
    "AR_REGION_E"  30 40 200 210
)
#    "OPEN_NPAC"  30 50 140 220
#    "CA_COASTAL" 35 50 230 240
#    "OPEN_NEPAC" 30 50 180 220
#)


number_of_years=25
nparams=5
for (( i=0 ; i < $(( ${#box_params[@]} / $nparams )) ; i++ )); do

    box_name="${box_params[$(( i * $nparams + 0 ))]}"
    lat_s="${box_params[$(( i * $nparams + 1 ))]}"
    lat_n="${box_params[$(( i * $nparams + 2 ))]}"
    lon_w="${box_params[$(( i * $nparams + 3 ))]}"
    lon_e="${box_params[$(( i * $nparams + 4 ))]}"


    for suffix in "" ; do

        count=1

        input_dir=${diagdata_dir}${suffix}/climanom_${yrng_str}/${AR_algo}
        
        eval "python3 $src_dir/plot_Gterms_latlon_range.py \\
            --number-of-years $number_of_years \\
            --input-dir $input_dir \\
            --lat-rng $lat_s $lat_n \\
            --lon-rng $lon_w $lon_e \\
            --output $fig_dir/Gterms_${box_name}_atmocn${suffix}_${count}.png \\
            --title $box_name \\
            --breakdown atmocn \\
            --no-display \\
            " &

        eval "python3 $src_dir/plot_Gterms_latlon_range.py \\
            --number-of-years $number_of_years \\
            --input-dir $input_dir \\
            --lat-rng $lat_s $lat_n \\
            --lon-rng $lon_w $lon_e \\
            --output $fig_dir/Gterms_${box_name}_atm${suffix}_${count}.png \\
            --title $box_name \\
            --breakdown atm \\
            --no-display \\
            " &

        eval "python3 $src_dir/plot_Gterms_latlon_range.py \\
            --number-of-years $number_of_years \\
            --input-dir $input_dir \\
            --lat-rng $lat_s $lat_n \\
            --lon-rng $lon_w $lon_e \\
            --output $fig_dir/Gterms_${box_name}_ocn${suffix}_${count}.png \\
            --title $box_name \\
            --breakdown ocn \\
            --no-display \\
            " &



        wait

    done


done
