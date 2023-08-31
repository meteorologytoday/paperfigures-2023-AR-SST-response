#!/bin/bash

source 00_setup.sh

AR_algo=ANOM_LEN

box_params=(
    "AR_REGION_W"  31 151
    "AR_REGION_E"  35 205
)

number_of_years=25
nparams=3
for (( i=0 ; i < $(( ${#box_params[@]} / $nparams )) ; i++ )); do

    box_name="${box_params[$(( i * $nparams + 0 ))]}"
    lat="${box_params[$(( i * $nparams + 1 ))]}"
    lon="${box_params[$(( i * $nparams + 2 ))]}"


    for suffix in "" ; do

        count=1

        input_dir=${diagdata_dir}${suffix}/climanom_${yrng_str}/${AR_algo}
        
        eval "python3 $src_dir/plot_Gterms_selected_point.py \\
            --number-of-years $number_of_years \\
            --input-dir $input_dir \\
            --lat $lat \\
            --lon $lon \\
            --output $fig_dir/Gterms_pt_${box_name}_atmocn${suffix}_${count}.png \\
            --skip-subfig-cnt 0 \\
            --breakdown atmocn \\
            --no-display \\
            --no-title \\
            " &

        eval "python3 $src_dir/plot_Gterms_selected_point.py \\
            --number-of-years $number_of_years \\
            --input-dir $input_dir \\
            --lat $lat \\
            --lon $lon \\
            --output $fig_dir/Gterms_pt_${box_name}_atm${suffix}_${count}.png \\
            --skip-subfig-cnt 2 \\
            --breakdown atm \\
            --no-display \\
            --no-title \\
            " &

        eval "python3 $src_dir/plot_Gterms_selected_point.py \\
            --number-of-years $number_of_years \\
            --input-dir $input_dir \\
            --lat $lat \\
            --lon $lon \\
            --output $fig_dir/Gterms_pt_${box_name}_ocn${suffix}_${count}.png \\
            --skip-subfig-cnt 4 \\
            --breakdown ocn \\
            --no-display \\
            --no-title \\
            " &



        wait

    done


done
