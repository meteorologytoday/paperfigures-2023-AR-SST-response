#!/bin/bash

source 00_setup.sh

box_params=(
    "AR_REGION_W"  31 151
    "AR_REGION_E"  35 205
)

box_params=(
    "MAXOCNIMPACT"  35 $((360 - 151))
    "OLD_AR_REGION_E"  35 205
    "AR_REGION_S"  37 193
    "AR_REGION_MAX"  39 193
    "AR_REGION_E"  39 210
)

box_params=(
    "MAXOCNIMPACT"  35 $((360 - 151))
)



number_of_years=25
nparams=3
for (( i=0 ; i < $(( ${#box_params[@]} / $nparams )) ; i++ )); do

    box_name="${box_params[$(( i * $nparams + 0 ))]}"
    lat="${box_params[$(( i * $nparams + 1 ))]}"
    lon="${box_params[$(( i * $nparams + 2 ))]}"


    for condition in "AR" "clim" ; do

        count=1

        input_dir=${diagdata_dir}${suffix}/climanom_${yrng_str}/${AR_algo}${algo_suffix}
        
        eval "python3 $src_dir/plot_Gterms_selected_point.py \\
            --number-of-years $number_of_years \\
            --input-dir $input_dir \\
            --lat $lat \\
            --lon $lon \\
            --output $fig_dir/Gterms_pt_${box_name}_atmocn_${condition}_${count}.png \\
            --skip-subfig-cnt 0 \\
            --breakdown atmocn \\
            --conditions "$condition" \\
            --no-display \\
            --no-title \\
            " &

        eval "python3 $src_dir/plot_Gterms_selected_point.py \\
            --number-of-years $number_of_years \\
            --input-dir $input_dir \\
            --lat $lat \\
            --lon $lon \\
            --output $fig_dir/Gterms_pt_${box_name}_atm_${condition}_${count}.png \\
            --skip-subfig-cnt 1 \\
            --breakdown atm \\
            --conditions "$condition" \\
            --no-display \\
            --no-title \\
            " &

        eval "python3 $src_dir/plot_Gterms_selected_point.py \\
            --number-of-years $number_of_years \\
            --input-dir $input_dir \\
            --lat $lat \\
            --lon $lon \\
            --output $fig_dir/Gterms_pt_${box_name}_ocn_${condition}_${count}.png \\
            --skip-subfig-cnt 2 \\
            --breakdown ocn \\
            --conditions "$condition" \\
            --no-display \\
            --no-title \\
            " &



        wait

    done


done
