#!/bin/bash

source 00_setup.sh

AR_algo=ANOM_LEN
#for suffix in "" "_500m"; do


lib_root="/cw3e/mead/projects/csg102/t2hsu/MITgcm-diagnostics"
run_root="/cw3e/mead/projects/csg102/t2hsu/AR_projects/project01/case04_ISOLATED_CYCLONE"

mitgcm_deltaT=60.0
mitgcm_dumpfreq=3600.0

export PYTHONPATH="$lib_root/src:$PYTHONPATH"



nproc=2

box_params=(
    "exp01" "run01_ctl-run02_fixedSST" "2017-01-04" "2017-01-04" "2017-01-09" 35 40 215 225
)

#"exp01" "run01_ctl-run02_fixedSST-run03_noadv-run04_hivisc-run05_hidiff" "2017-01-04" "2017-01-04" "2017-01-09" 35 40 215 225


skip_hrs=1
avg_hrs=3

figure_dir=figures_case04

box_params=(
    "OPEN_NPAC"  30 50 140 220
    "CA_COASTAL" 35 50 230 240
    "OPEN_NEPAC" 30 50 180 220
)



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
            --input-dir $input_dir \\
            --lat-rng $lat_s $lat_n \\
            --lon-rng $lon_w $lon_e \\
            --output $fig_dir/Gterms_atmocn${suffix}_${count}.png \\
            --title $box_name \\
            --breakdown atmocn \\
            " &

        wait

    done


done
