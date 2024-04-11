#!/bin/bash


source 00_setup.sh


for suffix in "" ; do

    wm_str="Oct-Nov Dec-Jan Feb-Mar"
    count=1

    input_dir=${diagdata_dir}${suffix}/climanom_${yrng_str}/${AR_algo}${algo_suffix}

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames IWV IVT SFCWIND \\
        --output $fig_dir/atmsfc${suffix}_${count}.svg \\
        --time $wm_str \\
        --no-display" &

    wait

    wm_str="Oct-Mar"

    count=$(( count + 1 ))
    echo "Plotting time: $wm_str"

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames IWV IVT SFCWIND \\
        --output $fig_dir/atmsfc${suffix}_${count}.svg \\
        --time $wm_str \\
        --add-thumbnail-title \\
        --thumbnail-offset 3 \\
        --no-display" &


    wait

done
