#!/bin/bash


source 00_setup.sh




AR_algo=ANOM_LEN



for suffix in "" ; do

    wm_str="1 2 3 4 5 6"
    count=1

    input_dir=${diagdata_dir}${suffix}/climanom_${yrng_str}/${AR_algo}

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames IWV IVT SFCWIND \\
        --output $fig_dir/atmsfc${suffix}_${count}.png \\
        --watermonths $wm_str \\
        --no-display" &

    wait

    wm_str="7"

    count=$(( count + 1 ))
    echo "Plotting watermonths: $wm_str"

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames IWV IVT SFCWIND \\
        --output $fig_dir/atmsfc${suffix}_${count}.png \\
        --watermonths $wm_str \\
        --add-thumbnail-title \\
        --thumbnail-offset 3 \\
        --no-display" &


    wait

done
