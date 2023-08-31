#!/bin/bash


source 00_setup.sh




AR_algo=ANOM_LEN



for suffix in "" ; do

    wm_str="Oct-Nov Dec-Jan Feb-Mar"
    count=1

    input_dir=${diagdata_dir}${suffix}/climanom_${yrng_str}/${AR_algo}

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames MLHADVT_g MLHADVT_ag ENT_ADV MLD lcc mcc hcc \\
        --output $fig_dir/analysis_${suffix}_${count}.png \\
        --time $wm_str \\
        --no-display" &

    wm_str="Oct-Mar"

    count=$(( count + 1 ))
    echo "Plotting time: $wm_str"

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames MLHADVT_g MLHADVT_ag ENT_ADV \\
        --output $fig_dir/analysis_advbkdn_${suffix}_${count}.png \\
        --time $wm_str \\
        --add-thumbnail-title \\
        --no-display" &

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames hcc mcc lcc MLD \\
        --output $fig_dir/analysis_mldandcld_${suffix}_${count}.png \\
        --time $wm_str \\
        --add-thumbnail-title \\
        --thumbnail-offset 3 \\
        --no-display" &

    wait

done
