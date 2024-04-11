#!/bin/bash


source 00_setup.sh

for suffix in "" ; do

    wm_str="Oct-Nov Dec-Jan Feb-Mar"
    count=1

    input_dir=${diagdata_dir}${suffix}/climanom_${yrng_str}/${AR_algo}${algo_suffix}

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames MLHADVT_ag ENT_ADV MLD tcc lcc mcc hcc MLT \\
        --output $fig_dir/analysis_${suffix}_${count}.svg \\
        --time $wm_str \\
        --no-display" &

    wm_str="Oct-Mar"

    count=$(( count + 1 ))
    echo "Plotting time: $wm_str"

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames tcc sst msl \\
        --output $fig_dir/analysis_mldandcld_${suffix}_${count}.svg \\
        --time $wm_str         \\
        --add-thumbnail-title  \\
        --thumbnail-offset 0   \\
        --no-display" &

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames MLHADVT_ag MLD dTdz_b \\
        --output $fig_dir/analysis_advbkdn_${suffix}_${count}.svg \\
        --time $wm_str \\
        --add-thumbnail-title \\
        --thumbnail-offset 3 \\
        --no-display" &

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames tcc hcc mcc lcc EXFevap \\
        --output $fig_dir/analysis_cloudcover_${suffix}_${count}.svg \\
        --time $wm_str \\
        --add-thumbnail-title \\
        --no-display" &




    wait

done
