#!/bin/bash

source 00_setup.sh

AR_algo=ANOM_LEN


#for suffix in "" "_500m"; do

for suffix in "" ; do

    wm_str="Oct-Nov Dec-Jan Feb-Mar"
    count=1

    input_dir=${diagdata_dir}${suffix}/climanom_${yrng_str}/${AR_algo}

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames dMLTdt BLANK SFCWIND MLD \\
        --output $fig_dir/G_terms_atmocn${suffix}_${count}.png \\
        --time $wm_str \\
        --no-display" &

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames MLG_frc MLG_frc_sw MLG_frc_lw MLG_frc_sh MLG_frc_lh MLG_frc_fwf \\
        --output $fig_dir/G_terms_atm${suffix}_${count}.png \\
        --time $wm_str \\
        --no-display" &

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames MLG_nonfrc MLG_adv MLG_vdiff MLG_ent MLG_hdiff \\
        --output $fig_dir/G_terms_ocn${suffix}_${count}.png \\
        --time $wm_str \\
        --no-display" &

    wait
    
    wm_str="Oct-Mar"

    count=$(( count + 1 ))
    echo "Plotting time: $wm_str"

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames dMLTdt BLANK SFCWIND MLD \\
        --output $fig_dir/G_terms_atmocn${suffix}_${count}.png \\
        --time $wm_str \\
        --add-thumbnail-title \\
        --mark-box strong_marine_warming \\
        --no-display" &


    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames MLG_frc MLG_frc_sw MLG_frc_lw MLG_frc_sh MLG_frc_lh MLG_frc_fwf \\
        --output $fig_dir/G_terms_atm${suffix}_${count}.png \\
        --time $wm_str \\
        --add-thumbnail-title \\
        --thumbnail-offset 3 \\
        --no-display" &

    eval "python3 $src_dir/plot_G_terms_map.py \\
        --input-dir $input_dir \\
        --varnames MLG_nonfrc MLG_adv MLG_vdiff MLG_ent MLG_hdiff \\
        --output $fig_dir/G_terms_ocn${suffix}_${count}.png \\
        --time $wm_str \\
        --add-thumbnail-title \\
        --thumbnail-offset 9 \\
        --no-display" &


    wait

done
