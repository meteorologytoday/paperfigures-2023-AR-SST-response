#!/bin/bash


source 00_setup.sh


count=1
for wm_str in "1 2 3 4 5 6" "7" ; do


    echo "Plotting watermonths: $wm_str"

    python3 $src_dir/plot_G_terms_map.py \
        --input-dir $diagdata_dir/climanom_${yrng_str} \
        --varnames dMLTdt \
        --output $fig_dir/G_terms_atmocn_${count}.png \
        --watermonths $wm_str \
        --no-display &


    python3 $src_dir/plot_G_terms_map.py \
        --input-dir $diagdata_dir/climanom_${yrng_str} \
        --varnames MLG_nonfrc MLG_adv MLG_vdiff MLG_ent MLG_hdiff MLD \
        --output $fig_dir/G_terms_ocn_${count}.png \
        --watermonths $wm_str \
        --no-display &

    python3 $src_dir/plot_G_terms_map.py \
        --input-dir $diagdata_dir/climanom_${yrng_str} \
        --varnames MLG_frc MLG_frc_sw MLG_frc_lw MLG_frc_sh MLG_frc_lh MLG_frc_fwf \
        --output $fig_dir/G_terms_atm_${count}.png \
        --watermonths $wm_str \
        --no-display &


    count=$(( count + 1 ))

done

wait
