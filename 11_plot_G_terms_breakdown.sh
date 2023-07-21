#!/bin/bash


source 00_setup.sh


wm_str="1 2 3 4 5 6"
count=1
AR_algo=ANOM_LEN

    
input_dir=$diagdata_dir/climanom_${yrng_str}/${AR_algo}

eval "python3 $src_dir/plot_G_terms_map.py \\
    --input-dir $input_dir \\
    --varnames dMLTdt BLANK SFCWIND MLD \\
    --output $fig_dir/G_terms_atmocn_${count}.png \\
    --watermonths $wm_str \\
    --no-display" &

eval "python3 $src_dir/plot_G_terms_map.py \\
    --input-dir $input_dir \\
    --varnames MLG_frc MLG_frc_sw MLG_frc_lw MLG_frc_sh MLG_frc_lh MLG_frc_fwf \\
    --output $fig_dir/G_terms_atm_${count}.png \\
    --watermonths $wm_str \\
    --no-display" &

eval "python3 $src_dir/plot_G_terms_map.py \\
    --input-dir $input_dir \\
    --varnames MLG_nonfrc MLG_adv MLG_vdiff MLG_ent MLG_hdiff \\
    --output $fig_dir/G_terms_ocn_${count}.png \\
    --watermonths $wm_str \\
    --no-display" &

wait

wm_str="7"

count=$(( count + 1 ))
echo "Plotting watermonths: $wm_str"

eval "python3 $src_dir/plot_G_terms_map.py \\
    --input-dir $input_dir \\
    --varnames dMLTdt MLG_frc MLG_nonfrc \\
    --output $fig_dir/G_terms_atmocn_${count}.png \\
    --watermonths $wm_str \\
    --add-thumbnail-title \\
    --no-display" &


eval "python3 $src_dir/plot_G_terms_map.py \\
    --input-dir $input_dir \\
    --varnames MLG_frc_sw MLG_frc_lw MLG_frc_sh MLG_frc_lh MLG_frc_fwf \\
    --output $fig_dir/G_terms_atm_${count}.png \\
    --watermonths $wm_str \\
    --add-thumbnail-title \\
    --thumbnail-offset 3 \\
    --no-display" &


eval "python3 $src_dir/plot_G_terms_map.py \\
    --input-dir $input_dir \\
    --varnames MLG_adv MLG_vdiff MLG_ent MLG_hdiff \\
    --output $fig_dir/G_terms_ocn_${count}.png \\
    --watermonths $wm_str \\
    --add-thumbnail-title \\
    --thumbnail-offset 8 \\
    --no-display" &


wait
