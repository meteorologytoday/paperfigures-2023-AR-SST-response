#!/bin/bash

source 00_setup.sh

py=python3
sh=bash


plot_codes=(
    $py $src_dir/plot_AR_freq_with_std.py "--input $diagdata_dir/AR_interannual_statistics_ANOM_LEN_${yrng_str}.nc --output $fig_dir/AR_freq_std.png --no-display"
    $py $src_dir/plot_EOF_analysis.py "--input $diagdata_dir/EOF.nc --input-NINO $climidx_dir/NINO34.nc --input-PDO $climidx_dir/PDO.nc --output-EOF $fig_dir/AR_EOF.png --output-timeseries $fig_dir/AR_EOF_timeseries.png --no-display"
    $py $src_dir/plot_AR_basic_diagnostics.py "--input $diagdata_dir/AR_simple_statistics_${yrng_str}.nc --output $fig_dir/zonal_mean_AR_forcing.png --no-display"
    $sh 12_plot_G_terms_breakdown.sh "BLANK" 
    $sh 13_plot_atm_frc.sh "BLANK"
    $sh 14_plot_additional_maps.sh "BLANK"
    $sh 22_plot_scatter.sh "BLANK"
    $sh 21_plot_Gterms_breakdown_latlon_box.sh "BLANK"
)

#plot_codes=(
#    $py $src_dir/plot_AR_freq_with_std.py "--input $diagdata_dir/AR_interannual_statistics_ANOM_LEN_${yrng_str}.nc --output $fig_dir/AR_freq_std.png --no-display"
#) 

mkdir $fig_dir

N=$(( ${#plot_codes[@]} / 3 ))
echo "We have $N file(s) to run..."
for i in $( seq 1 $(( ${#plot_codes[@]} / 3 )) ) ; do
    PROG="${plot_codes[$(( (i-1) * 3 + 0 ))]}"
    FILE="${plot_codes[$(( (i-1) * 3 + 1 ))]}"
    OPTS="${plot_codes[$(( (i-1) * 3 + 2 ))]}"
    echo "=====[ Running file: $FILE ]====="
    set -x
    eval "$PROG $FILE $OPTS" & 
done


wait

echo "Figures generation is complete."
echo "Please run 03_postprocess_figures.sh to postprocess the figures."
