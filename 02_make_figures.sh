#!/bin/bash

source 00_setup.sh

mkdir -p $fig_dir

plot_codes=(
    $sh 10_plot_AR_statistics.sh "BLANK"
    $sh 11_plot_atm_frc.sh "BLANK"
    $sh 21_plot_scatter.sh "BLANK"
    $sh 31_plot_dTdt_stat_decomp.sh "BLANK"
    $sh 41_plot_Gterms_breakdown_selected_point.sh "BLANK"
    $sh 51_plot_Gterms_breakdown.sh "BLANK"
    $sh 61_plot_additional_maps.sh "BLANK"
)


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
