#!/bin/bash

source 00_setup.sh

$py $src_dir/plot_AR_freq_with_std.py \
    --input $diagdata_dir/AR_interannual_statistics_${AR_algo}_${yrng_str}.nc \
    --output $fig_dir/AR_freq_std.svg \
    --no-display

$py $src_dir/plot_EOF_analysis.py \
    --input $diagdata_dir/EOF_${AR_algo}.nc --input-NINO $climidx_dir/NINO34.nc --nEOF 3 \
    --input-PDO $climidx_dir/PDO.nc --output-EOF $fig_dir/AR_EOF.svg \
    --output-timeseries $fig_dir/AR_EOF_timeseries.svg \
    --no-display

