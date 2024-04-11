#!/bin/bash
py=python3
sh=bash

dataset_date=20240410

src_dir=src
data_dir=./data$dataset_date
fig_dir=figures
finalfig_dir=final_figures
AR_algo="HMGFSC24_threshold-1998-2017"
beg_year=1993
end_year=2017
yrng_str="${beg_year}-${end_year}"
diagdata_dir=$data_dir/${yrng_str}_10N-60N-n25_100E-100W-n80
climidx_dir=$data_dir/climate_indices
annual_cnt_threshold=05

algo_suffix="_annual-cnt-threshold-${annual_cnt_threshold}"

echo "data_dir = $data_dir"
echo "AR_algo = $AR_algo"
echo "annual_cnt_threshold = $annual_cnt_threshold"

