#!/bin/bash

py=python3
bs=bash
srcdir=src
datadir=./data
figuredir=figures


beg_year=1993
end_year=2017

yrng_str="${beg_year}-${end_year}"
diagdatadir=$datadir/${yrng_str}_10N-60N-n25_120E-120W-n60
climidxdir=$datadir/climate_indices
yrng_str="${beg_year}-${end_year}"

plot_codes=(
    $py $srcdir/plot_AR_freq_with_std.py "--input $diagdatadir/AR_interannual_statistics_${yrng_str}.nc --output $figuredir/AR_freq_std.png"
    $py $srcdir/plot_EOF_analysis.py "--input $diagdatadir/EOF.nc --input-NINO $climidxdir/NINO34.nc --input-PDO $climidxdir/PDO.nc --output-EOF $figuredir/AR_EOF.png --output-timeseries $figuredir/AR_EOF_timeseries.png"
    $py $srcdir/plot_AR_basic_diagnostics.py "--input $diagdatadir/AR_simple_statistics_${yrng_str}.nc --output $figuredir/zonal_mean_AR_forcing.png"
)

mkdir $figuredir

N=$(( ${#plot_codes[@]} / 3 ))
echo "We have $N file(s) to run..."
for i in $( seq 1 $(( ${#plot_codes[@]} / 3 )) ) ; do
    PROG="${plot_codes[$(( (i-1) * 3 + 0 ))]}"
    FILE="${plot_codes[$(( (i-1) * 3 + 1 ))]}"
    OPTS="${plot_codes[$(( (i-1) * 3 + 2 ))]}"
    echo "=====[ Running file: $FILE ]====="
    eval "$PROG $FILE $OPTS" & 
done


wait

echo "Figures generation is complete."
echo "Please run 02_postprocess_figures.sh to postprocess the figures."
