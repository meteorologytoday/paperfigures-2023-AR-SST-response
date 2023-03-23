#!/bin/bash


echo "Making output directory 'final_figures'..."
mkdir final_figures


echo "Making final figures... "


# Merging two sub-figures
convert \( \
    \( figures/AR_freq_std.png \) \
    \( figures/AR_EOF.png \) \
    \( figures/AR_EOF_timeseries.png \) -gravity west -append \
    \) \( figures/zonal_mean_AR_forcing.png \) -gravity west +append \
     figures/merged-EOF-forcing.png

name_pairs=(
    merged-EOF-forcing.png fig01.png
)

N=$(( ${#name_pairs[@]} / 2 ))
echo "We have $N figure(s) to rename."
for i in $( seq 1 $N ) ; do
    src_file="${name_pairs[$(( (i-1) * 2 + 0 ))]}"
    dst_file="${name_pairs[$(( (i-1) * 2 + 1 ))]}"
    echo "$src_file => $dst_file"
    cp figures/$src_file final_figures/$dst_file 
done

echo "Done."
