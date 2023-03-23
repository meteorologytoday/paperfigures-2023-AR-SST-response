#!/bin/bash


source 00_setup.sh

echo "Making output directory 'final_figures'..."
mkdir $finalfig_dir


echo "Making final figures... "


# Merging two sub-figures
convert \( \
    \( $fig_dir/AR_freq_std.png \) \
    \( $fig_dir/AR_EOF.png \) \
    \( $fig_dir/AR_EOF_timeseries.png \) -gravity west -append \
    \) \( $fig_dir/zonal_mean_AR_forcing.png \) -gravity west +append \
     $fig_dir/merged-EOF-forcing.png

name_pairs=(
    merged-EOF-forcing.png fig01.png
)

N=$(( ${#name_pairs[@]} / 2 ))
echo "We have $N figure(s) to rename."
for i in $( seq 1 $N ) ; do
    src_file="${name_pairs[$(( (i-1) * 2 + 0 ))]}"
    dst_file="${name_pairs[$(( (i-1) * 2 + 1 ))]}"
    echo "$src_file => $dst_file"
    cp $fig_dir/$src_file $finalfig_dir/$dst_file 
done

echo "Done."
