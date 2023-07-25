#!/bin/bash


source 00_setup.sh

echo "Making output directory 'final_figures'..."
mkdir $finalfig_dir


echo "Making final figures... "

convert $fig_dir/G_terms_atmocn_2.png \
        $fig_dir/G_terms_atm_2.png \
        $fig_dir/G_terms_ocn_2.png \
        -gravity Northwest +append $fig_dir/merged-G_terms_map_breakdown.png

#convert $fig_dir/G_terms_atmocn_500m_2.png \
#        $fig_dir/G_terms_atm_500m_2.png \
#        $fig_dir/G_terms_ocn_500m_2.png \
#        -gravity Northwest +append $fig_dir/merged-G_terms_map_breakdown_500m.png


# Merging two sub-figures
convert \
    \(  \
        \( $fig_dir/AR_freq_std.png -gravity South -chop 0x100 \)  \
        \( $fig_dir/AR_EOF.png -gravity North -chop 0x200 -resize 90% \) -gravity East -chop 100x0  -gravity Northwest -append \
    \) -gravity East \(                               \
        $fig_dir/atmsfc_2.png -gravity West -chop 50x0          \
    \) -gravity Northwest +append       \
     $fig_dir/merged-EOF-forcing.png

convert \
    \( $fig_dir/analysis_advbkdn__2.png \)  \
    \( $fig_dir/analysis_mldandcld__2.png \)  \
    -gravity Northwest +append       \
     $fig_dir/merged-additional-analysis.png


convert \
    \( $fig_dir/dTdt_scatter_a.png \)  \
    \( $fig_dir/dTdt_scatter_b.png \)  \
    -gravity West +append       \
     $fig_dir/merged-dTdt_scatter.png



if [ ] ; then
# This adds in the EOF timeseries. I think it is okay to just write the
# correlation with PDO and ENSO in the paper or caption
convert \
    \(  \
        \( $fig_dir/AR_freq_std.png \)  \
        \( $fig_dir/AR_EOF.png \)       \
        \( $fig_dir/AR_EOF_timeseries.png \) -gravity Northwest -append \
    \) \(                               \
        $fig_dir/atmsfc_2.png           \
    \) -gravity Northwest +append       \
     $fig_dir/merged-EOF-forcing.png
fi

name_pairs=(
    merged-EOF-forcing.png                 fig01.png
    merged-dTdt_scatter.png                fig02.png
    merged-G_terms_map_breakdown.png       fig03.png
    merged-additional-analysis.png         fig04.png
    G_terms_atm_1.png                      fig05.png
    G_terms_ocn_1.png                      fig06.png
    G_terms_atmocn_1.png                   figS01.png
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
