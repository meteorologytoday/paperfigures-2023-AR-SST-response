#!/bin/bash


source 00_setup.sh



if [ -d "$finalfig_dir" ] ; then
    echo "Output directory '${finalfig_dir}' already exists. Do not make new directory."
else
    echo "Making output directory '${finalfig_dir}'..."
    mkdir $finalfig_dir
fi



echo "Making final figures... "

convert $fig_dir/G_terms_atmocn-ocn_2.png \
        $fig_dir/G_terms_atm_2.png \
        -gravity Northwest +append $fig_dir/merged-G_terms_map_breakdown.png

convert \
    \(  \
        \( $fig_dir/AR_freq_std.png \)  \
        \( $fig_dir/AR_EOF.png \) -gravity Northwest -append \
    \) -gravity East \(                               \
        $fig_dir/atmsfc_2.png -gravity West -chop 50x0          \
    \) -gravity Northwest +append       \
     $fig_dir/merged-EOF-forcing.png


convert \
    \( $fig_dir/analysis_mldandcld__2.png \)  \
    \( $fig_dir/analysis_advbkdn__2.png \)  \
    -gravity Northwest +append       \
     $fig_dir/merged-additional-analysis.png


convert \
    \( $fig_dir/dTdt_scatter_ALL_NPAC_a.png \)  \
    \( $fig_dir/dTdt_scatter_ALL_NPAC_b.png \)  \
    -gravity West +append       \
     $fig_dir/merged-dTdt_scatter.png




for box_name in AR_REGION_MAX OLD_AR_REGION_E MAXOCNIMPACT ; do

    for condition in AR clim ; do  
        convert \
            \( $fig_dir/Gterms_pt_${box_name}_atmocn_${condition}_1.png \)  \
            \( $fig_dir/Gterms_pt_${box_name}_atm_${condition}_1.png \)  \
            \( $fig_dir/Gterms_pt_${box_name}_ocn_${condition}_1.png \)  \
            -gravity West +append       \
             $fig_dir/merged-Gterms_pt_${box_name}_${condition}.png

    done
done
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
    dTdt_stat_ALL_NPAC.png                 fig03.png
    merged-Gterms_pt_MAXOCNIMPACT_AR.png   fig04.png
    merged-G_terms_map_breakdown.png       fig05.png
    merged-additional-analysis.png         fig06.png
    merged-Gterms_pt_MAXOCNIMPACT_clim.png figS01.png
    G_terms_atmocn_1.png                   figS02.png
    G_terms_atm_1.png                      figS03.png
    G_terms_ocn_1.png                      figS04.png
    analysis_cloudcover__2.png             figS05.png
)
#    merged-Gterms_pt_AR_REGION_W.png       figS01.png

N=$(( ${#name_pairs[@]} / 2 ))
echo "We have $N figure(s) to rename."
for i in $( seq 1 $N ) ; do
    src_file="${name_pairs[$(( (i-1) * 2 + 0 ))]}"
    dst_file="${name_pairs[$(( (i-1) * 2 + 1 ))]}"
    echo "$src_file => $dst_file"
    cp $fig_dir/$src_file $finalfig_dir/$dst_file 
done

echo "Done."
