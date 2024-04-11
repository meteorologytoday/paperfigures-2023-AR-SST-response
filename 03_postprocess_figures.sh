#!/bin/bash

source 00_setup.sh

if [ -d "$finalfig_dir" ] ; then
    echo "Output directory '${finalfig_dir}' already exists. Do not make new directory."
else
    echo "Making output directory '${finalfig_dir}'..."
    mkdir $finalfig_dir
fi



echo "Making final figures... "

python3 postprocess_figures.py --input-dir $fig_dir --output-dir $fig_dir

name_pairs=(
    merged-EOF-forcing.png                 fig01.png
    merged-dTdt_scatter.png                fig02.png
    dTdt_stat_ALL_NPAC.pdf                 fig03.pdf
    merged-Gterms_pt_MAXOCNIMPACT_AR.png   fig04.png
    merged-G_terms_map_breakdown.png       fig05.png
    merged-additional-analysis.png         fig06.png
    merged-Gterms_pt_MAXOCNIMPACT_clim.png figS01.png
    G_terms_atm_1.png                      figS02.png
    G_terms_ocn_1.png                      figS03.png
    G_terms_atmocn_1.png                   figS04.png
    analysis_cloudcover__2.png             figS05.png
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
