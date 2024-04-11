#!/bin/bash

source 00_setup.sh

if [ -d "$finalfig_dir" ] ; then
    echo "Output directory '${finalfig_dir}' already exists. Do not make new directory."
else
    echo "Making output directory '${finalfig_dir}'..."
    mkdir $finalfig_dir
fi



echo "Making final figures... "

echo "Figure 1: Merge EOF and forcing"
svg_stack.py --direction=v $fig_dir/AR_freq_std.svg $fig_dir/AR_EOF.svg > $fig_dir/tmp1.svg
svg_stack.py --direction=h $fig_dir/tmp1.svg $fig_dir/atmsfc_2.svg > $fig_dir/merged-EOF-forcing.svg


echo "Figure 2: Merge atmocn and ocn"
svg_stack.py --direction=h $fig_dir/dTdt_scatter_ALL_NPAC_a.svg $fig_dir/dTdt_scatter_ALL_NPAC_b.svg > $fig_dir/merged-dTdt_scatter.svg

echo "Figure 3: No need to do anything"

echo "Figure 4: No need to do anything"
for box_name in MAXOCNIMPACT ; do
    for condition in AR clim ; do
        f1=$fig_dir/Gterms_pt_${box_name}_atmocn_${condition}_1.svg
        f2=$fig_dir/Gterms_pt_${box_name}_atm_${condition}_1.svg
        f3=$fig_dir/Gterms_pt_${box_name}_ocn_${condition}_1.svg
        out_f=$fig_dir/merged-Gterms_pt_${box_name}_${condition}.svg
        svg_stack.py --direction=h $f1 $f2 $f3 > $out_f
    done
done

echo "Figure 5: merging"
svg_stack.py --direction=h $fig_dir/G_terms_atmocn-ocn_2.svg $fig_dir/G_terms_atm_2.svg > $fig_dir/merged-G_terms_map_breakdown.svg

echo "Figure 6: merging"
svg_stack.py --direction=h $fig_dir/analysis_mldandcld__2.svg  $fig_dir/analysis_advbkdn__2.svg > $fig_dir/merged-additional-analysis.svg




name_pairs=(
    merged-EOF-forcing.svg                 fig01.pdf
    merged-dTdt_scatter.svg                fig02.pdf
    dTdt_stat_ALL_NPAC.svg                 fig03.pdf
    merged-Gterms_pt_MAXOCNIMPACT_AR.svg   fig04.pdf
    merged-G_terms_map_breakdown.svg       fig05.pdf
    merged-additional-analysis.svg         fig06.pdf
    merged-Gterms_pt_MAXOCNIMPACT_clim.svg figS01.pdf
    G_terms_atm_1.svg                      figS02.pdf
    G_terms_ocn_1.svg                      figS03.pdf
    G_terms_atmocn_1.svg                   figS04.pdf
    analysis_cloudcover__2.svg             figS05.pdf
)

N=$(( ${#name_pairs[@]} / 2 ))
echo "We have $N figure(s) to rename and convert into pdf files."
for i in $( seq 1 $N ) ; do
    src_file="${name_pairs[$(( (i-1) * 2 + 0 ))]}"
    dst_file="${name_pairs[$(( (i-1) * 2 + 1 ))]}"
    
    echo "$src_file => $dst_file"
    cairosvg $fig_dir/$src_file -o $finalfig_dir/$dst_file 
done

echo "Done."
