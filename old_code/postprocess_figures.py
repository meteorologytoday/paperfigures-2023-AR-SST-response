from PIL import Image
import os
import argparse
import postprocess_tools
from pdf2image import convert_from_path

parser = argparse.ArgumentParser(
                    prog = 'postprocess_figures.py',
                    description = 'Use PIL to generate combined image.',
)

parser.add_argument('--input-dir',  type=str, help='Input directory', required=True)
parser.add_argument('--output-dir', type=str, help='Output directory', required=True)
args = parser.parse_args()
print(args)

# ==================================================

img_left = postprocess_tools.concatImages([
    os.path.join(args.input_dir, "AR_freq_std.png"),
    os.path.join(args.input_dir, "AR_EOF.png"),
], "vertical")

img_right = Image.open(os.path.join(args.input_dir, "atmsfc_2.png"))
img_right = img_right.crop((50, 0, *img_right.size))


new_img = postprocess_tools.concatImages(
    [ img_left,  img_right],
    "horizontal",
)


new_img.save(os.path.join(args.output_dir, "merged-EOF-forcing.png"), format="PNG")



# ==================================================
new_img = postprocess_tools.concatImages([
    os.path.join(args.input_dir, "G_terms_atmocn-ocn_2.png"), 
    os.path.join(args.input_dir, "G_terms_atm_2.png"), 
], "horizontal")

new_img.save(os.path.join(args.output_dir, "merged-G_terms_map_breakdown.png"), format="PNG")


# ==================================================
new_img = postprocess_tools.concatImages([
    os.path.join(args.input_dir, "analysis_mldandcld__2.png"),
    os.path.join(args.input_dir, "analysis_advbkdn__2.png"),
], "horizontal")

new_img.save(os.path.join(args.output_dir, "merged-additional-analysis.png"), format="PNG")

# ==================================================
new_img = postprocess_tools.concatImages([
    os.path.join(args.input_dir, "dTdt_scatter_ALL_NPAC_a.png"),
    os.path.join(args.input_dir, "dTdt_scatter_ALL_NPAC_b.png"),
], "horizontal")

new_img.save(os.path.join(args.output_dir, "merged-dTdt_scatter.png"), format="PNG")

# ======
for box_name in ["MAXOCNIMPACT",]: 
    for condition in ["AR", "clim"]:

        new_img = postprocess_tools.concatImages([
            os.path.join(args.input_dir, "Gterms_pt_%s_atmocn_%s_1.png" % (box_name, condition,)),
            os.path.join(args.input_dir, "Gterms_pt_%s_atm_%s_1.png" % (box_name, condition,)),
            os.path.join(args.input_dir, "Gterms_pt_%s_ocn_%s_1.png" % (box_name, condition,)),
        ], "horizontal")

        new_img.save(os.path.join(args.output_dir, "merged-Gterms_pt_%s_%s.png" % (box_name, condition,)), format="PNG")


