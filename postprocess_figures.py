from PIL import Image
import os

def toImagesObjects(images):

    _images = []

    for i, image in enumerate(images): 
        
        if isinstance(image, Image.Image):
            _images.append(image)
        else:
            _images.append(Image.open(image))

    return _images




def concatImages(images, direction):
   
    images = toImagesObjects(images)

    widths, heights = zip(*(i.size for i in images))
    
    if direction == "horizontal":
        
        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]

    
    elif direction == "vertical":
        
        max_width    = max(widths)
        total_height = sum(heights)

        new_im = Image.new('RGB', (max_width, total_height))
       
        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]


    else:

        raise Exception("Unknown direction: %s" % (str(direction),))


    return new_im




fig_dir="figures"
finalfig_dir="final_figures"

new_img = concatImages([
    os.path.join(fig_dir, "G_terms_atmocn-ocn_2.png"), 
    os.path.join(fig_dir, "G_terms_atm_2.png"), 
], "horizontal")

new_img.save(os.path.join(fig_dir, "merged-G_terms_map_breakdown.png"), format="PNG")
