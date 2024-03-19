from PIL import Image

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

