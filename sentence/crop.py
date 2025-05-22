# import numpy as np
# import pandas as pd
# from PIL import Image

# def get_first_last(mask, axis: int):
#     """ Find the first and last index of non-zero values along an axis in `mask` """
#     mask_axis = np.argmax(mask, axis=axis) > 0
#     a = np.argmax(mask_axis)
#     b = len(mask_axis) - np.argmax(mask_axis[::-1])
#     return int(a), int(b)


# def crop_borders(img, crop_color):
#     np_img = np.array(img)
#     mask = (np_img != crop_color)[..., 0]  # compute a mask
#     x0, x1 = get_first_last(mask, 0)  # find boundaries along x axis
#     y0, y1 = get_first_last(mask, 1)  # find boundaries along y axis
#     return img.crop((x0, y0, x1, y1))
# # img = Image.open(image_path).convert("L") 
# img = Image.open("../images/shree_0.jpg").convert("RGB")
# img = crop_borders(img, crop_color=(0, 0, 0))
# img.save("0d34A_cropped.png")

from PIL import Image, ImageChops

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

# Example usage:
im = Image.open("../images/aaronic_19.png")
trimmed_im = trim(im)
print(trimmed_im)
trimmed_im.save("trimmed_image.png")