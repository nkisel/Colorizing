# CS194-26: Project 1
# Converts an image containing stacked red, green, and blue
# negatives into one color image.

import numpy as np
import skimage as sk
import skimage.exposure as ske
import skimage.io as skio
import skimage.transform as skt
from skimage.filters import roberts
from os import listdir
from os.path import isfile, join

def ssd(src, dst):
    """ Sum of squared differences between two images. """
    return sum(sum((src[len(src) // 12 : 11 * len(src) // 12] - dst[len(dst) // 12 : 11 * len(dst) // 12]) ** 2))

def ncc(src, dst):
    """ Normalized cross correlation of two images. """
    src = src.flatten() / np.sum(np.abs(src[len(src) // 12 : 11 * len(src) // 12]))
    dst = dst.flatten() / np.sum(np.abs(dst[len(dst) // 12 : 11 * len(dst) // 12]))
    return np.dot(src, dst)

def displacement(src, dst, x, y, cc_func=ssd):
    """
    Determine a similarity score between SRC & DST based on CC_FUNC.
    """
    dst = np.roll(dst, x, axis=0)
    dst = np.roll(dst, y, axis=1)
    return cc_func(src, dst)

def contrast(im):
    """
    Return a re-contrasted image.
    Don't re-contrast if the image already has absolute white & black pixels.
    """
    dim, bright = sk.dtype_limits(im)
    if dim == 0 and bright == 255:
        return im
    return ske.exposure.equalize_hist(im)
    

def align(src, dst, window = 15):
    """ 
    Align the DST channel with the SRC channel.
    Return the displacements between offsets & best offset.

    """
    displacements = []
    best_d = np.inf
    # Consider resizing from absolute 30-width displacement to image size based displacement
    for y in range(-window, window, 1): # For each column in the image,
        for x in range(-window, window, 1): # For each row in the image,
            d = displacement(src, dst, x, y, cc_func=ssd)
            if d < best_d:
                best_d = d
                best_x = x
                best_y = y
            displacements.append(d)
    return best_x, best_y, displacements

def pyramid(src, dst):
    """
    Until DST has a width under 150 pixels, scale down each negative
    by a factor of four. Then, calculate the estimated offset from the blue 
    channel per interval, and re-align the image more accurately with each 
    rescaling.

    For images under a width of 400 pixels (all provided jpgs), use the single-scale approach.

    Return DST after realigning it.
    """

    if len(src) < 400:
        x, y, _ = align(src, dst, window = 15)
        dst = np.roll(dst, int(x), axis=0)
        dst = np.roll(dst, int(y), axis=1)
        return dst, x, y

    rescaled_src = [src]
    rescaled_dst = [dst]
    rescales = 0
    while len(rescaled_dst[len(rescaled_dst) - 1]) > 150:
        rescales += 1
        rescaled_src.append(skt.rescale(rescaled_src[len(rescaled_dst) - 1], 0.25, anti_aliasing=False))
        rescaled_dst.append(skt.rescale(rescaled_dst[len(rescaled_dst) - 1], 0.25, anti_aliasing=False))
    
    scale_factor = 4 ** rescales

    # The (x, y) offset in the original unscaled image.
    absolute_x = 0
    absolute_y = 0
    for i in range(len(rescaled_src) - 1, -1, -1):
        # Roll the constituent parts to our best knowledge.
        rescaled_dst[i] = np.roll(rescaled_dst[i], int(absolute_x // scale_factor), axis=0)
        rescaled_dst[i] = np.roll(rescaled_dst[i], int(absolute_y // scale_factor), axis=1)

        # Find the best SRC & DST alignment at this scale.
        x, y, _ = align(rescaled_src[i], rescaled_dst[i], window = 5)

        # Update the absolute scaling factors with our new estimate.
        absolute_x = absolute_x + (x * scale_factor)
        absolute_y = absolute_y + (y * scale_factor)
        scale_factor /= 4
    
    return rescaled_dst[0], absolute_x, absolute_y


def process_inputs(edge_detect = False, increase_contrast = False):
    """
    Processes each file in the ./in/ folder & outputs it into the ./out/ folder.
    
    EDGE_DETECT determines whether edge detection will be used, and
    INCREASE_CONTRAST determines if images will be recolored before matching.
    """

    path = "./in/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    print("Processing ", files)
    for file in files:

        # read in the image
        im = skio.imread(path + file)

        if increase_contrast:
            im = contrast(im)

        # convert to double (might want to do this later on to save memory)    
        im = sk.img_as_float(im)
            
        # compute the height of each part (just 1/3 of total)
        height = np.floor(im.shape[0] / 3.0).astype(int)

        # separate color channels
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        if edge_detect:
            orig_b = b
            orig_g = g
            orig_r = r
            b = roberts(b)
            g = roberts(g)
            r = roberts(r)

        ag, green_x, green_y = pyramid(b, g)
        ar, red_x, red_y = pyramid(b, r)

        if edge_detect:
            b = orig_b
            ag = np.roll(orig_g, int(green_x), axis=0)
            ag = np.roll(ag, int(green_y), axis=1)
            ar = np.roll(orig_r, int(red_x), axis=0)
            ar = np.roll(ar, int(red_y), axis=1)

        # create a color image
        im_out = np.dstack([ar, ag, b])

        # save the image
        fname = './out/{0}.jpg'.format(file)
        skio.imsave(fname, im_out)

        print("Processed {:>22}   Green: ({:+}, {:+})   Red: ({:+}, {:+})"\
            .format(file,
                    int(green_x), int(green_y),
                    int(red_x),   int(red_y)
            )
        )

        # display the image
        # skio.imshow(im_out)
        # skio.show()
        
process_inputs(edge_detect=False, increase_contrast=False)