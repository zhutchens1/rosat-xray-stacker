import scipy.ndimage as nd
import numpy as np
import os
from astropy.io import fits
from progressbar import ProgressBar
import matplotlib.pyplot as plt
import time
import random
random.seed = 46
pbar=ProgressBar()

def shuffleimage(image):
    image=list(image.flatten())
    image=random.sample(image, len(image))
    return np.array(image).reshape((300,300))

def compareimages(*images):
    fig,ax= plt.subplots(ncols=len(images))
    for i,img in enumerate(images):
        ax[i].imshow(img)
    plt.show()

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = nd.zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

readpath = "/srv/scratch/zhutchen/khess_images/poor_coverage/"
writepath = "/srv/scratch/zhutchen/khess_images/poor_coverage_augmented/"
files = os.listdir(readpath)
print("Augmenting dataset to path "+writepath)
for f in pbar(files):
    if f.endswith('.fits'):
        hdulist = fits.open(readpath+f)
        image = hdulist[0].data
        savenamestart = writepath+f[:-5]

        # (0) Write original
        #hdulist.writeto(writepath+f)
        
        """
        # (1) +90 deg rotation
        hdulist[0].data = nd.rotate(image, 90)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")
 
        # (2) +180 deg rotation
        hdulist[0].data = nd.rotate(image,180)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (3) +270 deg rotation
        hdulist[0].data = nd.rotate(image,270)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (4) Vertical Flip
        hdulist[0].data = np.flipud(image)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (5) LR Flip
        hdulist[0].data = np.fliplr(image)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (6) Vertical Flip + 90 deg
        hdulist[0].data = nd.rotate(np.flipud(image),90)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (7) Vertical Flip + 180 deg
        hdulist[0].data = nd.rotate(np.flipud(image),180)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (8) Vertical Flip + 270 deg
        hdulist[0].data = nd.rotate(np.flipud(image),270)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")
        
        # (9) LR Flip + 90 deg
        hdulist[0].data = nd.rotate(np.fliplr(image),90)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (10) LR Flip + 180 deg
        hdulist[0].data = nd.rotate(np.fliplr(image),180)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (11) LR Flip + 270 deg
        hdulist[0].data = nd.rotate(np.fliplr(image),270)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")
        """
        # (1) Reorder rows and translate by 75 wrap
        x = np.copy(image)
        np.random.shuffle(x)
        hdulist[0].data = nd.shift(x,75,mode='wrap')
        hdulist.writeto(savenamestart + "rowshuffle_75wrap" + ".fits")

        # (2) Stack with flipped, shuffled version of itself
        simage = nd.fliplr(shuffleimage(image))
        hdulist[0].data = np.sum([image,simage],axis=0)/2.
        hdulist.writeto(savenamestart + "selfstackflipshuffle" + ".fits")

        # (3) Stack with a translated, flipped version of itself
        hdulist[0].data = np.sum([image, np.flipud(nd.shift(image,67,mode='wrap'))],axis=0)/2.
        hdulist.writeto(savenamestart + "selfstack_udflip67mirror" + ".fits")

        # (4) Stack with a 2nd random image that has been translated 100 pixels (wrap)
        img2index = random.randint(0,len(files)-1)
        image2 = nd.shift(fits.open(readpath+files[img2index], ignore_missing_end=True)[0].data, 100, mode='wrap')
        hdulist[0].data = np.sum([image,image2],axis=0)/2.
        hdulist.writeto(savenamestart + "stackrandom_wrap100" + ".fits")

        # (5) Stack with a 2nd random image that has been translated 75 pixels (mirror)
        img2index = random.randint(0,len(files)-1)
        image2 = nd.shift(fits.open(readpath+files[img2index], ignore_missing_end=True)[0].data, 75, mode='mirror')
        hdulist[0].data = np.sum([image,image2],axis=0)/2.
        hdulist.writeto(savenamestart + "stackrandom_mirror75" + ".fits")

        # (6) Pure Random Shuffle #1
        hdulist[0].data = shuffleimage(image)
        hdulist.writeto(savenamestart + "random1" + ".fits")

        # (7) Pure Random Shuffle #2
        hdulist[0].data = shuffleimage(image)
        hdulist.writeto(savenamestart + "random2" + ".fits")

        # (8) Wrap 68 + Vertical Flip
        hdulist[0].data = np.flipud(nd.shift(image,80,mode='wrap'))
        hdulist.writeto(savenamestart + 'wrap68vflip' + ".fits")

        # (9) Wrap 100
        hdulist[0].data = nd.shift(image,100,mode='wrap')
        hdulist.writeto(savenamestart + 'wrap100' + ".fits")

        # (10) Mirror 50 + LR flip
        hdulist[0].data = np.fliplr(nd.shift(image,50,mode='mirror'))
        hdulist[0].writeto(savenamestart + 'lrflipmirror50' + ".fits")
  
        # (11) Mirror 100
        hdulist[0].data = nd.shift(image,100,mode='mirror')

        # (12) Translate 50 pixels +x
        hdulist[0].data = nd.shift(image,50)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (13) Translate 50 pixels -x
        hdulist[0].data = nd.shift(image,-50)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (14) Translate 50 pixels +y
        hdulist[0].data = nd.shift(image,50).T
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (15) Translate 50 pixels -y
        hdulist[0].data = nd.shift(image,-50).T
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (16) Translate 50 pixels +x and 90 deg rotation
        hdulist[0].data = nd.rotate(nd.shift(image,50),90)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (17) Translate 50 pixels -x and 180 deg rotation
        hdulist[0].data = nd.rotate(nd.shift(image,-50),180)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (18) Translate 50 pixels +y and 270 deg rotation
        hdulist[0].data = nd.rotate(nd.shift(image,50).T,270)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (19) Translate +50x pixels + LR Flip
        hdulist[0].data = np.fliplr(nd.shift(image,50))
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (20) Translate -50x pixels + LR Flip
        hdulist[0].data = np.fliplr(nd.shift(image,-50))
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (21) Translate +50x pixels + LR flip
        hdulist[0].data = np.fliplr(nd.shift(image.T,-50))
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (22) Translate +50x and Vertical Flip
        hdulist[0].data = np.flipud(nd.shift(image,50))
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")
        
        # (23) Translate -50x and Vertical Flip 
        hdulist[0].data = np.flipud(nd.shift(image,-50))
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (24) Translate +50y and Vertical Flip
        hdulist[0].data = np.flipud(nd.shift(image.T,50))
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (25) 0.5 Zoom and +50x translate
        hdulist[0].data = clipped_zoom(nd.shift(image,50),0.5)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (26) 0.5 Zoom and +50x translate and Vertical Flip
        hdulist[0].data = np.flipud(clipped_zoom(nd.shift(image,50),0.5))
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

        # (27) 0.5 Zoom and -50y translate and Horizontal Flip
        hdulist[0].data = np.fliplr(clipped_zoom(nd.shift(image.T,-50),0.5))
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")
 
        # (28) 0.5 Zoom and -50x translate and 90 deg rotation
        hdulist[0].data = nd.rotate(clipped_zoom(nd.shift(image,-50),0.5), 90)
        hdulist.writeto(savenamestart + str(time.time()) + ".fits")

