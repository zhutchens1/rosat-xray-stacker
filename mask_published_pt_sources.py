import numpy as np
import pandas as pd
from astropy.wcs import WCS, utils as wcsutils
from astropy.io import fits
import astropy.units as uu
from astropy.coordinates import SkyCoord
from photutils.apertures import CircularAperture

def mask_individual_image(img_arr,wcs,ra,dec,extent):
    # get range of RA/Dec to feed in 
    in_image_sel = ...
    source_in_image_skycoord = SkyCoord(ra=ra[in_image_sel]*uu.deg, dec=dec[in_image_sel]*uu.deg)
    source_xpos, source_ypos = wcsutils.skycoord_to_pixel(source_in_image_skycoord, wcs)
    xypos = np.array([source_xpos,source_ypos]).T
    apertures = CircularAperture(xypos,r=extent[in_image_sel])
    masks = apertures.to_mask(method='center')
    newimage = np.zeros_like(img_arr)
    final_mask = np.sum(np.array([msk.to_image(shape=((img_arr.shape[0],img_arr.shape[1]))) for msk in masks]),axis=0)
    replacesel = (final_mask>0)
    newimage[~replacesel]=image[~replacesel]
    return newimage

def generate_masked_images(indirec,outdirec,ra,dec,extent):
    files = os.listdir(indirec)
    for ff in files:
        hdulist = fits.open(indrec+ff)
        wcs = WCS(hdulist[0])
        image = hdulist[0].data
        masked_im = mask_individual_image(image,wcs,ra,dec,extent)
        hdulist[0].data=masked_im
        hdulist.writeto(outdirec+ff,overwrite=True)
        hdulist.close()

if __name__=='__main__':
    x = pd.read_hdf('rass2rxs.hdf5')
    ra = np.array(x.ra)
    dec = np.array(x.dec)
    extent = np.array(x.source_extent)

    
