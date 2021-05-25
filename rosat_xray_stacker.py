"""
Stack diffuse X-ray emission in galaxy groups from ROSAT All-Sky Survey Data.
Author: Zack Hutchens (G3 subteam with Kelley Hess + Andrew Baker)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.skyview import SkyView as sv
from astropy.io.fits import fits
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder, CircularAperture
import os
import sys

from skyview_downloader import download_images_java










def mask_point_sources(imgfiledir, outfiledir, scs_cenfunc=np.mean, scs_sigma=3, scs_maxiters=2, starfinder_fwhm=3,\
    starfinder_threshold=8, mask_aperture_radius=5, imagewidth=300, imageheight=300, examine_result=False):
    """
    Adapted from Kelley Hess
    Mask point sources in a series of X-ray FITS images.
    This code will read the raw images, find sources and
    apply masks, and write-out masked images.

    Parameters
    ------------------
    imgfiledir : str
        File path where raw images are stored (use a trailing /; e.g. "/home/images/").
    outfiledir : str
        Directory where masked images should be written (use trailing /).

    Returns
    ------------------
    None. All output images (with point sources masked) are written to `outfilepath`.

    """
    assert callable(scs_cenfunc), "Argument `cenfunc` for sigma_clipped_stats must be callable."
    imagefiles = os.listdir(imgfiledir)
    for i,imgpath in enumerate(imagefiles):
        # get image
        hdulist = fits.open(imgfiledir+imgpath)
        image = hdulist[0].data
        
        # get image stats
        mean, median, std = sigma_clipped_stats(image[image!=0],sigma=scs_sigma, iters=scs_maxiters, cenfunc=scs_cenfunc)
        mean2, median2, std2 = sigma_clipped_stats(image,sigma=scs_sigma, iters=np.max([1,scs_maxiters-1]), cenfunc=scs_cenfunc) # why this?
        
        # find point sources using DAOStarFinder (photutils)
        daofind = DAOStarFinder(fwhm=starfinder_fwhm, threshold=mean+starfinder_threshold*std)
        table = daofind.findstars(image)

        # create and apply masks (unless it is a diffuse bright source?)
        positions=(table['xcentroid'],table['ycentroid'])
        apertures=CircularAperture(positions,r=mask_aperture_radius)
        masks=apertures.to_mask(method='center')

        # Create new image
        newimage = np.zeros_like(image)
        #newmask = np.zeros_like(image)
        newmask = np.sum([msk.to_image(shape=((imagewidth,imageheight))) for msk in masks])

        replacesel = np.where(np.logical_and(newmask>0,image>mean+std))
        newimage[replacesel] = mean2
        newimage[~replacesel] = image[~replacesel]

        # examine, if requested
        if examine_result:
            pass
        # write to file and continue
        hdulist[0].data=newimage
        #hdulist.writeto()
        hdulist.close()

def rosat_xray_stacker(imgfilepath, grpra, grpdec, grpid, surveys, centralname=''):
    """
    Stack RASS X-ray data for a sample of galaxy groups.

    Parameters
    -------------------
    grpid : np.array
       Galaxy group ID numbers.

    Returns
    -------------------
    """
    # Prepare group data
    grpra=np.array(grpra)
    grpdec=np.array(grpdec)
    grpid=np.array(grpid).astype(int)
    if isinstance(surveys, str):
        surveys = [surveys]
    if not isinstance(centralname, str):
        centralname=list(centralname)

    ############################################
    ############################################
    # (1) Download X-ray images using SkyView
    ############################################
    ############################################
    
    # check that provided directory has all the right files; otherwise download them.
    dirfiles = os.listdir(imgfilepath)
    nimagesneeded = len(surveys)*len(grpra) 
    nimagesindir = np.sum([int('.fits' in fname) for fname in dirfiles])
    if nimagesindir==0:
        print('Provided directory is empty... downloading RASS images.')
        download_images_java(imgfilepath, grpra, grpdec, grpid, surveys, centralname)
    elif nimagesindir<nimagesneeded:
        print('Provided directory has too few files...')
        answ=input("Enter 'c' to re-download RASS images into provided directory (may overwrite existing files): ")
        if answ=='c':download_images_java(imgfilepath, grpra, grpdec, grpid, surveys, centralname)
        else:sys.exit()
    else:
        print('Provided directory has sufficient *.fits files -- proceeding without downloading RASS images')
     
    # (2) Image Inspection/Classification 
    # need to inspect images? yes, or say file where classifications are stored

    # (3) Image Stacking 
