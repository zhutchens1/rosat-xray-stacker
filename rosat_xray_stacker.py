"""
Stack diffuse X-ray emission in galaxy groups from ROSAT All-Sky Survey Data.
Author: Zack Hutchens (G3 subteam with Kelley Hess + Andrew Baker)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.skyview import SkyView as sv
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder, CircularAperture
from scipy.ndimage import gaussian_filter
import os
import sys

from skyview_downloader import download_images_java


def mask_point_sources(imgfiledir, outfiledir, scs_cenfunc=np.mean, scs_sigma=3, scs_maxiters=2, smoothsigma=1.0,\
                       starfinder_fwhm=3, starfinder_threshold=8, mask_aperture_radius=5, imagewidth=300,\
                        imageheight=300, examine_result=False):
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
    scs_cenfunc : callable, default np.mean
        Function to set the center value for sigma-clipping statistics, use np.mean or np.median.
    scs_sigma : int, default 3
        Number of standard deviations to use as the upper/lower clipping limit.
    scs_maxiters : int, default 2
        Maximum number of sigma-clipping iterations to perform.
    smoothsigma : float, default 1.0
        Standard deviation of Gaussian kernel used for smoothing, prior to point source detection.
        If this parameter is None, then no smoothing is performed.
    starfinder_fwhm : int, default 3
        The full width-half maximum (FWHM) of the major axis of Gaussian kernel (in pixels) used by DAOStarFinder.
    starfinder_threshold : int, default 8
        From DAOStarFinder, "the absolute image value above which to select sources".
    mask_aperture_radius : int, default 5
        Radius of applied masks (pixels).
    imagewidth : int, default 300
        Width of masked image for DAOStarFinder; should match size of input image.
    imageheight : int, default 300
        Height of masked image for DAOStarFinder; should match size of input image.
    examine_result : bool, default False
        If True, display the raw image, masks, and masked image in each iteration.

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
        mean, median, std = sigma_clipped_stats(image[image!=0],sigma=scs_sigma, maxiters=scs_maxiters, cenfunc=scs_cenfunc)
        mean2, median2, std2 = sigma_clipped_stats(image,sigma=scs_sigma, maxiters=np.max([1,scs_maxiters-1]), cenfunc=scs_cenfunc)

        # smooth image before finding sources
        if smoothsigma is not None:        
            smoothimg = gaussian_filter(image, sigma=smoothsigma)
        else:
            smoothimg = np.copy(image)

        # find point sources using DAOStarFinder (photutils)
        daofind = DAOStarFinder(fwhm=starfinder_fwhm, threshold=mean+starfinder_threshold*std)
        table = daofind.find_stars(smoothimg)
        
        if table is not None:
            # create and apply masks (unless it is a diffuse bright source?)
            positions=np.transpose(np.array([table['xcentroid'],table['ycentroid']]))
            apertures=CircularAperture(positions,r=mask_aperture_radius)
            masks=apertures.to_mask(method='center')

            # Create new image
            newimage = np.zeros_like(image)
            newmask = np.zeros_like(image)
            newmask = np.sum(np.array([msk.to_image(shape=((imagewidth,imageheight))) for msk in masks]), axis=0)

            replacesel = np.logical_and(newmask>0,image>mean+std)
            newimage[replacesel] = mean2
            newimage[~replacesel] = image[~replacesel]

            # examine, if requested
            if examine_result:
                print(mean2)
                fig, ax = plt.subplots(ncols=4, figsize=(16,7))
                ax[0].imshow(image, vmin=0, vmax=np.max(image))
                #ax[0].plot(table['xcentroid'], table['ycentroid'], 'x', color='yellow')
                ax[0].set_title("Raw Image")
                ax[1].imshow(smoothimg, vmin=0, vmax=np.max(image))
                ax[1].set_title("Gaussian-smoothed Image\n Smoothing Sigma = "+str(smoothsigma))
                ax[2].imshow(newmask, cmap='binary')
                ax[2].set_title("Masks")
                ax[3].imshow(newimage, vmin=0, vmax=np.max(image))
                ax[3].set_title("Masked Image")
                plt.show()
        else:
            print('skipping '+imgpath+': no point sources found')
            newimage=np.copy(image)
        # write to file and continue
        hdulist[0].data=newimage
        savepath=outfiledir+imgpath[:-5]+"_pntsourcesremoved.fits"
        print(savepath)
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

    ############################################
    ############################################
    # (2)
     
    # (2) Image Inspection/Classification 
    # need to inspect images? yes, or say file where classifications are stored

    # (3) Image Stacking






if __name__=='__main__':
    #mask_point_sources('/srv/two/zhutchen/rosat_xray_stacker/g3rassimages/eco/', 'anywhere/', examine_result=True, starfinder_threshold=7, smoothsigma=0.5, starfinder_fwhm=3)
    #mask_point_sources('/srv/scratch/zhutchen/eco03822files/', 'anywhere/', examine_result=True, smoothsigma=None)
    mask_point_sources('/srv/scratch/zhutchen/khess_images/poor_coverage/', 'whatever/', examine_result=True, smoothsigma=0.5)
     
