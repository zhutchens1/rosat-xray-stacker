import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import gaussian
from scipy.stats import multivariate_normal
from photutils.datasets import make_random_gaussians_table, make_gaussian_sources_image
import pandas as pd

def mask_point_sources(self, imgfiledir, outfiledir, scs_cenfunc=np.mean, scs_sigma=3, scs_maxiters=2, smoothsigma=1.0,\
                    starfinder_fwhm=3, starfinder_threshold=8, mask_aperture_radius=5, imagewidth=300,\
                    imageheight=300, examine_result=False):
    """
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
    for grp in self.grpid:
        pass
    for i,imgpath in enumerate(imagefiles): ### iterate by group and not by image!!
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
        savepath=outfiledir+imgpath#[:-5]+"_pntsourcesremoved.fits"
        hdulist.writeto(savepath)

def generate_synthetic_images(baseimgpath, outpath, Noutput, nsourcedist,\
    radii_dist, source_ampl, xbounds, ybounds, maskwidth=10):
    """
    Generate synthetic RASS images containing synethetic point sources,
    to test the recovery of point source removal.

    Parameters
    ------------------
    baseimagpath : str
        Path to file where base image is located. This is the 
        image for which point sources will be added, so it should
        not contain bright sources or large artefacts.
    outpath : str
        Path to folder where synthetic images should be written.
    Noutput: int
        Number of synthetic images to generate.
    nsourcedist : array_like
        Distribution of N_sources from which the number of synthetic
        sources is determined. A distribution of [2,3,4], for example,
        will mean that each synthetic image contains between 2 and 4
        synthetic sources. To hold constant, pass as one element array,
        e.g. [2].    
    radii_dist : array_like
        Distribution of radii (in pixels) from which synthetic source radii
        should be drawn.
    source_ampl : array_like
        Distribution of source amplitudes (pixel values) from which synthetic
        source radii should be drawn.
    xbounds : 2-element tuple
        Specifies portion of base image where synthetic sources can be introduced. 
        Each element of the tuple should represent an integer pixel location limiting
        source introduction, e.g. xbounds=[20,180] would introduce sources only on the 
        range x=20 to x=180 px.
    ybounds : 2-element tuple
        Specifies portion of y-axis in image where synthetic sources can be introduced,
        see description of 'xbounds'
    maskwidth : int, default 10 px
        Width of masks containing point sources that should be introduced to images.

    Returns 
    ------------------
    ptsrc_dir = pandas.DataFrame object
        Directory of point sources injected into each copy of the base image,
        indexed by the synethetic image ID, which is also located in each
        image's file handle.
    """
    Nsources_per_image = np.random.choice(nsourcedist, size=Noutput)
    hdulist = fits.open(baseimgpath, memap=True)
    baseimg = hdulist[0].data
    x_range=np.arange(0,maskwidth,1)

    output_index=[]
    output_radii=[]
    output_ampl=[]
    output_xpos=[]
    output_ypos=[]
    for ii in range(0,Noutput):
        xpositions=np.random.choice(np.arange(xbounds[0],xbounds[1],1), size=Nsources_per_image[ii])
        ypositions=np.random.choice(np.arange(ybounds[0],ybounds[1],1), size=Nsources_per_image[ii])
        sourceradii = np.random.choice(radii_dist, size=Nsources_per_image[ii]) # px
        sourcefluxes = np.random.choice(source_ampl, size=Nsources_per_image[ii]) # px values
        newimage = baseimg*1
        
        output_index.append(ii*np.ones(Nsources_per_image[ii],dtype=int))
        output_radii.append(sourceradii)
        output_ampl.append(sourcefluxes)
        output_ypos.append(ypositions)
        output_xpos.append(xpositions)

        for jj in range(0,Nsources_per_image[ii]):
            gaussx = sourcefluxes[jj] * np.exp(-1/sourceradii[jj]*(x_range - maskwidth/2.)**2.)
            gauss2D = gaussx*gaussx[:,None]
            newimage[xpositions[jj]-maskwidth//2:xpositions[jj]+maskwidth//2, ypositions[jj]-maskwidth//2:ypositions[jj]+maskwidth//2]=gauss2D
        hdulist[0].data=newimage
        hdulist.writeto(outpath+"syntheticRASS{a}".format(a=ii)+".fits", overwrite=True)

    output_index=np.concatenate(output_index)
    output_radii=np.concatenate(output_radii)
    output_ampl=np.concatenate(output_ampl)
    output_ypos=np.concatenate(output_ypos)
    output_xpos=np.concatenate(output_xpos)
    ptsrc_dir = pd.DataFrame(np.array([output_index,output_radii,output_ampl,output_ypos,output_xpos]).T,\
        columns=['image','radius_px','ampl','ypos','xpos'])
    ptsrc_dir.to_csv("syntheticsources.csv", index=False)

######################################################################
######################################################################
######################################################################

if __name__=='__main__':
    base_image = 'RASS-Int_Hard_grp112.0_ECO11873.fits'
    generate_synthetic_images(base_image, '/srv/scratch/zhutchen/synthetic_rass/', Noutput=10, nsourcedist=[2,3,4],\
        radii_dist=[2,3,4], source_ampl=[1,2,3], xbounds=[20,300-20], ybounds=[20,300-20])
