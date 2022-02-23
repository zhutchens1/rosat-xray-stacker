import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder, CircularAperture
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree, distance_matrix

def mask_point_sources(imgfiledir, scs_cenfunc=np.mean, scs_sigma=3, scs_maxiters=2, smoothsigma=1.0,\
                    starfinder_fwhm=3, starfinder_threshold=8, mask_aperture_radius=5, imagewidth=300,\
                    imageheight=300, examine_result=False):
    """
    Mask point sources in a series of X-ray FITS images.
    This code is a copy of the main function in rosat_xray_
    stacker.mask_point_sources, meant to optimize the param-
    eters. It does not output images, and instead outputs a
    dataframe containing the positions and fluxes of sources.

    Parameters
    ------------------
    imgfiledir : str
        File path where raw images are stored (use a trailing /; e.g. "/home/images/").
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

    """
    imagedfs=[]
    assert callable(scs_cenfunc), "Argument `cenfunc` for sigma_clipped_stats must be callable."
    imagefiles = os.listdir(imgfiledir)
    for i,imgpath in enumerate(imagefiles): ### iterate by group and not by image!!
        print('processing '+imgpath)
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
            tmpdf = table.to_pandas()
            tmpdf['image']=int(imgpath[13:-5])
            imagedfs.append(tmpdf)

            # examine, if requested
            if examine_result:
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
    return pd.concat(imagedfs)

def generate_synthetic_images(baseimgpath, outpath, Noutput, nsourcedist,\
    radii_dist, source_ampl, xbounds, ybounds, maskwidth=10, examine=False):
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
        **This parameter must be an even integer.**
    examine : bool, default False
        if True, each synthetic image will be opened using matplotlib.pyplot.imshow
        during each iteration.

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
            gaussx = np.exp(-1/sourceradii[jj]**2. * (x_range - maskwidth/2.)**2.)
            gauss2D = gaussx*gaussx[:,None]
            gauss2D = sourcefluxes[jj]*gauss2D # gauss 2D has value 1 at mean, so this raises overall flux level
            newimage[ypositions[jj]-maskwidth//2:ypositions[jj]+maskwidth//2, xpositions[jj]-maskwidth//2:xpositions[jj]+maskwidth//2]=gauss2D
        if examine:
            plt.figure()
            plt.imshow(newimage)
            plt.show()
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

def compare_dataframes(daodf, syndf, tol=3):
    """
    Compare the dataframe of synthetic sources with the dataframe
    returned from the DAO wrapper, i.e. found sources. Dataframe
    keys are hardcoded in this function with the index being `image`.

    Parameters
    ----------------
    daodf : pandas.DataFrame object
        Dataframe returned from mask_point_sources.
    syndf : pandas.DataFrame object
        Dataframe saved by generate_synthetic_images.
    tol : real
        Tolerance for which sources will be considered a match.
        Units in pixels.

    Returns
    -----------------
    tpr : float
        Fraction of "real" sources (from synthetic images) recovered
        by the point source removal algorithm; N(real)/len(syndf).
    fpr : float
        Fraction of identified point sources that are false positives,
        i.e. not included in the synthetic images; N(false)/len(daodf).
    """
    truepositives=0
    falsepositives=0
    imageids = np.unique(np.array(syndf.image))
    for real_source_id in imageids:
        daoimage = daodf[daodf.image==real_source_id]
        synimage = syndf[syndf.image==real_source_id]
        synx, syny = np.array(synimage.xpos), np.array(synimage.ypos)
        daox, daoy = np.array(daoimage.xcentroid), np.array(daoimage.ycentroid)
        if len(daoimage)==0 and len(synimage)>0:
            print('found nothing :(') # it found nothing
        elif len(synimage)==1 and len(daoimage)==1:
            dist = np.sqrt((synx-daox)*(synx-daox) + (syny-daoy)*(syny-daoy))
            match = int(dist<tol)
            truepositives+=match
            falsepositives+=(1-match)
        elif len(synimage)>=len(daoimage):
            synX = np.array([synx,syny]).T
            daoX = np.array([daox,daoy]).T
            syntree=KDTree(synX)
            daotree=KDTree(daoX)
            dists=distance_matrix(synX,daoX)
            truepositives += len(dists[dists<tol])
        else:
            # now you have >=1 false pos.
            # shouldn't happen if SNR cut good
            synX = np.array([synx,syny]).T
            daoX = np.array([daox,daoy]).T
            syntree=KDTree(synX)
            daotree=KDTree(daoX)
            dists=distance_matrix(daoX,synX)
            ntrue = len(dists[dists<tol])
            truepositives += ntrue
            falsepositives += (len(daox)-ntrue)
    # compute tpr, fpr
    print(truepositives)
    frac_sources_recovered = truepositives/len(syndf)
    frac_falsepos = falsepositives/len(daodf)
    return frac_sources_recovered, frac_falsepos 


def gen_param_grids(imgfolder, synfile, smooth_kernel_size,SNR_threshold,fwhm):
    """
    Return matrices containing the true positive and false
    positive rates matching a grid of test parameters.

    Parameters
    ---------------
    imgfolder : str
        Path to folder where synthetic images are stored.
    synfile : str
        Path to CSV file where synthetic point sources info is stored,
        from generate_synthetic_images.
    smooth_kernel_size : array_like
        Scales of Gaussian smoothing kernels to implement
        before point source removal, size M.
    SNR_threshold : array_like
        List of SNR thresholds to try (e.g., [3,4,5]), size N.
    fwhm : array_like
        List of aperture FWHM values to try, size K.

    Returns
    ----------------
    tpr_matrix, fpr_matrix : np.array
        Matrices of size (M,N,K) that contain the true positive
        fraction and false positive fraction for every combination
        of the input parameters.
    """
    smooth_kernel_size=np.array(smooth_kernel_size)
    SNR_threshold=np.array(SNR_threshold)
    fwhm=np.array(fwhm)
    MM = len(smooth_kernel_size)
    NN = len(SNR_threshold)
    KK = len(fwhm)
    tpr_matrix = np.zeros((MM,NN,KK))
    fpr_matrix = np.zeros_like(tpr_matrix)
    for ii in range(0,MM):
        for jj in range(0,NN):
            for kk in range(0,KK):
                sources = mask_point_sources(imgfolder, smooth_sigma=smooth_kernel_size[ii],\
                    starfinder_threshold=SNR_threshold[jj], starfinder_fwhm=fwhm[kk]) 
                tpr, fpr = compare_dataframes(sources, pd.read_csv(synfile))
                tpr_matrix[ii,jj,kk]=tpr 
                fpr_matrix[ii,jj,kk]=fpr
    return tpr_matrix, fpr_matrix

######################################################################
######################################################################
######################################################################

if __name__=='__main__':
    base_image = 'RASS-Int_Hard_grp112.0_ECO11873.fits'
    generate_synthetic_images(base_image, '/srv/scratch/zhutchen/synthetic_rass/', Noutput=100,\
         nsourcedist=[1,2,3,4,5,6,7,8],\
         radii_dist=[2,3,4,5],\
         source_ampl=np.random.normal(3e-2, 5e-3, size=100),\
         xbounds=[20,300-20], ybounds=[20,300-20], maskwidth=16, examine=False)

    sources = mask_point_sources('/srv/scratch/zhutchen/synthetic_rass/', starfinder_threshold=5)
    tpr,fpr=compare_dataframes(sources, pd.read_csv("syntheticsources.csv"))
    print(tpr,fpr)

