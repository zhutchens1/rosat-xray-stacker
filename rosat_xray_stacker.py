"""
Stack diffuse X-ray emission in galaxy groups from ROSAT All-Sky Survey Data.
Author: Zack Hutchens
"""
from scipy import ndimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from astroquery.skyview import SkyView as sv
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder, CircularAperture
from scipy.ndimage import gaussian_filter
import os
import sys
from skyview_downloader import download_images_java
import pickle
from numba import jit, prange

def scale_image(output_coords,scale):
    return (output_coords[0]/scale+150-150/scale, output_coords[1]/scale+150-150/scale)

def get_circle(R):
    theta=np.linspace(0,2*np.pi,1000)
    x = R*np.cos(theta)
    y = R*np.sin(theta)
    return x,y

class rosat_xray_stacker:
    """
    A class for stacking ROSAT All-Sky Survey (RASS) X-ray images of galaxy
    groups, including for image retreival (from NASA SkyView), image processing,
    and stacking.

    Attributes
    -----------------
    grpid : iterable
        List of group ID numbers. Must be unique, no duplicates.
        (Must pass by group and not by galaxy.)
    grpra : iterable
        Right-ascensions of galaxy groups in decimal degrees.
        This coordinate determines the position for image extraction/stacking.
    grpdec : iterable
        Declinations of galaxy groups in decimal degrees.
        This coordinate determines the position for image extraction/stacking.
    grpcz : iterable
        Recessional velocities of groups in km/s.
    centralname : str or iterable, default None
        Name of central galaxy in each group, or other group identifier
        beyond grpid.
    surveys : iterable, default None
        Surveys from which to extract/stack images, corresponding to the
        documentation from NASA SkyView (e.g. 'RASS-Int Hard').
    """
    def __init__(self, grpid, grpra, grpdec, grpcz, centralname='', surveys=None):
        self.grpid = np.array(grpid)
        self.grpra = np.array(grpra)
        self.grpdec = np.array(grpdec)
        self.grpcz = np.array(grpcz)
        if isinstance(surveys, str):
            surveys = [surveys]
        if not isinstance(centralname, str):
            centralname=list(centralname)
        self.surveys = surveys
        self.centralname = centralname
        self.goodflag = np.full(len(self.grpid),1)
        self.detection = np.zeros(len(self.grpid))

    def download_images(self, imgfilepath):
        """
        Download RASS images corresponding to the given group catalog.
        The number of download images is N*M, where N is the number of
        galaxy groups (i.e. len(grpra)) and M is len(surveys). If the
        provided path already contains <= N*M FITS files, the program
        will ask permission to re-download the images. This will over-
        write any existing data in imgfilepath.

        Parameters
        ------------------
        imgfilepath : str
            System file path where download images should be stored.

        Returns
        -------------------
        None
        """
        dirfiles = os.listdir(imgfilepath)
        nimagesneeded = len(self.surveys)*len(self.grpra)
        nimagesindir = np.sum([int('.fits' in fname) for fname in dirfiles])
        if nimagesindir==0:
            print('Provided directory is empty... downloading RASS images.')
            download_images_java(imgfilepath, self.grpra, self.grpdec, self.grpid, self.surveys, self.centralname)
        elif nimages>0 and nimagesindir<=nimagesneeded:
            print('The provided directory already contains some FITS images.')
            answ=input("Enter 'c' to re-download RASS images into the provided directory, or enter any other key to exit the program: ")
            if answ=='c':download_images_java(imgfilepath, self.grpra, self.grpdec, self.grpid, self.surveys, self.centralname)
            else:sys.exit()
        else:
            print('Error in download_images: please clear the directory and try re-downloading.')
            sys.exit()

    
    def sort_images(self, imagefiledir, snrmin=0.2, maxzero=0.98):
        """
        Sort raw RASS images into good and poor coverage based on a SNR
        and zero-pixel threshold. This function creates a new attribute
        of the stacker object called `rosat_xray_stacker.goodimage`,
        a boolean array that indicates whether the raw RASS image had good
        or poor coverage.

        Parameters
        ----------------------------
        rawpath : str
            Path location where raw images are stored.
        snrmin : scalar
            Minimum signal-to-noise ratio used to make cut. Default
            is 0.2 (anything below 0.2 marked 'poor'.
        maxzero : scalar
            Maximum fraction of zero pixels for a good image, default 0.98.
        outpath : str, default None
            Path where good images should be written. Default none.
       
        Returns 
        ----------------------------
        None. The good image flag attribute is modified in place.

        """
        # sort directory files with order of group catalog
        imagefiles = os.listdir(imagefiledir)
        imagenames = np.array(os.listdir(imagefiledir))
        assert len(self.grpid)==len(imagenames), "Number of files in directory must match number of groups."
        imageIDs = np.array([float(imgnm.split('_')[2][3:]) for imgnm in imagenames])
        _, order = np.where(self.grpid[:,None]==imageIDs)
        imageIDs = imageIDs[order]
        imagenames = imagenames[order]
        assert (imageIDs==self.grpid).all(), "ID numbers are not sorted properly."

        # make SNR and zero-pixel assessments of images
        snr = np.zeros(len(imagefiles))
        zcount = np.zeros(len(imagefiles))
        goodflag = np.zeros(len(imagefiles))
        for i,imgpath in enumerate(imagenames):
            if imgpath.endswith('.fits'):
                hdulist = fits.open(imagefiledir+imgpath,memap=False)
                image = hdulist[0].data.flatten()
                hdulist.close()
                snr[i] = np.mean(image)/np.std(image)
                zcount[i] = len(image[image==0])/len(image)
        sel = np.where(np.logical_and(snr>=snrmin, zcount<maxzero))
        goodflag[sel]=1
        self.goodflag = goodflag

    def measure_SNR_1Mpc(self,imagefiledir,snrthreshold=3):
        """
        Measure the signal-to-noise ratio within the central 1Mpc sky area of the 
        group image. This SNR measurement is used to determine whether the good
        image is marked as a detection or nondetection, and thus whether point
        source removal should be applied.

        Parameters
        -------------------
        imagepath : str
            Directory where raw images were stored. Only images with goodflag=1
            will have their SNR measured.
        snrthreshold : float, default 3
            Confidence threshold for which signals are considered detections. Default 3-sigma.
        
        Returns
        -------------------
        None. Modifies rosat_xray_stacker.detection in place and creates rosat_xray_stacker.centralSNR.
        """
        imagefiles = os.listdir(imagefiledir)
        imagenames = np.array(os.listdir(imagefiledir))
        assert len(self.grpid)==len(imagenames), "Number of files in directory must match number of groups."
        imageIDs = np.array([float(imgnm.split('_')[2][3:]) for imgnm in imagenames])
        _, order = np.where(self.grpid[:,None]==imageIDs)
        imageIDs = imageIDs[order]
        imagenames = imagenames[order]
        assert (imageIDs==self.grpid).all(), "ID numbers are not sorted properly."

        # make SNR measurement
        snr = np.zeros(len(imagenames))
        X,Y = np.meshgrid(np.arange(0,300,1), np.arange(0,300,1))
        for i,imgpath in enumerate(imagenames):
            if imgpath.endswith('.fits'):
                hdulist = fits.open(imagefiledir+imgpath,memap=False)
                image = hdulist[0].data
                hdulist.close() 
                radius = (1/(self.grpcz[i]/70.))*206265/45. # in px
                dist_from_center = np.sqrt((X-150.)**2. + (Y-150.)**2.)
                measuresel = np.where(np.logical_and(dist_from_center<radius, image>0))
                snr[i] = np.mean(image[measuresel])/np.std(image[measuresel])
        self.centralSNR = snr 
        self.detection = (snr>snrthreshold) & (self.goodflag==1) 

 
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
            hdulist.close()


    def scale_subtract_images(self, imagefiledir, outfiledir, progressConf=False):
        """
        Subtract >5*sigma pixels from images and scale images to 
        a common redshift.
        
        Parameters
        -------------
        imgfiledir : str
            Path to directory containing input images for scaling.
            Each FITS file in this directory must be named consistently
            with the rest of this program (e.g. RASS-Int_Broad_grp13_ECO03822.fits).
        outfiledir : str
            Path where scaled images should be written.

        Returns
        -------------
        Scaled/subtracted images are written to the specified path.
        """
        czmax = np.max(self.grpcz)
        imagenames = np.array(os.listdir(imagefiledir))
        imageIDs = np.array([float(imgnm.split('_')[2][3:]) for imgnm in imagenames])
        for k in range(0,len(imagenames)):
            hdulist = fits.open(imagefiledir+imagenames[k], memap=False)
            img = hdulist[0].data
            im2=img*1
            mean=np.mean(im2[np.where(im2!=0)])
            std = np.std(im2[np.where(im2!=0)])
            im2[im2 > mean+5*std]=mean
            im2[130:170,130:170]=img[130:170,130:170] # preserve inner portion
            img = np.copy(im2)
            czsf = self.grpcz[self.grpid==imageIDs[k]]/czmax
            img = ndimage.geometric_transform(img, scale_image, cval=0, extra_keywords={'scale':czsf})
            hdulist[0].data = img
            hdulist.writeto(outfiledir+imagenames[k], overwrite=True)
            hdulist.close()
            if progressConf: print("Image {} complete.".format(k))

    def stack_images(self, imagefiledir, outfiledir, stackproperty, binedges):
        return self.stack_images_func(self.grpid, self.grpcz, imagefiledir, outfiledir, stackproperty, binedges)

    @staticmethod
    def stack_images_func(grpid, grpcz, imagefiledir, outfiledir, stackproperty, binedges):
        """
        Stack X-ray images of galaxy groups in bins of group properties
        (e.g. richness or halo mass).

        Parameters
        --------------------
        imgfiledir : str
            Path to directory containing input images for stacking.
            Each FITS file in this directory must be named consistently
            with the rest of this program (e.g. RASS-Int_Broad_grp13_ECO03822.fits).
        outfiledir : str
            Path where stacked images should be written. The number of
            images written will depend on bins.
        stackproperty : iterable
            Group property to be used for binning (e.g. halo mass). This
            list should include an entry for *every* galaxy, as to match
            the length and order of self.grpid.
        bins : iterable
            Array of bins for stacking. It should represent the bin *edges*. 
            Example: if bins=[11,12,13,14,15,16], then the resulting bins
            are [11,12], [12,13], [13,14], [14,15], [15,16].
        
        Returns
        --------------------

        """
        imagenames = np.array(os.listdir(imagefiledir))
        assert len(grpid)==len(imagenames), "Number of files in directory must match number of groups."
        imageIDs = np.array([float(imgnm.split('_')[2][3:]) for imgnm in imagenames])
        _, order = np.where(grpid[:,None]==imageIDs)
        imageIDs = imageIDs[order]
        imagenames = imagenames[order]
        assert (imageIDs==grpid).all(), "ID numbers are not sorted properly."

        czmax = np.max(grpcz)
        stackproperty = np.asarray(stackproperty)
        binedges = np.array(binedges)
        leftedges = binedges[:-1]
        rightedges = binedges[1:]
        bincenters = (leftedges+rightedges)/2.
        finalimagelist = []
        n_in_bin=[]
        for i in range(0,len(bincenters)):
            stacksel = np.where(np.logical_and(stackproperty>=leftedges[i], stackproperty<rightedges[i]))
            imagenamesneeded = imagenames[stacksel]
            imageIDsneeded = imageIDs[stacksel]
            images_to_stack = []
            for j in range(0,len(imagenamesneeded)):
                img = imagenamesneeded[j]
                hdulist = fits.open(imagefiledir+img, memmap=False)
                img = hdulist[0].data
                hdulist.close()
                images_to_stack.append(img)
            avg, median, std = sigma_clipped_stats(np.array(images_to_stack), sigma=10., maxiters=1, axis=0)
            n_in_bin.append(len(images_to_stack))
            finalimagelist.append(avg)
            print("Bin {} done.".format(i))
        return n_in_bin, bincenters, finalimagelist

if __name__=='__main__':
    ecocsv = pd.read_csv("../g3groups/ECOdata_G3catalog_luminosity.csv")
    ecocsv = ecocsv[ecocsv.g3fc_l==1] # centrals only
    eco = rosat_xray_stacker(ecocsv.g3grp_l, ecocsv.g3grpradeg_l, ecocsv.g3grpdedeg_l, ecocsv.g3grpcz_l, centralname=ecocsv.name, surveys=['RASS-Int Hard'])
    eco.sort_images('./g3rassimages/eco/')
    #eco.measure_signal_1Mpc('./g3rassimages/eco/')
    #eco.download_images('./g3rassimages/eco/')
    #eco.mask_point_sources('/srv/two/zhutchen/rosat_xray_stacker/g3rassimages/eco/', 'whatever/', examine_result=True, smoothsigma=3, starfinder_threshold=5)
    #eco.mask_point_sources('/srv/scratch/zhutchen/eco03822files/', 'whatever/', examine_result=True, smoothsigma=3, starfinder_threshold=5)
    #eco.scale_subtract_images("./g3rassimages/eco/", "./g3rassimages/eco_scaled/", True)
    nbin, bincenters, images = eco.stack_images("./g3rassimages/eco_scaled/", "whatever", np.asarray(ecocsv.g3logmh_l), binedges=np.arange(12,16,1))
    Rvirs = ((3*10**bincenters) / (4*np.pi*337*0.3*1.36e11) )**(1/3)
    rvirscales = 0.5 * Rvirs/(7000/70.) * 206265 / 45. / 3.
    print(rvirscales)
    images = [gaussian_filter(images[i],2) for i in range(0,len(images))]
    maxes = np.asarray([np.max(im) for im in images])
    scaleto = np.mean(maxes)-0.5*np.std(maxes)
    czmax = np.max(ecocsv.g3grpcz_l)
    print(czmax)
    for index, image in enumerate(images):
        plt.figure()
        plt.imshow(image, extent=[-150,150,-150,150],vmax=scaleto,vmin=0)
        Rvir = ((3*10**bincenters[index]) / (4*np.pi*337*0.3*1.36e11) )**(1/3)
        print(Rvir)
        Rvirx, Rviry = get_circle(Rvir/(czmax/70.) * 206265 * (1/45))
        plt.plot(Rvirx, Rviry, color='orange', linewidth=2)
        plt.title(r'$<\log M_{\rm vir}>=$ '+'{:0.2f}'.format(bincenters[index])+' (N={})'.format(nbin[index]))
        plt.show()
