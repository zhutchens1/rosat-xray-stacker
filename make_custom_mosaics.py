import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from reproject.mosaicking import reproject_and_coadd, find_optimal_celestial_wcs
from reproject import reproject_interp, reproject_exact
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as uu
import matplotlib.pyplot as plt

def make_custom_mosaics(groupid, groupra, groupdec, count_paths, exp_paths, outsz, outdir, **rckwargs):
    # for each group id,
    #   get list of 9 nearest image names from image keys
    #   get filepaths to those counts and exposure maps (each as list)
    #   reproject and coadd into a mosaic
    #   extract final image of specified cut out at the group RA/Dec
    groupid=np.array(groupid)
    groupra=np.array(groupra)
    groupdec=np.array(groupdec)
    count_paths=np.array(count_paths,dtype=object)
    exp_paths=np.array(exp_paths,dtype=object)
    for ii,gg in enumerate(groupid):
        cname=count_paths[ii]
        ename=exp_paths[ii]
        print(cname.shape,ename.shape)
        cmosaic = reproject_and_coadd(cname,**rckwargs)
        emosaic = reproject_and_coadd(ename,**rckwargs)
        extract_write_from_mosaic(mosaic,outsz,outdir)
        exit()
    
def extract_write_from_mosaic(image,sz,path):
    # look at astropy cutout 2D as a way to do this.
    pass


def get_neighbor_images(groupra, groupdec, imagera, imagedec, imagename, kk=9):
    """
    Given a set of galaxies or galaxy groups, and a separate set of 
    images, find the k neighboring images surrounding each group (incl.
    the image that contains the group of interest.) This algoritm implements
    sklearn.neighbors.KDTree to find images nearby to groups.

    Parameters
    ------------------------
    groupra : iterable
        RA of group centers, size N, where N is the number
        of unique groups in the catalog. Decimal degrees.
    groupdec : iterable
        Declination of group centers, size N. Decimal degrees.
    imagera : iterable
        RA of image centers, size p. Decimal degrees.
    imagedec : iterable
        Declination of image centers, size p. Decimal degrees.
    imagename : iterable
        Array of identificiation values for each image. Can be strings,
        int, etc. Size p.
    kk : int, default 9
        Number of images to return for each group. This argument is
        passed as `k` to sklearn.neighbors.KDTree.
        
    Returns
    -------------------------
    neighbors : np.array of shape (N,kk)
        Matrix containing the kk-nearest images, denoted using values
        from `imagename`, for each observation in the group* dataset.
    """
    groupra=np.array(groupra)
    groupdec=np.array(groupdec)
    imagera=np.array(imagera)
    imagedec=np.array(imagedec)
    dtr=np.pi/180.
    imageX = np.sin(np.pi/2.-imagedec*dtr)*np.cos(imagera*dtr) 
    imageY = np.sin(np.pi/2.-imagedec*dtr)*np.sin(imagera*dtr)
    imageZ = np.cos(np.pi/2.-imagedec*dtr)
    groupX = np.sin(np.pi/2.-groupdec*dtr)*np.cos(groupra*dtr) 
    groupY = np.sin(np.pi/2.-groupdec*dtr)*np.sin(groupra*dtr)
    groupZ = np.cos(np.pi/2.-groupdec*dtr)
    imagename=np.array(imagename,dtype=object)
    XX = np.array([imageX,imageY,imageZ])
    tree = KDTree(np.array(XX.T))
    groupdata = np.array([groupX,groupY,groupZ])
    idx = tree.query(groupdata.T, kk, return_distance=False)
    neighbors = imagename[idx]
    return neighbors


if __name__=='__main__':
    eco = pd.read_csv("../g3groups/ECOdata_G3catalog_luminosity.csv")
    eco = eco[eco.g3fc_l==1]
    
    rasstable = pd.read_csv("RASS_public_contents_lookup.csv")
    econame=np.array(eco.name,dtype=object)
    names=get_neighbor_images(eco.g3grpradeg_l, eco.g3grpdedeg_l, rasstable.ra, rasstable.dec, rasstable.image, 5)

    radeg,dedeg = np.array(eco.g3grpradeg_l), np.array(eco.g3grpdedeg_l)
    exposuremaps=np.zeros_like(names,dtype='object')
    countmaps=np.zeros_like(names,dtype='object')
    for ii,subarr in enumerate(names):
        for jj,name in enumerate(subarr):
            obs=name.split('.')[0]
            countmaps[ii][jj]='../rass/'+obs+'/'+obs+'_im1.fits'
            exposuremaps[ii][jj]='../rass/'+obs+'/'+obs+'_mex.fits'
    
    #sw=sw.Swarp()
    #swarp.path_swarp = '.'
    #swarp.swarp_configuration_file='config.swarp'
    #swarp.list_images = countmaps[0]
    #swarp.filename_final='testmosaic.fits'
    #swarp.mosaic_images()
    #oproj = WCS(fits.open(countmaps[0][0])[1].header)
    #make_custom_mosaics(eco.g3grp_l, eco.g3grpradeg_l, eco.g3grpdedeg_l, countmaps, exposuremaps, (512,512), 'whatev',\
    #    output_projection=fits.open(countmaps[0][0])[1].header, reproject_function=reproject_interp, hdu_in=1)
    print(countmaps[0])
    print(radeg[0],dedeg[0])
    hdulist=[fits.open(cmap)[0] for cmap in countmaps[0]]
    import time
    tt=time.time()
    wcs_out,shape_out = find_optimal_celestial_wcs(hdulist)
    array, footprint = reproject_and_coadd(hdulist, output_projection=wcs_out, reproject_function=reproject_exact, shape_out=shape_out)
    print('elapsed time: ', time.time()-tt)
    hdu=fits.PrimaryHDU(array,header=wcs_out.to_header())
    hdulist=fits.HDUList([hdu])
    hdulist.writeto("countmap0mosaic.fits")
   
    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(array, origin='lower', vmin=600, vmax=800)
    ax1.set_title('Mosaic')
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(footprint, origin='lower')
    ax2.set_title('Footprint') 
    plt.show()

    
    print(type(array))
    image = Cutout2D(array,position=SkyCoord(ra=radeg[0]*uu.degree,dec=dedeg[0]*uu.degree),wcs=wcs_out,size=512)
    print(image) 
    plt.figure()
    plt.imshow(image.data)
    plt.title("Final 512x512 Cutout")
    plt.show()
    
    hdu = fits.PrimaryHDU(image.data, header=image.wcs.to_header())
    hdulist=fits.HDUList([hdu])
    hdulist.writeto("countmap0.fits")
