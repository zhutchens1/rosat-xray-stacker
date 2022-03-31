import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from reproject.mosaicking import reproject_and_coadd

def make_custom_mosaics(groupid, groupra, groupdec, count_paths, exp_paths, outsz, outdir, **rckwargs):
    # for each group id,
    #   get list of 9 nearest image names from image keys
    #   get filepaths to those counts and exposure maps (each as list)
    #   reproject and coadd into a mosaic
    #   extract final image of specified cut out at the group RA/Dec
    groupid=np.array(groupid)
    groupra=np.array(groupra)
    groupdec=np.array(groupdec)
    imagepaths=np.array(imagepaths,dtype=object)
    for ii,gg in enumerate(groupid):
        cname=count_paths[ii]
        ename=exp_paths[ii]
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
    names=get_neighbor_images(eco.g3grpradeg_l, eco.g3grpdedeg_l, rasstable.ra, rasstable.dec, rasstable.image, 9)

    exposuremaps=np.zeros_like(names,dtype='object')
    countmaps=np.zeros_like(names,dtype='object')
    for ii,subarr in enumerate(names):
        for jj,name in enumerate(subarr):
            obs=name.split('.')[0]
            countmaps[ii][jj]='../rass/'+obs+'/'+obs+'_im1.fits.Z'
            exposuremaps[ii][jj]='../rass/'+obs+'/'+obs+'_mex.fits.Z'
    
