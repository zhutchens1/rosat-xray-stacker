import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from reproject import reproject_and_coadd

def make_custom_mosaics(groupid, groupra, groupdec, imagepaths, outsz, outdir, **rckwargs):
    # for each group id,
    #   get list of 9 nearest image names from image keys
    #   get filepaths to those counts and exposure maps (each as list)
    #   reproject and coadd into a mosaic
    #   extract final image of specified cut out at the grpra/grpdec
    groupid=np.array(groupid)
    groupra=np.array(groupra)
    groupdec=np.array(groupdec)
    imagepaths=np.array(imagepaths)
    for ii,gg in enumerate(groupid):
        fpaths=imagepaths[ii]
        mosaic = reproject_and_coadd(fpaths,**rckwargs)
        extract_write_from_mosaic(mosaic,outsz,outdir)
    
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
    sel = np.where(econame=='ECO03822')
    print(eco[['radeg','dedeg']][eco.name=='ECO03822'])
    print(names[sel])
    
