"""
Stack diffuse X-ray emission in galaxy groups from ROSAT All-Sky Survey Data.
Author: Zack Hutchens (G3 subteam with Kelley Hess + Andrew Baker)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.skyview import SkyView as sv
import os
import sys

from grab_rass_images import download_images


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
        download_images(imgfilepath, grpra, grpdec, grpid, surveys, centralname)
    elif nimagesindir<nimagesneeded:
        print('Provided directory has too few files...')
        answ=input("Enter 'c' to re-download RASS images into provided directory (may overwrite existing files): ")
        if answ=='c':download_images(imgfilepath, grpra, grpdec, grpid, surveys, centralname)
        else:sys.exit()
    else:
        print('Provided directory has sufficient *.fits files -- proceeding without downloading RASS images')
     
    # (2) Image Inspection/Classification 
    # need to inspect images? yes, or say file where classifications are stored

    # (3) Image Stacking 

    pass


if __name__=='__main__':
    g3groups = pd.read_csv("../g3groups/ECO_G3groupcatalog_030821.csv")
    g3groups = g3groups[(g3groups.g3fc_l==1)] # select only centrals since we want group info.
    
    rosat_xray_stacker('./g3rassimages/', g3groups.g3grpradeg_l, g3groups.g3grpdedeg_l, g3groups.g3grp_l, \
                       surveys=['RASS Background 1', 'RASS-Cnt Soft'], \
                       centralname=g3groups.name)

