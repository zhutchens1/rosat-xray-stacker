from skyview_downloader import download_images_java
import numpy as np
import random

if __name__=="__main__":
    #####
    dets = np.genfromtxt("/srv/scratch/zhutchen/poor_coverage_berlinddr7_mr18.dat")
    ra = np.float64(dets[:,1])
    dec = np.float64(dets[:,2])
    grpid = np.float64(dets[:,0])

    # need to get random offsets of total length 1.25
    rng = np.random.default_rng()
    radius = 1.25 # deg
    xoffsets = radius*rng.random(len(ra))
    yoffsets = np.sqrt(radius**2. - xoffsets**2.)
    print(np.sqrt(xoffsets**2. + yoffsets**2.))

    # add in offsets
    ra1 = ra+xoffsets
    dec1 = dec+yoffsets
   
    ra2 = ra-xoffsets
    dec2 = dec-yoffsets

    download_images_java('/srv/scratch/zhutchen/khess_images/poor_coverage_new/', ra1, dec1, grpid, ['RASS-Int Hard'], centralname='offset1')
    download_images_java('/srv/scratch/zhutchen/khess_images/poor_coverage_new/', ra2, dec2, grpid, ['RASS-Int Hard'], centralname='offset2')
    
