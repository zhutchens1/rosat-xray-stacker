from skyview_downloader import download_images_java
import numpy as np

if __name__=="__main__":
    #####
    pc = np.genfromtxt("/srv/scratch/zhutchen/poor_coverage_berlinddr7_mr18.dat")
    ra = np.float64(pc[:,1])
    dec = np.float64(pc[:,2])
    grpid = np.float64(pc[:,0])
    download_images_java('/srv/scratch/zhutchen/khess_images/poor_coverage/', ra, dec, grpid, ['RASS-Int Hard'], centralname='')

    #####
    ndets = np.genfromtxt("/srv/scratch/zhutchen/non_detections_berlinddr7_mr18.dat")
    ra = np.float64(ndets[:,1])
    dec = np.float64(ndets[:,2])
    grpid = np.float64(ndets[:,0])
    download_images_java('/srv/scratch/zhutchen/khess_images/nondetections/', ra, dec, grpid, ['RASS-Int Hard'], centralname='')

    #####
    dets = np.genfromtxt("/srv/scratch/zhutchen/detections_berlinddr7_mr18.dat")
    ra = np.float64(dets[:,1])
    dec = np.float64(dets[:,2])
    grpid = np.float64(dets[:,0])
    download_images_java('/srv/scratch/zhutchen/khess_images/detections/', ra, dec, grpid, ['RASS-Int Hard'], centralname='')
    
