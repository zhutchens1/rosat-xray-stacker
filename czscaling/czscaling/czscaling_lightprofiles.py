import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot 
from astropy.io import fits
from scipy.ndimage import gaussian_filter, geometric_transform
import os
import matplotlib.pyplot as plt

def get_intensity_profile_physical(img, radii, grpdist, npix=300, centerx=150, centery=150):
    intensity = np.float64(np.zeros_like(radii[:-1]))
    intensity_err = np.float64(np.zeros_like(radii[:-1]))
    luminosity = np.zeros_like(radii[:-1])
    X,Y = np.meshgrid(np.arange(0,npix,1),np.arange(0,npix,1))
    dist_from_center = np.sqrt((X-centerx)*(X-centerx) + (Y-centery)*(Y-centery))
    for ii in range(0,len(radii)-1):
        measuresel = np.logical_and(dist_from_center>=radii[ii],dist_from_center<=radii[ii+1])
        flux = np.average(img[measuresel]) # cts s^-1
        radii_Mpc = radii*(45)/206265*grpdist # radians to Mpc
        area = np.pi*radii_Mpc[ii+1]*radii_Mpc[ii+1] - np.pi*radii_Mpc[ii]*radii_Mpc[ii] # Mpc^2
        intensity[ii] = flux/area # cts/s/Mpc^2
        intensity_err[ii] = np.std(img[measuresel])/area/np.sqrt(len(measuresel[0]))
    return radii_Mpc[:-1], intensity, intensity_err

def scale_image(output_coords,scale):
    return (output_coords[0]/scale+150-150/scale, output_coords[1]/scale+150-150/scale)

def get_chisq(imagefile, grpdist, Dratio):
    """
    Get the chi-squared value between surface brightness profiles
    for an original RASS image and one scaled to larger distance D.
    
    Parameters
    -------------------
    imagefile : str
        Path to image.
    Di : float
        Ratio between image distance and desired scaling distance (<1). 
    
    Returns
    -------------------
    chi2 : float
        Chi-squared value between original and scaled light profiles.
    """
    image = fits.open(imagefile)[0].data
    scaled = geometric_transform(image, scale_image, cval=0, extra_keywords={'scale':Dratio})

    fig,axs=plt.subplots()
    axs[0].imshow(image)
    axs[1].imshow(scaled)
    plt.show()

    px=np.arange(2,140,2)
    radii1,intensity1,_=get_intensity_profile_physical(image,px,grpdist) # rad, cts/s/sr
    radii2,intensity2,_=get_intensity_profile_physical(scaled,px,grpdist) # rad, cts/s/sr
    
    sel = np.where(intensity2>0)
    radii1=radii1[sel]    
    radii2=radii2[sel]
    intensity1=intensity1[sel] 
    intensity2=intensity2[sel] 

    #plt.figure()
    #plt.plot(radii1,intensity1, '.')
    #plt.plot(radii2,intensity2, '.')
    #plt.yscale('log')
    #plt.show()
    return np.sum((intensity2-intensity1)*(intensity2-intensity1)/intensity1) # chi-sq


if __name__=='__main__':
    ecocsv = pd.read_csv("../../g3groups/ECOdata_G3catalog_luminosity.csv")
    ecocsv = ecocsv[ecocsv.g3fc_l==1] # centrals only
    halomass=np.zeros_like(ecocsv.g3logmh_l)
    grpid=np.array(ecocsv.g3grp_l)
    chisq = np.zeros_like(grpid)
    czmax=7000.
    for ii,fnm in enumerate(os.listdir("../g3rassimages/broad/eco_broad_cts/")):
        grpid = int(fnm.split('_')[2][3:-5])
        df = ecocsv[ecocsv.g3grp_l==grpid]
        grpcz = float(df.g3grpcz_l)
        halomass[ii] = float(df.g3logmh_l)
        chisq[ii]=get_chisq("../g3rassimages/broad/eco_broad_cts/"+fnm, grpdist=grpcz/70., Dratio=grpcz/czmax)
    plt.figure()
    plt.scatter(halomass,chisq,s=2)
    plt.show()
