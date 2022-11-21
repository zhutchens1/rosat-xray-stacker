import numpy as np
from  matplotlib import pyplot as plt
from  astropy.io import fits
from scipy import ndimage

def scale_image(output_coords,scale):
    return (output_coords[0]/scale+150-150/scale, output_coords[1]/scale+150-150/scale)

def get_intensity_profile(img, radii, npix=300, centerx=150, centery=150, cz=None):
    intensity = np.zeros_like(radii[:-1])
    intensity_err = np.zeros_like(radii[:-1])
    luminosity = np.zeros_like(radii[:-1])
    X,Y = np.meshgrid(np.arange(0,npix,1),np.arange(0,npix,1))
    dist_from_center = np.sqrt((X-centerx)*(X-centerx) + (Y-centery)*(Y-centery))
    for ii in range(0,len(radii)-1):
        measuresel = np.logical_and(dist_from_center>=radii[ii],dist_from_center<=radii[ii+1])
        flux = np.average(img[measuresel]) # cts cm^-2 s^-1
        if cz is not None:
            area = (45*(1/206265)*(cz/70.))**2. * (np.pi*radii[ii+1]*radii[ii+1] - np.pi*radii[ii]*radii[ii])
        else:
            area = np.pi*radii[ii+1]*radii[ii+1] - np.pi*radii[ii]*radii[ii] # px
        intensity[ii] = flux/area # cts/cm2/s/px
        intensity_err[ii] = np.std(img[measuresel])/area/np.sqrt(len(measuresel[0]))
        Distance = 100 # Mpc
        luminosity[ii] = 4*np.pi*intensity[ii]*(radii[ii]*(45./1)*(1/206265)*Distance*3.086e24)**2.
        # units cts/s
    return radii[:-1], intensity, intensity_err, luminosity

if __name__=="__main__":
    image=fits.open("eco03822_cnthard.fits")[0].data
    print('Original: ',np.sum(image))
    r_value = np.linspace(1,130,100) 
    radii0, intensity0, ierr0, _ = get_intensity_profile(image, r_value, cz=None)
    #radii0 = radii0*45*(1/206265)*(6558/70.) # Mpc
    for max_z in (7000,10000,14000,7/3*6558):
        czsf=6558/max_z
        #r_value = np.linspace(1,130,int(100/czsf)+1) 
        scaled=ndimage.geometric_transform(image, scale_image, cval=0, extra_keywords={'scale':czsf})
        print('Scaled: ', np.sum(scaled))
        radii, intensity, ierr, _ = get_intensity_profile(scaled, r_value, cz=None)
        #radii = radii*45*(1/206265)*(max_z/70.) # Mpc
        plt.figure()
        plt.errorbar(radii0,intensity0,yerr=ierr0,label='Original Image')
        plt.errorbar(radii,intensity, yerr=ierr,label='Scaled Image')
        #plt.xlim(0,2)
        plt.title(r"$d_{\rm scale}/d_{\rm grp}$ = "+"{:0.2f}".format(1/czsf))
        plt.xlabel("Radius [px]")
        plt.ylabel(r"Surface Brightness [cts $\rm s^{-1}$ $\rm px^{-1}$]")
        plt.legend(loc='best')
        plt.yscale('log')
        plt.show()
