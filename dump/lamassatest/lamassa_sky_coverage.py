from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def drawrectangle(axesObject, racen, deccen, area):
    length = np.sqrt(area)
    x1 = np.linspace(racen-length/2, racen+length/2, 10)
    x2 = np.zeros(10)+(racen+length/2)
    x3 = x1
    x4 = np.zeros(10)+(racen-length/2)

    y1 = np.zeros(10)+(deccen - length/2)
    y2 = np.linspace(deccen-length/2, deccen+length/2, 10)
    y3 = np.zeros(10)+(deccen + length/2)
    y4 = y2
    x = np.concatenate((x1,x2,x3,x4))
    y = np.concatenate((y1,y2,y3,y4))

hdu1 = fits.open("XMM_archive_ao10_multiwavelength.fits")
hdu2 = fits.open("XMM_multiwavelength_cat_ao13.fits")
hdu3 = fits.open("Chandra_multiwavelength_new_spectra.fits")
resolve = pd.read_csv("../g3groups/RESOLVEdata_G3catalog_luminosity.csv")
resolve = resolve[(resolve.cz>4000) & (resolve.cz<7000) & (resolve.fl_insample==1) & (resolve.f_b==1)]

ao10_ra = np.array(hdu1[1].data.field(2))
ao10_dec = np.array(hdu1[1].data.field(3))

xmm_ra = np.array(hdu2[1].data.field(2))
xmm_dec = np.array(hdu2[1].data.field(3))

chandra_ra = np.array(hdu3[1].data.field(2))
chandra_dec = np.array(hdu3[1].data.field(3))

resolvera = np.array(resolve.radeg)
resolvedec = np.array(resolve.dedeg)

sel1 = lambda x: np.where(np.logical_and(x>-1, x<70))
sel2 = lambda x: np.where(np.logical_and(x>305,x<365))

# -1,70 - 305,361
fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(20,5), sharey=True)
ax2.plot(resolvera[sel1(resolvera)], resolvedec[sel1(resolvera)], 'r.')
ax2.plot(ao10_ra[sel1(ao10_ra)], ao10_dec[sel1(ao10_ra)], 'kx', alpha=0.1)
ax2.plot(xmm_ra[sel1(xmm_ra)], xmm_dec[sel1(xmm_ra)], 'kx', alpha=0.1)
ax2.plot(chandra_ra[sel1(chandra_ra)], chandra_dec[sel1(chandra_ra)], 'kx', alpha=0.1)

ax.plot(resolvera[sel2(resolvera)], resolvedec[sel2(resolvera)], 'r.', label='RESOLVE Galaxies')
ax.plot(ao10_ra[sel2(ao10_ra)], ao10_dec[sel2(ao10_ra)], 'kx', alpha=0.1, label='LaMassa et al. Point Source Catalog')
ax.plot(xmm_ra[sel2(xmm_ra)], xmm_dec[sel2(xmm_ra)], 'kx', alpha=0.1)
ax.plot(chandra_ra[sel2(chandra_ra)], chandra_dec[sel2(chandra_ra)], 'kx', alpha=0.1)



ax.set_ylabel("Declination [deg]")
ax.set_xlabel("RA [deg]")
ax2.set_xlabel("RA [deg]")
ax.legend(loc='upper left', framealpha=1)
plt.show()
