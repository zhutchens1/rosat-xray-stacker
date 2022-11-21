from astropy.io import fits
from astropy.coordinates import SkyCoord
from pyvo.dal import imagesearch
pos = SkyCoord.from_name('M17')
table = imagesearch('https://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia?type=at&ds=asky&',
                   pos, size=0.25).to_table()
table = table[(table['band'].astype('S') == 'K') & (table['format'].astype('S') == 'image/fits')]
m17_hdus =  [fits.open(url)[0] for url in table['download'].astype('S')]

from reproject.mosaicking import find_optimal_celestial_wcs
wcs_out, shape_out = find_optimal_celestial_wcs(m17_hdus)
print(m17_hdus)
print(wcs_out)
