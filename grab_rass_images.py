from astroquery.skyview import SkyView as sv
from astropy.io import fits

def grab_images():
    """
    Extract group images from the ROSAT All-Sky Survey and save to local disk.

    Parameters
    ------------------

    Returns
    ------------------
    """
    pass


if __name__=='__main__':
    # try for RESOLVE grp 836 (N=40 w/ central rf0673)
    data = sv.get_images(position='181.1126055, 1.8961243', survey=['RASS Background 1', 'RASS-Cnt Soft'])
    #data[0].writeto('rf0673_rassbg1.fits')
    data[1].writeto('rf0673_rasscntsoft.fits')
