from astroquery.skyview import SkyView as sv
from astropy.io import fits

def download_images(grpra, grpdec, grpid, surveys, centralname=''):
    """
    Extract group images from the ROSAT All-Sky Survey and save to local disk.

    Parameters
    ------------------
    path : str
        Path to folder where downloaded images should be saved.
    grpra : iterable
        Right-acsencion of group center in degrees
    grpdec : iterable
        Declination of group center in degrees
    grpid : iterable
        Unique group ID number corresponding to each RA/Dec pair.
    surveys : str or iterable
        Surveys from which to download images (e.g. ['RASS Background 1', 'RASS-Cnt Soft']; see astroquery.skyview docs)
    centralname : str or iterable
        Name of central galaxy for each group. If included, this will be saved in the download image filenames.

    Returns
    ------------------
    None. All images are downloaded and stored in `path`.    
    """
    surveys_for_save = [srvy.replace(" ", "_") for srvy in surveys]
    for i in range(0,len(grpra)):
        location = "{a:0.5f}, {b:0.5f}".format(a=grpra[i], b=grpdec[i])
        images = sv.get_images(position=location, survey=surveys)
        for j, img in enumerate(images):
            savename = surveys_for_save[j]+"_grp{:d}_".format(grpid[i])+"_"+centralname+".fits" 
            print(savename)
            #img.write_to(savename)
    

if __name__=='__main__':
    # try for RESOLVE grp 836 (N=40 w/ central rf0673)
    data = sv.get_images(position='194.898, 27.9594', survey=['RASS Background 1', 'RASS-Cnt Soft'])
    data[0].writeto('rf0673_rassbg1.fits')
    data[1].writeto('eco03822_rasscntsoft.fits')
