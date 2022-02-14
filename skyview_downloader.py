from astroquery.skyview import SkyView as sv
from astropy.io import fits
import os

def download_images_astroquery(path, grpra, grpdec, grpid, surveys, centralname=''):
    """
    Extract group images from the ROSAT All-Sky Survey and save to local disk.
    NOTE (5/21/2021): Astroquery/SkyView Query is unable to access all RASS data files,
    specifically the intensity maps for soft/hard/broad. We recommend using the java
    downloader (our wrapper `download_images_java` instead).

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
    if isinstance(centralname,str):
        centralname = [centralname for i in range(0,len(grpra))]
    for i in range(0,len(grpra)):
        location = "{a:0.5f}, {b:0.5f}".format(a=grpra[i], b=grpdec[i])
        images = sv.get_images(position=location, survey=surveys)
        for j, img in enumerate(images):
            savename = surveys_for_save[j]+"_grp{}".format(grpid[i])+"_"+centralname[i]+".fits" 
            img.writeto(path+savename, overwrite=True)
 


def download_images_java(path, grpra, grpdec, grpid, surveys, centralname=''):
    """
    Extract group images from the ROSAT All-Sky Survey and save to local disk
    using the SkyView java tool.

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
    if isinstance(centralname,str):
        centralname = [centralname for i in range(0,len(grpra))]

    # check to see if skyview.jar exists; otherwise wget it
    if os.path.exists('./skyview.jar'):
        print('File skyview.jar successfully loaded.')
    else:
        print("SkyView JAR file not found... downloading file from http://skyview.gsfc.nasa.gov/jar/skyview.jar")
        try: 
            os.system("wget http://skyview.gsfc.nasa.gov/jar/skyview.jar")
        except OSError:
            print("Could not successfully download Skyview jar file. Please download manually or check file permissions.")
            sys.exit()
    # begin downloading images
    for i in range(0,len(grpra)):
        for j in range(0,len(surveys)):
            downloadcmd='java -jar skyview.jar survey="'+surveys[j]+'" position="'+"{a:0.5f}, {b:0.5f}".format(a=grpra[i], b=grpdec[i])+'"'
            savename = path+surveys_for_save[j]+"_grp{}".format(grpid[i])+"_"+centralname[i]+".fits"
            if os.path.exists(savename):
                print('Skipping '+savename+' ; file already exists')
            else:
                downloadcmd+=' output='+savename
                os.system(downloadcmd)





if __name__=='__main__':
    data = sv.get_images(position='194.898, 27.9594', survey=['RASS-Int Broad'])

    data[0].writeto('coma_skyview_broad.fits')
