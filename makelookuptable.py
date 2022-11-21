import os
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import pandas as pd

if __name__=='__main__':
    ra = []
    dec = []
    fnames=[]
    for i,e in enumerate(os.listdir('public_contents/')):
        if e.endswith('.public_contents'):
            f=open('public_contents/'+e, 'r')
            RA_set, DEC_set=0,0
            for line in f.readlines():
                if ("RIGHT_ASCENSION" in line) and (not RA_set):
                    RA=str(line[19:-2])
                    ra.append(RA)
                    RA_set=1
                    fnames.append(e)
                if ("DECLINATION" in line) and (not DEC_set):
                    DEC=str(line[15:-2])
                    dec.append(DEC)
                    print('------------------')
                    DEC_set=1
            print('appended coords for ', f)
            f.close()

    fnames=np.array(fnames)
    imagecenters = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    imagera=imagecenters.ra.value
    imagede=imagecenters.dec.value
    coords=np.array([imagera,imagede]).T
    table=pd.DataFrame(coords, columns=['ra','dec'])
    table['image']=fnames
    print(table)
    print(table[table.image=='rs931232n00.public_contents'])
    table.to_csv("RASS_public_contents_lookup.csv",index=False)

    coma = SkyCoord(ra=194.898*u.degree, dec=27.9594*u.degree)
    sep=coma.separation(imagecenters)
    index = np.argmin(sep)
    print(sep[index], fnames[index])
