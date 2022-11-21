import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import anderson_ksamp as adtest, ks_2samp as kstest
from astropy.io import fits
from os import listdir
import pandas as pd
from astroML.plotting import scatter_contour
from center_binned_stats import center_binned_stats
from scipy.ndimage import geometric_transform

def scale_image(output_coords,scale,imwidth):
    mid = imwidth//2
    return (output_coords[0]/scale+mid-mid/scale, output_coords[1]/scale+mid-mid/scale)

def compute_sums():
    direc1='../g3rassimages/broad/eco_broad_cts_ptsrc_rm/'
    direc2='../g3rassimages/broad/eco_broad_cts_scaled/'
    df = pd.read_csv("../../g3groups/ECOdata_G3catalog_luminosity.csv")
    df = df[df.g3fc_l==1.]
    files = listdir(direc1)
    origsum=[]
    scalsum=[]
    name=[]
    scales=[]
    for ff in files:
        print(ff)
        grpid = ff.split('_')[2][3:-5]
        grpid = (float(grpid))
        sfactor = float(df.g3grpcz_l[df.g3grp_l==grpid])/7470.
        #print(sfactor)

        original = fits.open(direc1+ff)[0].data
        rescaled = geometric_transform(original, scale_image, extra_arguments=(sfactor,original.shape[0]))
        original = sfactor*sfactor*original.flatten()
        rescaled = rescaled.flatten()
    
        name.append(ff)
        origsum.append(np.sum(original))
        scalsum.append(np.sum(rescaled))
        scales.append(sfactor)
    pd.DataFrame(np.array([name,origsum,scalsum,scales]).T, columns=['filename','origsum','scalsum','sfactor']).to_csv("pixelsums.csv",index=False)

if __name__=='__main__':
    #compute_sums()
    df = pd.read_csv("pixelsums.csv")
    plt.figure()
    plt.title("Max Reconstruction Error: "+"{:0.2f}".format(np.max((df.scalsum-df.origsum).abs()/df.origsum)*100.)+"%")
    plt.scatter(df.origsum,(df.scalsum-df.origsum).abs()/df.origsum, color='k',s=2,alpha=0.3)
    median, binc, _, _ = center_binned_stats(df.origsum,(df.scalsum-df.origsum).abs()/df.origsum,'median',bins=10)
    plt.plot(binc,median,color='orange',label="Median")
    plt.xlabel("Sum of Pixel Counts (Original)")
    plt.ylabel("Reconstruction Error")
    plt.legend(loc='best')
    plt.yscale('log')
    plt.show()

    plt.clf()
    plt.title("Max Reconstruction Error: "+"{:0.2f}".format(np.max((df.scalsum-df.origsum).abs()/df.origsum)*100.)+"%")
    plt.scatter(df.sfactor,(df.scalsum-df.origsum).abs()/df.origsum,color='k',s=2,alpha=0.5)
    median,binc,_,_ = center_binned_stats(df.sfactor,(df.scalsum-df.origsum).abs()/df.origsum,'median',bins=10)
    plt.plot(binc,median,color='orange',label='Median')
    plt.xlabel(r"$cz_{\rm original}\, / \,cz_{\rm scale}$")
    plt.ylabel("Reconstruction Error")
    plt.yscale('log')
    plt.legend(loc='best')
    plt.show()
