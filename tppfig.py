import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import pickle
import numpy as np
from rosat_xray_stacker import rosat_xray_stacker, measure_optimal_snr, get_intensity_profile_physical
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from seaborn import kdeplot
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['font.family'] = 'sans-serif'
#rcParams['font.sans-serif'] = ['Helvetica']
#rcParams['text.usetex'] = True
rcParams['grid.color'] = 'k'
rcParams['grid.linewidth'] = 0.2
my_locator = MaxNLocator(6)
singlecolsize = (3.3522420091324205, 2.0717995001590714)
doublecolsize = (7.500005949910059, 4.3880449973709)

"""
This file performs all the neccessary prepration for 
ECO RASS images prior to stacking: downloading images,
sorting them into sorting them into good vs. poor coverage,
and scaling them to the proper distance.  
"""
if __name__=='__main__':
    do_preparation = False
    scale_images = False
    stack_images = False
    analyze = True

    if do_preparation:
        ecocsv = pd.read_csv("../g3groups/ECOdata_G3catalog_luminosity.csv")
        ecocsv = ecocsv[ecocsv.g3fc_l==1] # centrals only

        eco = rosat_xray_stacker(ecocsv.g3grp_l, ecocsv.g3grpradeg_l, ecocsv.g3grpdedeg_l, ecocsv.g3grpcz_l, centralname=ecocsv.name, \
            surveys=['RASS-Int Hard'])
        #if not images_downloaded: eco.download_images('./g3rassimages/eco_hard/')
        #eco.sort_images('./g3rassimages/eco_hard_cts/')
        #eco.measure_SNR_1Mpc('./g3rassimages/eco_hard_cts/', 5)
        eco.mask_point_sources('./g3rassimages/hard/eco_hard_cts/', './g3rassimages/hard/eco_hard_cts_ptsrc_rm/',\
                 examine_result=False, smoothsigma=3, starfinder_threshold=5)

    if scale_images:
        ecocsv = pd.read_csv("../g3groups/ECOdata_G3catalog_luminosity.csv")
        ecocsv = ecocsv[ecocsv.g3fc_l==1] # centrals only
        ncores = 20
        eco = rosat_xray_stacker(ecocsv.g3grp_l, ecocsv.g3grpradeg_l, ecocsv.g3grpdedeg_l, ecocsv.g3grpcz_l, centralname=ecocsv.name, \
            surveys=['RASS-Int Hard'])
        subframes = np.array_split(ecocsv,ncores)
        subobjects = [rosat_xray_stacker(tdf.g3grp_l, tdf.g3grpradeg_l, tdf.g3grpdedeg_l, tdf.g3grpcz_l, centralname=tdf.name,\
            surveys=['RASS-Int Hard']) for tdf in subframes]

        import multiprocessing
        #processes=[None]*ncores
        #for jj in range(0,ncores):
        #    processes[jj]=multiprocessing.Process(target=subobjects[jj].scale_subtract_images,\
        #        args=("./g3rassimages/hard/eco_hard_exp/", "./g3rassimages/hard/eco_hard_exp_scaled/",True,512,45,70.,2530,7470,True))
        #for jj in range(0,ncores):
        #    processes[jj].start()
        #for jj in range(0,ncores):
        #    processes[jj].join()

        #for jj in range(0,ncores):
        #    processes[jj].start()
        #for jj in range(0,ncores):
        #    processes[jj].join()

        processes=[None]*ncores
        for jj in range(0,ncores):
            processes[jj]=multiprocessing.Process(target=subobjects[jj].scale_subtract_images,\
                args=("./g3rassimages/hard/eco_hard_cts_ptsrc_rm/", "./g3rassimages/hard/eco_hard_cts_scaled/",True,512,45,70.,2530.,7470.,True))
        for jj in range(0,ncores):
            processes[jj].start()
        for jj in range(0,ncores):
            processes[jj].join()
        #eco.scale_subtract_images("./g3rassimages/eco_hard_cts/", "./g3rassimages/eco_hard_cts_scaled/", crop=True, progressConf=True, imwidth=512)
        #eco.scale_subtract_images("./g3rassimages/eco_hard_exp/", "./g3rassimages/eco_hard_exp_scaled/", crop=True, progressConf=True, imwidth=512)
        eco.to_pickle("eco_hard_prepared_images_110121.pkl")
    
    if stack_images:
        ecocsv = pd.read_csv("../g3groups/ECOdata_G3catalog_luminosity.csv")
        ecocsv = ecocsv[ecocsv.g3fc_l==1] # centrals only
        #eco = pickle.load(open("eco_prepared_images_110121.pkl",'rb'))
        eco = rosat_xray_stacker(ecocsv.g3grp_l, ecocsv.g3grpradeg_l, ecocsv.g3grpdedeg_l, ecocsv.g3grpcz_l, centralname=ecocsv.name, \
            surveys=['RASS-Int Hard'])

        stackb = [11,12.1,13.3,15]
        stackID,nbin,bincenters,counts=eco.stack_images("./g3rassimages/hard/eco_hard_cts_scaled/",np.asarray(ecocsv.g3logmh_l), binedges=stackb)
        stackID,nbin,bincenters,times=eco.stack_images("./g3rassimages/hard/eco_hard_exp_scaled/",np.asarray(ecocsv.g3logmh_l), binedges=stackb)
        stacks = [stackID,nbin,bincenters,counts,times]
        pickle.dump(stacks, open('ecostackedimages_hard_032722.pkl','wb'))

    if analyze:
        import matplotlib.pyplot as plt
        binedges=[11,12.1,13.3,15]
        stackID,nbin,bincenters,counts,times = pickle.load(open('ecostackedimages_hard_032722.pkl','rb'))
        images = [ct/ex for (ct,ex) in zip(counts,times)]
        visimages = [gaussian_filter(images[i],2) for i in range(0,len(images))]
        Rvir = ((3*10**bincenters) / (4*np.pi*337*0.3*1.36e11) )**(1/3)
        Rvir_px = Rvir/100 * 206265 / 45.
        snrs = [measure_optimal_snr(images[ii],times[ii],7000,Rvir[ii])[0] for ii in range(0,len(images))]
        apfrac = [measure_optimal_snr(images[ii],times[ii],7000,Rvir[ii])[1] for ii in range(0,len(images))]
        maxes = np.asarray([np.max(im) for im in visimages])
        scaleto = np.mean(maxes)-0.5*np.std(maxes)
        azradiipx = np.arange(2,130,2)
        ecocsv=pd.read_csv("../g3groups/ECOdata_G3catalog_luminosity.csv")
        nums_from_aas=[0.894, 0.101, 0.055] 
        fig, axs = plt.subplots(ncols=len(images), figsize=(doublecolsize[0],0.7*doublecolsize[1]))
        for jj in range(0,len(images)):
            #axs[jj].set_title(r"$\left<\log M_{337}\right> = $ "+str(bincenters[jj]))
            axs[jj].set_title("{:0.1f} < ".format(binedges[jj])+r"$\log M_{\rm halo}$ < " + "{:0.1f}".format(binedges[jj+1]))
            shw=axs[jj].imshow(visimages[jj], norm=LogNorm())
            #clb=fig.colorbar(shw,orientation='horizontal',ax=axs[jj],pad=0.1)#,ticks=[np.min(visimages[jj]), (np.min(visimages[jj])+np.max(visimages[jj]))/2., np.max(visimages[jj])])
            #clb.ax.set_xticklabels([np.min(visimages[jj]), (np.min(visimages[jj])+np.max(visimages[jj]))/2., np.max(visimages[jj])])
            #clb.ax.locator_params(nbins=4)
            #clb.set_ticks(np.linspace(np.min(visimages[jj]),np.max(visimages[jj]),3))
            #clb.ax.set_title('counts / sec', fontsize=10)
            axs[jj].annotate(r"{} Images".format(nbin[jj]), xy=(1,5), backgroundcolor='white',fontsize=8)
            axs[jj].annotate(r"$S/N = $"+"{:.3f}".format(snrs[jj]), xy=(1,12), backgroundcolor='white',fontsize=8)

            circ=plt.Circle((visimages[jj].shape[0]//2, visimages[jj].shape[1]//2), 0.5*Rvir_px[jj],\
                   edgecolor='red', facecolor='None')
            axs[jj].add_patch(circ)
            axs[jj].set_xticks(np.arange(0,60,10))
            axs[jj].set_yticks(np.arange(0,60,10))
            #sel = (ecocsv.g3logmh_l>binedges[jj]) & (ecocsv.g3logmh_l<binedges[jj+1])
            #mediangrpgs = np.median((10**ecocsv[sel].g3grplogG_l/1.4/10**ecocsv[sel].g3grplogS_l))
            axs[jj].annotate(r"$\left<{M_{\rm HI,\, grp}}/{M_{\rm star,\, grp}}\right>=$"+" {:0.3f}".format(nums_from_aas[jj]),xy=(3,55),backgroundcolor='white',fontsize=9)
            #imwidth = images[jj].shape[0]
            #radiiMpc,SB,SBerr = get_intensity_profile_physical(images[jj], azradiipx, 100., npix=imwidth, centerx=imwidth//2, centery=imwidth//2)
            #axs[1][jj].plot(radiiMpc,SB,'.')
            #axs[1][jj].set_xscale('log')
            #axs[1][jj].set_yscale('log')
            axs[jj].set_xlabel("pixels")
            axs[jj].set_ylabel("pixels")
        plt.tight_layout()
        plt.savefig("RASS_stacking_TPP.pdf",dpi=500)
        #axs[0].set_ylabel(r"Surface Brightness [cts s$^{-1}$ Mpc$^2$]")
        plt.show()
