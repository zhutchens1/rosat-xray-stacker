import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import pickle
import numpy as np
from rosat_xray_stacker import rosat_xray_stacker, measure_optimal_snr, get_intensity_profile_physical
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm

"""
This file performs all the neccessary prepration for 
ECO RASS images prior to stacking: downloading images,
sorting them into sorting them into good vs. poor coverage,
and scaling them to the proper distance.  
"""
if __name__=='__main__':
    do_preparation = False
    scale_images = False
    stack_images = True
    analyze = True


    if do_preparation:
        ecocsv = pd.read_csv("../g3groups/ECOdata_G3catalog_luminosity.csv")
        ecocsv = ecocsv[ecocsv.g3fc_l==1] # centrals only

        eco = rosat_xray_stacker(ecocsv.g3grp_l, ecocsv.g3grpradeg_l, ecocsv.g3grpdedeg_l, ecocsv.g3grpcz_l, centralname=ecocsv.name, \
            surveys=['RASS-Int Soft'])
        #if not images_downloaded: eco.download_images('./g3rassimages/soft/eco_soft/')
        #eco.sort_images('./g3rassimages/soft/eco_soft_cts/')
        #eco.measure_SNR_1Mpc('./g3rassimages/soft/eco_soft_cts/', 5)
        eco.mask_point_sources('./g3rassimages/soft/eco_soft_cts/', './g3rassimages/soft/eco_soft_cts_ptsrc_rm/',\
                 examine_result=False, smoothsigma=3, starfinder_threshold=5)

    if scale_images:
        ecocsv = pd.read_csv("../g3groups/ECOdata_G3catalog_luminosity.csv")
        ecocsv = ecocsv[ecocsv.g3fc_l==1] # centrals only
        ncores = 20
        eco = rosat_xray_stacker(ecocsv.g3grp_l, ecocsv.g3grpradeg_l, ecocsv.g3grpdedeg_l, ecocsv.g3grpcz_l, centralname=ecocsv.name, \
            surveys=['RASS-Int Soft'])
        subframes = np.array_split(ecocsv,ncores)
        subobjects = [rosat_xray_stacker(tdf.g3grp_l, tdf.g3grpradeg_l, tdf.g3grpdedeg_l, tdf.g3grpcz_l, centralname=tdf.name,\
            surveys=['RASS-Int Soft']) for tdf in subframes]

        import multiprocessing
        processes=[None]*ncores
        for jj in range(0,ncores):
            processes[jj]=multiprocessing.Process(target=subobjects[jj].scale_subtract_images,\
                args=("./g3rassimages/soft/eco_soft_exp/", "./g3rassimages/soft/eco_soft_exp_scaled/",True,512,45,70.,2530,7470,True))
        for jj in range(0,ncores):
            processes[jj].start()
        for jj in range(0,ncores):
            processes[jj].join()

        #processes=[None]*ncores
        #for jj in range(0,ncores):
        #    processes[jj]=multiprocessing.Process(target=subobjects[jj].scale_subtract_images,\
        #        args=("./g3rassimages/soft/eco_soft_cts_ptsrc_rm/", "./g3rassimages/soft/eco_soft_cts_scaled/",True,512,45,70.,2530.,7470.,True))
        #for jj in range(0,ncores):
        #    processes[jj].start()
        #for jj in range(0,ncores):
        #    processes[jj].join()
        #eco.scale_subtract_images("./g3rassimages/soft/eco_soft_cts/", "./g3rassimages/soft/eco_soft_cts_scaled/", crop=True, progressConf=True, imwidth=512)
        #eco.scale_subtract_images("./g3rassimages/soft/eco_soft_exp/", "./g3rassimages/soft/eco_soft_exp_scaled/", crop=True, progressConf=True, imwidth=512)
        eco.to_pickle("eco_soft_prepared_images_110121.pkl")
    
    if stack_images:
        ecocsv = pd.read_csv("../g3groups/ECOdata_G3catalog_luminosity.csv")
        ecocsv = ecocsv[ecocsv.g3fc_l==1] # centrals only
        #eco = pickle.load(open("eco_prepared_images_110121.pkl",'rb'))
        eco = rosat_xray_stacker(ecocsv.g3grp_l, ecocsv.g3grpradeg_l, ecocsv.g3grpdedeg_l, ecocsv.g3grpcz_l, centralname=ecocsv.name, \
            surveys=['RASS-Int Soft'])

        stackID,nbin,bincenters,counts=eco.stack_images("./g3rassimages/soft/eco_soft_cts_scaled/",np.asarray(ecocsv.g3logmh_l), [11,12.1,13.3,15])
        stackID,nbin,bincenters,times=eco.stack_images("./g3rassimages/soft/eco_soft_exp_scaled/",np.asarray(ecocsv.g3logmh_l), [11,12.1,13.3,15])
        images = [ct/ex for (ct,ex) in zip(counts,times)]
        stacks = [stackID,nbin,bincenters,counts,times]
        pickle.dump(stacks, open('ecostackedimages_soft_032722.pkl','wb'))

    if analyze:
        import matplotlib.pyplot as plt
        stackID,nbin,bincenters,counts,times = pickle.load(open('ecostackedimages_soft_032722.pkl','rb'))
        images = [ct/ex for (ct,ex) in zip(counts,times)]
        visimages = [gaussian_filter(images[i],2) for i in range(0,len(images))]
        Rvir = ((3*10**bincenters) / (4*np.pi*337*0.3*1.36e11) )**(1/3)
        snrs = [measure_optimal_snr(images[ii],times[ii],7000,Rvir[ii])[0] for ii in range(0,len(images))]
        apfrac = [measure_optimal_snr(images[ii],times[ii],7000,Rvir[ii])[1] for ii in range(0,len(images))]
        maxes = np.asarray([np.max(im) for im in visimages])
        scaleto = np.mean(maxes)-0.5*np.std(maxes)
        azradiipx = np.arange(2,130,2) 
        fig, axs = plt.subplots(nrows=2, ncols=len(bincenters), figsize=(19,7))
        for jj in range(0,len(bincenters)):
            axs[0][jj].set_title(r"$\left<\log M_{337}\right> = $ "+str(bincenters[jj]))
            #axs[0][jj].imshow(visimages[jj], extent=[-150,150,-150,150],vmax=scaleto,vmin=0)
            #axs[0][jj].imshow(visimages[jj], extent=[-150,150,-150,150], norm=LogNorm())
            axs[0][jj].imshow(visimages[jj], norm=LogNorm())
            axs[0][jj].annotate(r"{} Images".format(nbin[jj]), xy=(2,6), backgroundcolor='white',fontsize=8)
            axs[0][jj].annotate(r"$S/N = $"+"{:0.3f}".format(snrs[jj]), xy=(2,10), backgroundcolor='white',fontsize=8)

            imwidth = images[jj].shape[0]
            radiiMpc,SB,SBerr = get_intensity_profile_physical(images[jj], azradiipx, 100., npix=imwidth, centerx=imwidth//2, centery=imwidth//2)
            axs[1][jj].plot(radiiMpc,SB,'.')
            axs[1][jj].set_xscale('log')
            axs[1][jj].set_yscale('log')
            axs[1][jj].set_xlabel("Radius [Mpc]")
            axs[1][jj].set_ylim(1e-4,1e-1)
        #plt.tight_layout()
        axs[1][0].set_ylabel(r"Surface Brightness [cts s$^{-1}$ Mpc$^2$]")
        plt.suptitle("Soft-Band ROSAT")
        plt.show()
