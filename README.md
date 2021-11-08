# Stacking Diffuse X-ray Emission in Galaxy Groups with ROSAT

This repository stores code enabling the retrieval, processing, and stacking of diffuse X-ray emission in galaxy groups from the ROSAT All-Sky Survey (RASS). As of May 2021, this code is currently under development by the RESOLVE-G3 (Gas in Galaxy Groups) X-ray subteam.

## Code Description
<details>

The main class in this code is `rosat_xray_stacker`. It can be initialized as
```
stacker = rosat_xray_stacker(grpid, grpra, grpdec, grpcz)
```
where `grpid`, `grpra`, `grpdec`, and `grpcz` are the ID, RA (deg), Declination (deg), and cz (km/s) of each
galaxy group. It is also possible to specify the central galaxy name for each group. In principle, galaxies could be passed to this object instead of groups, as long as each entry contains a unique ID number. The `grpid` parameter is used
throughout the methods of the class to sort/organize images and group metadata. The primary methods of this class are

* `to_pickle`: save the stacker object to a serialized Python package.
* `download_images`: obtain RASS images for each group from NASA SkyView.
* `sort_images`: Sort raw RASS images into good or poor coverage based on user specified S/N and zero-count thresholds.
* `measure_SNR_1Mpc`: Measure the signal-to-noise ratio within the central 1 Mpc on-sky of the group image.
* `mask_point_sources`: Mask point sources in RASS images based on a variety of user-chosen parameters.
* `scale_subtract_images`: Scale all images in the catalog to a common distance for stacking.
* `stack_images`: Stack images according to a user defined group property and specified binning.

The relevant input parameters and output for each method is noted in the function docstrings.


</details>

## Package Requirements
<details>

This code uses the following dependencies. 

* Python >=3.6
* NumPy
* Matplotlib (pyplot, colors, cm)
* Astroquery (skyview)
* Astropy (fits, stats)
* Photutils (DAOStarFinder, CircularAperture)

</details>
