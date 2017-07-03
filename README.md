# TempPlot_vFinal
Temperature Plot and Optical depth map of Star forming molecular clouds using Planck Satellite Data

# ****************************************************************************
# Author: Nikhil S Hubballi <nikhil.hubballi@gmail.com>                      *
# Date: July, 2016                                                           *
# Purpose: to fit modified black body function to the observed data points   *
# from Planck and 100 micron IRAS fluxes. Query and get temperature plot,    *
# optical depth plot, spectral emissivity index plot and also the relation   *
# between temperature and spectral emissivity index for the entire region    *
# of molecular cloud under inspection                                        *
# ****************************************************************************


Packages required for the python program
1. astropy.io
2. lmfit
3. matplotlib
4. numpy
5. scipy
6. astropy.convolution
7. easygui
8. os

Steps to be followed to obtain the temperature map
1. Use Skyview to obtain Planck and IRAS satellite data.
	url: http://skyview.gsfc.nasa.gov/current/cgi/query.pl

2. Specify the images size in both pixels and degrees such that plate scale of the image is 1.7 arcmin.
	Formula = Image size (degrees) * 60 / Image size (pixels) = 1.7

3. Download the data corresponding to Planck 143, 217, 353, 545, 857 GHz and iris 100 corresponding to IRAS.

4. Place the above images in the folder named "Data" (present in the root folder of the program)

5. After opening the terminal from the directory where the programs are located, run the run.py python file.
	~/...../$ python run.py

6. The code will start depending on the proper verification FITS files present in the directory.

7. A user dialogue opens from where each of the corresponding FITS files have to be chosen by the user.

8. The code will take around 2 hours depending on the dimensions of the fits images and the processor capabilities.

9. After the code completion, you'll see three fits files named "Temp_plot.fits", "Optical_depth.fits", "Beta.fits",
"data.dat", "temp_hist.png", "beta_hist.png".

10. For correlation plot, run the "contour_beta_temp.py" python file.
