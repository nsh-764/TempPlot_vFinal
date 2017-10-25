# ****************************************************************************
# Author: Nikhil S Hubballi <nikhil.hubballi@gmail.com>                      *
# Date: July, 2016                                                           *
# Purpose: to fit modified black body function to the observed data points   *
# from Planck and 100 micron IRAS fluxes. Query and get temperature plot,    *
# optical depth plot, spectral emissivity index plot and also the relation   *
# between temperature and spectral emissivity index for the entire region    *
# of molecular cloud under inspection                                        *
# ****************************************************************************

"""
Program to find the temperature map, spectral emissivity index map, optical depth map
of the molecular cloud
Inputs: 1. Planck images (143GHz, 217GHz, 353GHz, 545GHz, 857GHz) of the source
        2. IRAS image (Iris 100 micro meter) (fits images)
        with header files containing pixel scale ('CDELT1','CDELT2'), dimensions
Outputs: 1. Temperature plot (Temp) of the whole region of molecular cloud
         2. Optical depth plot (tau0)
         3. Spectral emissivity plot (beta)
         4. Temperature vs Beta plot (Relation)
         5. Histograms of temperature and beta
"""
from psf import *
from pypher import *
from lmfit import minimize, Parameters
import matplotlib.pyplot as plt
from astropy.io import fits
from numpy import *
import numpy as np
from astropy.convolution import convolve
import easygui

# Read the input images ************************************************************************************************
print ('>>> Input data')
cd = os.getcwd()
planck143 = easygui.fileopenbox("143GHz image")
f143 = fits.open(planck143)
data143 = fits.getdata(planck143)
print ('>>>  143 GHz image received')

planck217 = easygui.fileopenbox("217GHz image")
f217 = fits.open(planck217)
data217 = fits.getdata(planck217)
print ('>>>  217 GHz image received')

planck353 = easygui.fileopenbox("353GHz image")
f353 = fits.open(planck353)
data353 = fits.getdata(planck353)
print ('>>>  353 GHz image received')

planck545 = easygui.fileopenbox("545GHz image")
f545 = fits.open(planck545)
data545 = fits.getdata(planck545)
print ('>>>  545 GHz image received')

planck857 = easygui.fileopenbox("857GHz image")
f857 = fits.open(planck857)
data857 = fits.getdata(planck857)
print ('>>>  857 GHz image received')

iras3000 = easygui.fileopenbox("IRIS100 image")
f3000 = fits.open(iras3000)
data3000 = fits.getdata(iras3000)
print ('>>>  3000 GHz image received')

'''Dimension of the images'''
dimx = int(f857[0].header['NAXIS1'])
dimy = int(f857[0].header['NAXIS2'])
print ('Image data received....')

# PSF and Kernel Generation ********************************************************************************************
print ('>>> Generating PSF and Kernel')
pl_scale = float(np.abs(f857[0].header['CDELT1'])) * 60  # Pixel Scale
FWHM = [7.1 / pl_scale, 5.5 / pl_scale, 5.0 / pl_scale, 5.0 / pl_scale, 5.0 / pl_scale, 2 / pl_scale]
fwhm_ref = np.amax(FWHM)
k = 'target'
psf_target = PointSpreadFunction(fwhm_ref, fwhm_ref, k)  # Reference PSF: psf143 in this case
psf_target = str(cd) + '/PSF/psf' + str(k) + '.fits'
psf = ['PSF1', 'PSF2', 'PSF3', 'PSF4', 'PSF5', 'PSF6']
kernel = ['K1', 'K2', 'K3', 'K4', 'K5', 'K6']
for i in range(len(FWHM)):
    psf[i] = PointSpreadFunction(FWHM[i], fwhm_ref, i)  # PSF (re-sampled to reference psf size)
    psf_source = str(cd) + '/PSF/psf' + str(i) + '.fits'
    kernel[i] = pypher(psf_source, psf_target, kernel[i], pl_scale)  # Kernel
print ('Done...')
# Convolution of images ************************************************************************************************
print ('>>> Convolving Images')
pix_143 = convolve(data143, kernel[0])
pix_143 *= 358.040  # Conversion from K-->MJy/sr
fits.writeto(str(cd) + '/Convolved/conv143.fits', pix_143, clobber=True)
# *******
pix_217 = convolve(data217, kernel[1])
pix_217 *= 415.465  # Conversion from K-->MJy/sr
fits.writeto(str(cd) + '/Convolved/conv217.fits', pix_217, clobber=True)
# *******
pix_353 = convolve(data353, kernel[2])
pix_353 *= 246.543  # Conversion from K-->MJy/sr
fits.writeto(str(cd) + '/Convolved/conv353.fits', pix_353, clobber=True)
# *******
pix_545 = convolve(data545, kernel[3])
fits.writeto(str(cd) + '/Convolved/conv545.fits', pix_545, clobber=True)
# *******
pix_857 = convolve(data857, kernel[4])
fits.writeto(str(cd) + '/Convolved/conv857.fits', pix_857, clobber=True)
# *******
pix_3000 = convolve(data3000, kernel[5])
fits.writeto(str(cd) + '/Convolved/conv3000.fits', pix_3000, clobber=True)
print ('Done...')
# Pixel by pixel black body curve fitting ******************************************************************************
print ('>>> Fitting black body curve')
'''define objective function: returns the array to be minimized.'''


def fcn2min(params, x, data):
    """ model decaying sine wave, subtract data"""
    tau0 = params['tau0'].value
    beta = params['beta'].value
    Td = params['Td'].value

    model = A11 * tau0 * (xnu / 1200.0e9) ** beta * xnu ** 3 / (np.exp(A12 * (xnu / Td)) - 1)

    return model - yvalu


tmp = np.zeros(shape=(dimx, dimy))
beta = np.zeros(shape=(dimx, dimy))
tau0 = np.zeros(shape=(dimx, dimy))
f = open(str(cd) + '/Results/data.dat', 'w')
for x in range(dimx):
    for y in range(dimy):
        # ******************************************************************************************
        A11 = 1.4733e-50 / 1.0e-20
        A12 = 4.8e-11
        # ******************************************************************************************
        # data to be fitted
        xwave = np.array([100.0, 350.0, 550.0, 850.0, 1382.5, 2100.0])
        xnu = 3.0e8 / (xwave * 1.0e-6)
        yvalu = np.array([pix_3000[x][y], pix_857[x][y], pix_545[x][y], pix_353[x][y], pix_217[x][y], pix_143[x][y]])
        # ******************************************************************************************
        params = Parameters()
        params.add('tau0', value=1.0e-3, min=1.0e-5)
        params.add('beta', value=2.0, min=1.0, max=2.2)
        params.add('Td', value=20.0, min=2.7, max=1000.0)
        # do fit, here with least square model
        result = minimize(fcn2min, params, args=(xwave, yvalu))
        # calculate final result
        final = yvalu + result.residual
        # write error report
        # report_fit(result.params)
        # *******************************************************************************************
        plck = 6.63e-34
        kb = 1.38e-23
        cl = 3.0e8
        tmp[x][y] = result.params['Td'].value
        tau0[x][y] = result.params['tau0'].value
        beta[x][y] = result.params['beta'].value
        nu0 = 1200.0e9
        print("%f \t %f \t %f \t %f \t %f \n" % (x, y, tmp[x][y], tau0[x][y], beta[x][y]))
        f.write('{0:f} {1:f} {2:f} {3:f} {4:f} \n'.format(x, y, tmp[x][y], tau0[x][y], beta[x][y]))
f.close()
print ('Done...')
#  Results and Plots****************************************************************************************************
print ('>>> plotting results')
''' Temperature Plot of the cloud '''
fig1 = plt.figure(1)
fig1 = plt.imshow(tmp, origin='lower')
fig1 = plt.colorbar()
fits.writeto(str(cd) + '/Results/Temp_plot.fits', tmp, clobber=True)

''' Optical Depth map of the cloud '''
fig2 = plt.figure(2)
fig2 = plt.imshow(tau0, origin='lower')
fig2 = plt.colorbar()
fits.writeto(str(cd) + '/Results/Optical_depth.fits', tau0, clobber=True)

''' Spectral Emissivity index map of the cloud '''
fig3 = plt.figure(3)
fig3 = plt.imshow(beta, origin='lower')
fig3 = plt.colorbar()
fits.writeto(str(cd) + '/Results/Beta.fits', beta, clobber=True)

''' Relation between Temperature and spectral emissivity index '''
fig4 = plt.figure(4)
fig4 = plt.scatter(tmp, beta, s=0.01, edgecolor='red')
fig4 = plt.xlim(12, 22)
fig4 = plt.ylim(1.2, 2.2)
fig4 = plt.xlabel(r'$Temperature$')
fig4 = plt.ylabel(r'$Beta$, Spectral Emissivity Index')
fig4 = plt.grid(True)
fig4 = plt.savefig(str(cd) + '/Results/Temp_beta.png')

''' Histogram of temperature values '''
hist1 = plt.figure(5)
data = loadtxt(str(cd) + "/Results/data.dat")
x = data[:, 2]
bins = np.linspace(10, 25, 1000)
y, binEdges = np.histogram(x, bins)  # alpha=0.5
bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
hist1 = plt.plot(bincenters, y, '-')
hist1 = plt.xlabel(r'$Temperature$')
hist1 = plt.ylabel('Number of Pixels')
hist1 = plt.savefig(str(cd) + '/Results/Temp_hist.png')

''' Histogram of beta values '''
hist2 = plt.figure(6)
x = data[:, 4]
bins = np.linspace(1.0, 2.1, 500)
y, binEdges = np.histogram(x, bins)  # alpha=0.5
bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
hist2 = plt.plot(bincenters, y, '-')
hist2 = plt.xlabel(r'$Beta$')
hist2 = plt.ylabel('Number of Pixels')
hist2 = plt.savefig(str(cd) + '/Results/Beta_hist.png')
print ('Done...')
# **********************************************************************************************************************
plt.show()
