# ****************************************************************************
# Author: Nikhil S Hubballi                                                  *
# Date: July, 2016                                                           *
# Purpose: to fit modified black body function to the observed data points   *
# from Planck and 100 micron IRAS fluxes. Query and get results for          *
# for individual pixels.                                                     *
# ****************************************************************************
"""
Program to find the black body curve fit of the molecular cloud for any position in the image
Inputs: 1. Planck images (143GHz, 217GHz, 353GHz, 545GHz, 857GHz) of the source
        2. IRAS image (Iris 100 micro-meter) (fits images)
        with header files containing pixel scale ('CDELT1','CDELT2'), dimensions
Output: Black body curve fit plot for the position in the molecular cloud with
        temperature and spectral emissivity index(beta) values
"""
from psf import *
from pypher import *
from lmfit import minimize, Parameters, report_fit
import matplotlib.pyplot as plt
from astropy.io import fits
from numpy import *
import numpy as np
from astropy.convolution import convolve
import easygui


# Read the input images ************************************************************************************************
print ('>>> Input data')
cd = os.getcwd()

planck143 = easygui.fileopenbox("143 Ghz image")
f143 = fits.open(planck143)
data143 = fits.getdata(planck143)
print ('>>>  143 GHz image received')

planck217 = easygui.fileopenbox("217 Ghz image")
f217 = fits.open(planck217)
data217 = fits.getdata(planck217)
print ('>>>  217 GHz image received')

planck353 = easygui.fileopenbox("353 Ghz image")
f353 = fits.open(planck353)
data353 = fits.getdata(planck353)
print ('>>>  353 GHz image received')

planck545 = easygui.fileopenbox("545 Ghz image")
f545 = fits.open(planck545)
data545 = fits.getdata(planck545)
print ('>>>  545 GHz image received')

planck857 = easygui.fileopenbox("857 Ghz image")
f857 = fits.open(planck857)
data857 = fits.getdata(planck857)
print ('>>>  857 GHz image received')

iras3000 = easygui.fileopenbox("IRIS100 image")
f3000 = fits.open(iras3000)
data3000 = fits.getdata(iras3000)
print ('>>>  3000 GHz image received')

'''Dimension of the images'''
dimx = float(f857[0].header['NAXIS1'])
dimy = float(f857[0].header['NAXIS2'])
print ('Image data received....')

# PSF and Kernel Generation ********************************************************************************************
print ('>>> Generating PSF and Kernel')
pl_scale = float(np.abs(f857[0].header['CDELT1'])) * 60  # Pixel Scale
FWHM = [7.1 / pl_scale, 5.5 / pl_scale, 5.0 / pl_scale, 5.0 / pl_scale, 5.0 / pl_scale, 2.0 / pl_scale]
fwhm_ref = np.amax(FWHM)
k = 'target'
psf_target = PointSpreadFunction(fwhm_ref, fwhm_ref, k)  # Reference PSF: psf143 in this case
psf_target = str(cd)+'/PSF/psf'+str(k)+'.fits'
psf = ['PSF1', 'PSF2', 'PSF3', 'PSF4', 'PSF5', 'PSF6']
kernel = ['K1', 'K2', 'K3', 'K4', 'K5', 'K6']
for i in range(len(FWHM)):
    psf[i] = PointSpreadFunction(FWHM[i], fwhm_ref, i+1)  # PSF (re-sampled to reference psf size)
    psf_source = str(cd)+'/PSF/psf'+str(i)+'.fits'
    kernel[i] = pypher(psf_source, psf_target, kernel[i], pl_scale)  # Kernel
print ('Done...')

# Convolution of images ************************************************************************************************
print ('>>> Convolving Images')
pix_143 = convolve(data143, kernel[0])
pix_143 *= 358.040  # Conversion from K-->MJy/sr
fits.writeto(str(cd)+'/Convolved/conv143.fits', pix_143, clobber=True)
# *******
pix_217 = convolve(data217, kernel[1])
pix_217 *= 415.465  # Conversion from K-->MJy/sr
fits.writeto(str(cd)+'/Convolved/conv217.fits', pix_217, clobber=True)
# *******
pix_353 = convolve(data353, kernel[2])
pix_353 *= 246.543  # Conversion from K-->MJy/sr
fits.writeto(str(cd)+'/Convolved/conv353.fits', pix_353, clobber=True)
# *******
pix_545 = convolve(data545, kernel[3])
fits.writeto(str(cd)+'/Convolved/conv545.fits', pix_545, clobber=True)
# *******
pix_857 = convolve(data857, kernel[4])
fits.writeto(str(cd)+'/Convolved/conv857.fits', pix_857, clobber=True)
# *******
pix_3000 = convolve(data3000, kernel[5])
fits.writeto(str(cd)+'/Convolved/conv3000.fits', pix_3000, clobber=True)
print ('Done...')

# Pixel by pixel black body curve fitting ******************************************************************************
# ******INPUT******
inp = raw_input('Try? (y/n) :')

while str(inp) == 'y':
    x = input("Provide x co-ord: ")
    y = input("\nProvide y co-ord: ")
    print ('>>> Fitting black body curve')
    # *********************************************************************
    A11 = 1.4733e-50 / 1.0e-20
    A12 = 4.8e-11
    # *********************************************************************
    # data to be fitted
    xwave = np.array([100.0, 350.0, 550.0, 850.0, 1382.5, 2100.0])
    xnu = 3.0e8 / (xwave * 1.0e-6)
    yvalu = np.array([pix_3000[y][x], pix_857[y][x], pix_545[y][x], pix_353[y][x], pix_217[y][x], pix_143[y][x]])

    # *********************************************************************
    '''define objective function: returns the array to be minimized'''

    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data"""
        tau0 = params['tau0'].value
        beta = params['beta'].value
        Td = params['Td'].value

        model = A11 * tau0 * (xnu / 1200.0e9) ** beta * xnu ** 3 / (np.exp(A12 * (xnu / Td)) - 1)
        return model - yvalu


    params = Parameters()
    params.add('tau0', value=1.0e-3, min=1.0e-5)
    params.add('beta', value=2.0, min=1.0, max=2.2)
    params.add('Td', value=20.0, min=2.7, max=1000.0)

    # do fit, here with least square model
    result = minimize(fcn2min, params, args=(xwave, yvalu))
    # calculate final result
    final = yvalu + result.residual
    # write error report
    report_fit(result.params)
    # ****************************************************
    plck = 6.63e-34
    kb = 1.38e-23
    cl = 3.0e8
    Temp = result.params['Td'].value
    tau0 = result.params['tau0'].value
    beta = result.params['beta'].value
    nu0 = 1200.0e9

    # ****************************************************
    xw = empty(3000)
    for i in range(1, 3000):
        xw[i] = i
    f1 = open(str(cd)+'/Results/Inu_nuTFIT.dat', 'w')
    for j in range(1, 3000):
        nu = cl / (xw[j] * 1.0e-6)
        A11 = (2 * plck * nu * nu * nu) / (cl * cl)
        A12 = (plck * nu) / (kb * Temp)
        A13 = 1 / (exp(A12) - 1)
        A14 = tau0 * pow((nu / nu0), beta)
        Inu = A11 * A13 * A14 * 1.0e+20
        f1.write('{0:f} {1:f} {2:.30f}\n'.format(nu, xw[j], Inu))
    f1.close()
    nuFITa, wlFITa, InuFITa = loadtxt(str(cd)+'/Results/Inu_nuTFIT.dat', unpack=True)
    print ('Model fitted...')
    print ('Plotting the blackbody fit...')
    if Temp <= 16.5:
        T = 'Molecular cloud region'
    else:
        T = 'Non-molecular cloud region'
    # *****************************************************
    plt.figure(figsize=(10, 7))
    # Black Body fit (continuous)
    plt.plot(wlFITa, InuFITa, 'b-', label="Modified BB Fit")
    # Black body fit values (data points)
    plt.plot(xwave, final, 'rs', markersize=10, markerfacecolor='white', label="Fit Values")
    # Pixel values
    plt.plot(xwave, yvalu, 'ro', label="Pixel Value")
    # plot parameters
    plt.xscale('log'), plt.yscale('log')
    plt.xlim(30.0, 5000.0), plt.ylim(0.01, 1000.0)
    plt.text(40, 300, str(T)+'\nT = '+str(round(Temp, 3))+'K , Beta = '+str(round(beta, 3)), fontdict=None)
    plt.xlabel('Wavelength (microns)'), plt.ylabel('Intensity (MJy/sr)')
    plt.legend(loc='upper right'), plt.legend(numpoints=1)
    plt.savefig(str(cd)+'/Results/'+str(T)+'.png')
    print ('Done...')
    # Plot *****************************************************************************************************************
    plt.show()
    inp = raw_input('Try another? (y/n) :')
 




