from astropy.io import fits
from astropy.convolution import Gaussian2DKernel as G2Dk
import scipy.ndimage.interpolation as congrid
import os

cd = os.getcwd()

def PointSpreadFunction(sigma_source, sigma_target, ref):
    """
        Generate the PSF with given parameters

        Parameters
        ----------
        sigma_source : `float`
            source psf standard deviation
        sigma_target : 'float'
            target psf standard deviation
        ref : int
            File number

        Returns
        -------
        output : `numpy.ndarray`
            PSF data array
    """
    psf_source = G2Dk(stddev=sigma_source/2.35482)
    psf_target = G2Dk(stddev=sigma_target/2.35482)
    fac = float(psf_target.shape[0]) / float(psf_source.shape[0])
    psf_resized = congrid.zoom(psf_source, fac, order=3)
    hdu = fits.PrimaryHDU(psf_resized)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(str(cd)+'/PSF/psf'+str(ref)+'.fits', clobber=True)
    return psf_resized

