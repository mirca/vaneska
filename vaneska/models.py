"""
This module has the code to infer PSF models.

Interface:
    classes should be parametrized by, at least, flux and
    centroid positions, which should be of type tf.Variable.

TODO:

"""
import math

from astropy.io import fits as pyfits
from lightkurve.utils import channel_to_module_output
import numpy as np
import tensorflow as tf

from .interpolate import ScipyRectBivariateSpline


class Model:
    """
    Pretty dumb Gaussian model.

    Attributes
    ----------
    shape : tuple
        shape of the TPF. (row_shape, col_shape)
    col_ref, row_ref : int, int
        column and row coordinates of the bottom
        left corner of the TPF
    """
    def __init__(self, shape, col_ref, row_ref):
        self.shape = shape
        self.col_ref = col_ref
        self.row_ref = row_ref
        self._init_grid()

    def _init_grid(self):
        r, c = self.row_ref, self.col_ref
        s1, s2 = self.shape
        self.y, self.x = np.mgrid[r:r+s1-1:1j*s1, c:c+s2-1:1j*s2]


class Gaussian(Model):
    def __call__(self, *params):
        return self.evaluate(*params)

    def evaluate(self, flux, xo, yo, a, b, c):
        """
        Evaluate the Gaussian model

        Parameters
        ----------
        flux : tf.Variable
        xo, yo : tf.Variable, tf.Variable
            Center coordiantes of the Gaussian.
        a, b, c : tf.Variable, tf.Variable
            Parameters that control the rotation angle
            and the stretch along the major axis of the Gaussian,
            such that the matrix M = [a b ; b c] is positive-definite.

        References
        ----------
        https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
        """
        dx = self.x - xo
        dy = self.y - yo
        psf = tf.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))
        psf_sum = tf.reduce_sum(psf)
        return flux * psf / psf_sum


class Moffat(Model):
    def __call__(self, *params):
        return self.evaluate(*params)

    def evaluate(self, flux, xo, yo, a, b, c, beta):
        dx = self.x - xo
        dy = self.y - yo
        psf = tf.divide(1., tf.pow(1. + a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2, beta))
        psf_sum = tf.reduce_sum(psf)
        return flux * psf / psf_sum


class KeplerPRF:
    def __init__(self, channel, shape, column, row):
        self.channel = channel
        self.shape = shape
        self.column = column
        self.row = row
        self.x, self.y, self.prf_func, self.supersampled_prf = self.init_prf()

    def __call__(self, flux, xc, yc):
        return self.evaluate(flux, xc, yc)

    def evaluate(self, flux, xc, yc):
        dx = tf.subtract(self.x, xc)
        dy = tf.subtract(self.y, yc)
        return flux * self.prf_func(dy, dx)

    def _read_prf_files(self, path, ext):
        prf_file = pyfits.open(path)
        prf_data = prf_file[ext].data
        # looks like these data below are the same for all prf calibration files
        crval1p = prf_file[ext].header['CRVAL1P']
        crval2p = prf_file[ext].header['CRVAL2P']
        cdelt1p = prf_file[ext].header['CDELT1P']
        cdelt2p = prf_file[ext].header['CDELT2P']
        prf_file.close()

        return prf_data, crval1p, crval2p, cdelt1p, cdelt2p

    def init_prf(self):
        min_prf_weight = 1e-6 # minimum weight for the PRF
        module, output = channel_to_module_output(self.channel)

        # determine suitable PRF calibration file
        if module < 10:
            prefix = 'kplr0'
        else:
            prefix = 'kplr'
        prfs_url_path = "http://archive.stsci.edu/missions/kepler/fpc/prf/extracted/"
        prf_file_path = prfs_url_path + prefix + str(module) + '.' + str(output) + '_2011265_prf.fits'

        # get the data of the PRF for the 5 supersampled PRFs
        n_prfs = 5
        prf_array = [0] * n_prfs
        crval1p = np.zeros(n_prfs, dtype='float32')
        crval2p = np.zeros(n_prfs, dtype='float32')
        for i in range(n_prfs):
            prf_array[i], crval1p[i], crval2p[i], cdelt1p, cdelt2p = self._read_prf_files(prf_file_path, i+1)
        prf_array = np.array(prf_array)

        column_array = np.arange(.5 * (1. - prf_array[0].shape[1]),
                                 .5 * (1. + prf_array[0].shape[1])) * cdelt1p
        row_array = np.arange(.5 * (1. - prf_array[0].shape[0]),
                              .5 * (1. + prf_array[0].shape[0])) * cdelt2p

        prf = np.zeros_like(prf_array[0])
        ref_column = self.column + .5 * self.shape[1]
        ref_row = self.row + .5 * self.shape[0]

        # Weight those 5 PRFs w.r.t. the distance from the target star
        prf_weights = np.sqrt((ref_column - crval1p) ** 2
                              + (ref_row - crval2p) ** 2)
        mask = prf_weights < min_prf_weight
        prf_weights[mask] = min_prf_weight

        normalized_prf = np.sum(np.sum(prf_array, axis=(1, 2)) / prf_weights)
        normalized_prf /= np.sum(normalized_prf * cdelt1p * cdelt2p)

        # give the PRF a "parametrizable" form
        prf_spline = ScipyRectBivariateSpline(row_array, column_array, normalized_prf)
        xp = np.arange(self.column + .5, self.column + self.shape[1] + .5)
        yp = np.arange(self.row + .5, self.row + self.shape[0] + .5)

        return [tf.constant(xp, dtype=tf.float64),
                tf.constant(yp, dtype=tf.float64),
                prf_spline, normalized_prf]
