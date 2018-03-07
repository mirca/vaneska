"""
This module has the code to infer PSF models.

Interface:
    classes should be parametrized by, at least, flux and
    centroid positions, which should be of type tf.Variable.

TODO:

"""

import numpy as np
import tensorflow as tf

class Gaussian:
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
        psf = tf.exp(-(a * (self.x - xo) ** 2
                       + 2 * b * (self.x - xo) * (self.y - yo)
                       + c * (self.y - yo) ** 2))
        psf_sum = tf.reduce_sum(psf)
        return flux * psf / psf_sum

class KeplerPRF:
    def __init__(self, channel, shape, col_ref, row_ref):
        self.channel = channel
        self.shape = shape
        self.col_ref = col_ref
        self.row_ref = row_ref

        # self.x, self.y should be constant tensors
        self.x, self.y, self.interp = self.init_prf()
        x = tf.placeholder(dtype=tf.float64)
        y = tf.placeholder(dtype=tf.float64)

    def __call__(self, flux, xc, yc):
        return self.evaluate(flux, xc, yc)

    def evaluate(self, flux, xc, yc):
        dx = self.x - xc
        dy = self.y - yc
        self.interp_tf = tf.py_func(self.interp, [dx, dy], tf.float64)
        return flux * self.interp_tf

    def init_prf(self):
        pass
