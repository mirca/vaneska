"""
This module has the code to infer PSF models.

Interface:
    classes should be parametrized by, at least, flux and
    centroid positions, which should be of type tf.Variable.
"""

import numpy as np
import tensorflow as tf

class Gaussian:
    """
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
        psf = tf.exp(-(a * (self.x - xo) ** 2
                       + 2 * b * (self.x - xo) * (self.y - yo)
                       + c * (self.y - yo) ** 2))
        psf_sum = tf.reduce_sum(psf)
        return flux * psf / psf_sum
