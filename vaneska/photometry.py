"""This module implements the necessary code
to perform PSF photometry.
"""

import tqdm
import numpy as np

class PSFPhotometry:
    """
    Estimate the parameters of a PSF model.

    Attributes
    ----------
    optimizer : instance of tf.contrib.opt.ScipyOptimizerInterface
        The optimizer that will be used to estimate the parameters
        of the PSF model.
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def fit(self, pixel_flux, data_placeholder, var_list, feed_dict, session):
        """
        Parameters
        ----------
        pixel_flux : ndarray
            The TPF-like pixel flux time series. The first dimension
            must represent time, and the remaining two dimensions
            must represent the spatial dimensions.
        data_placeholder : tf.placeholder
            A placeholder which will be used to pass the n-th time stamp
            to `self.optimizer.minimize`.
        var_list : list
            The list of parameters (as tensors) to optimize for.
        feed_dict : dict
            Dictionary of additional arguments used to feed the loss function.
        session : instance of tf.Session
        """
        opt_params = []
        cadences = range(pixel_flux.shape[0])

        for n in tqdm.tqdm(cadences):
            feed_dict.update({data_placeholder: pixel_flux[n]})
            self.optimizer.minimize(session=session, feed_dict=feed_dict)
            opt_params.append([session.run(var) for var in var_list])

        return opt_params
