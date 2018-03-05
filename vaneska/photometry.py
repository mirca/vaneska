"""This module implements the necessary code
to perform PSF photometry.
"""

import tqdm

class PSFPhotometry:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def fit(self, pixel_flux, data_placeholder, var_list, session):
        opt_params = []
        cadences = range(pixel_flux.shape[0])

        for n in tqdm.tqdm(cadences):
            self.optimizer.minimize(session=session, feed_dict={data_placeholder: pixel_flux[n]})
            opt_params.append([session.run(var) for var in var_list])

        return np.asarray(opt_params)
