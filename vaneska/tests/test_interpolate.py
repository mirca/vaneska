import numpy as np
import tensorflow as tf

from ..interpolate import ScipyRectBivariateSpline

def test_ScipyRectBivariateSpline():
    x, y = np.mgrid[-5:5:10*1j, -5:5:20*1j]
    x = tf.Variable(x, dtype=tf.float64)
    y = tf.Variable(y, dtype=tf.float64)

    z = x ** 2 + y ** 3
    dzdx = 2 * x
    dzdy = 3 * y ** 2

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        interp = ScipyRectBivariateSpline(sess.run(x[:, 0]), sess.run(y[0]),
                                          sess.run(z))
        zp = interp(x[:, 0], y[0])
        grad = tf.gradients(zp, [x, y])
        assert sess.run(tf.reduce_sum(tf.subtract(grad[0], dzdx))) < 1e-6
        assert sess.run(tf.reduce_sum(tf.subtract(grad[1], dzdy))) < 1e-6
