import numpy as np
import scipy
import tensorflow as tf
from tensorflow.python.framework import ops

# interp is an object of scipy.interpolate.RectBivariateSpline

class ScipyRectBivariateSpline:
    def __call__(self, x, y, name=None):
        return self.interpolate(x, y, name)

    def __init__(self, x, y, z, bbox=[None, None, None, None], kx=3, ky=3, s=0):
        self.interp = scipy.interpolate.RectBivariateSpline(x, y, z, bbox,
                                                            kx, ky, s)

    def _Interpolate(self, x, y):
        return self.interp(x, y)

    def _InterpolateGradImpl(self, x, y, bz):
        return (np.sum(bz * self.interp(x, y, dx=1), axis=1),
                np.sum(bz * self.interp(x, y, dy=1), axis=0))

    def _InterpolateGrad(self, op, grads):
        return tf.py_func(self._InterpolateGradImpl,
                          [op.inputs[0], op.inputs[1], grads],
                          [tf.float64, tf.float64])

    def py_func(self, func, inp, Tout, stateful=True, name=None, grad=None):
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1e+8))
        tf.RegisterGradient(rnd_name)(grad)
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name}):
            return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

    def interpolate(self, x, y, name=None):
        with ops.op_scope([x, y], name, "Interpolate") as name:
            result = self.py_func(self._Interpolate,
                                  [x, y],
                                  [tf.float64],
                                  name=name,
                                  grad=self._InterpolateGrad)
            return result[0]
