from tensorflow.python.framework import ops

# interp is an object of scipy.interpolate.RectBivariateSpline

def _Interpolate(x, y):
    return interp(x, y)

def _InterpolateGradImpl(x, y, bz):
    return bz * interp(x, y, dx=1), bz * interp(x, y, dy=1)

def _InterpolateGrad(op, grads):
    return tf.py_func(_InterpolateGradImpl, [op.inputs[0], op.inputs[1], grads[0]],
                      [tf.float64, tf.float64])

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1e+8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def interpolate(x, y, name=None):
    with ops.op_scope([x, y], name, "Interpolate") as name:
        result = py_func(_Interpolate,
                         [x, y],
                         [tf.float64],
                         name=name,
                         grad=_InterpolateGrad)
        return result[0]
