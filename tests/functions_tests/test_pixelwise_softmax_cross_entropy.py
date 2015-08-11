import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


if cuda.available:
    cuda.init()


class TestPixelwiseSoftmaxCrossEntropy(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (10, 3, 16, 16)).astype(numpy.float32)
        self.t = numpy.random.randint(
            3, size=(10, 1, 16, 16)).astype(numpy.int32)

    def check_forward(self, x0_data, x1_data):
        x = chainer.Variable(x0_data)
        t = chainer.Variable(x1_data)
        loss = functions.pixelwise_softmax_cross_entropy(x, t)
        loss_value = cuda.to_cpu(loss.data)
        self.assertEqual(loss_value.dtype, numpy.float32)
        self.assertEqual(loss_value.shape, ())

        # Compute expected value
        loss_expect = 0.0
        for i in numpy.ndindex(self.t.shape):
            _x = self.x[i[0], :, i[2], i[3]]
            _x -= numpy.max(_x)
            numpy.exp(_x, out=_x)
            _x /= numpy.sum(_x)
            loss_expect += -numpy.log(_x[self.t[i]])
        loss_expect /= self.t.size

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    @attr.gpu
    @condition.retry(3)
    def test_forwrad_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    def check_backward(self, x0_data, x1_data):
        x = chainer.Variable(x0_data)
        t = chainer.Variable(x1_data)
        loss = functions.pixelwise_softmax_cross_entropy(x, t)
        loss.backward()
        self.assertEqual(None, t.grad)

        func = loss.creator
        f = lambda: func.forward((x.data, t.data))
        gx, = gradient_check.numerical_grad(f, (x.data,), (1,), eps=0.02)

        gradient_check.assert_allclose(gx, x.grad)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))


testing.run_module(__name__, __file__)
