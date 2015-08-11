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
        self.x0 = numpy.random.uniform(
            -1, 1, (10, 3, 16, 16)).astype(numpy.float32)
        self.x1 = numpy.random.randint(
            3, size=(10, 1, 16, 16)).astype(numpy.int32)

    def check_forward(self, x0_data, x1_data):
        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        loss = functions.pixelwise_softmax_cross_entropy(x0, x1)
        loss_value = cuda.to_cpu(loss.data)
        self.assertEqual(loss_value.dtype, numpy.float32)
        self.assertEqual(loss_value.shape, ())

        # Compute expected value
        loss_expect = 0.0
        tmp = None
        for i in numpy.ndindex(self.x1.shape):
            _x0 = self.x0[i[0], :, i[2], i[3]]
            _x0 -= numpy.max(_x0)
            numpy.exp(_x0, out=_x0)
            _x0 /= numpy.sum(_x0)
            if tmp is None:
                tmp = _x0
            loss_expect += -numpy.log(_x0[self.x1[i]])
        print loss_expect
        print tmp, self.x1[0, :, 0, 0]
        loss_expect /= self.x1.size

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    # @condition.retry(3)
    # def test_forward_cpu(self):
    #     self.check_forward(self.x0, self.x1)

    @attr.gpu
    @condition.retry(3)
    def test_forwrad_gpu(self):
        self.check_forward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1))

    # def check_backward(self, x0_data, x1_data):
    #     x0 = chainer.Variable(x0_data)
    #     x1 = chainer.Variable(x1_data)
    #     loss = functions.pixelwise_softmax_cross_entropy(x0, x1)
    #     loss.backward()
    #
    #     func = loss.creator
    #     f = lambda: func.forward((x0.data, x1.data))
    #     gx0, gx1 = gradient_check.numerical_grad(
    #         f, (x0.data, x1.data), (1,), eps=1e-2)
    #
    #     gradient_check.assert_allclose(gx0, x0.grad)
    #     gradient_check.assert_allclose(gx1, x1.grad)
    #     self.assertEqual(x0.grad.dtype, numpy.float32)
    #     self.assertEqual(x1.grad.dtype, numpy.float32)

    # @condition.retry(3)
    # def test_backward_cpu(self):
    #     self.check_backward(self.x0, self.x1)

    # @attr.gpu
    # @condition.retry(3)
    # def test_backward_gpu(self):
    #     self.check_backward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1))


testing.run_module(__name__, __file__)
