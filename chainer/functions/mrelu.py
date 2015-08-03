import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _fwd_mrelu():
    return cuda.elementwise(
        '''
        float* y, const float* x, const float* W, int cdim, int rdim
        ''', '''
        int c = i / rdim % cdim;
        float w = abs(W[c]);
        y[i] = max(x[i] - w, 0.0) + min(x[i] + w, 0.0);
        ''',
        'fwd_mrelu')


class MReLU(function.Function):

    """Margin ReLU function.

    MReLU function is written in elementwise equation as
    :math:`MReLU(x) = \max(x - m, 0) + \min(x + m, 0)`, where :math:`m` is a
    parameter array.

    When the MReLU function is combined with two-dimensional convolution, the
    elements of parameter :math:`m` are typically shared across the same filter
    of different pixels. In order to support such usage, this function supports
    the shape of parameter array that indicates leading dimensions of input
    arrays except the batch dimension.

    Args:
        shape (tuple of ints): Shape of the parameter array.
        init (float): Initial parameter value.

    """
    parameter_names = 'W',
    gradient_names = 'gW',

    def __init__(self, shape=(), init=0.25):
        self.W = numpy.full(shape, init, dtype=numpy.float32)
        self.gW = numpy.empty_like(self.W)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

        x_type, = in_types
        W = type_check.Variable(self.W, 'W')

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim >= W.shape.__len__() + 1,
            x_type.shape[1: 1 + len(self.W.shape)] == W.shape
        )

    def forward_gpu(self, x):
        cdim = self.W.size
        rdim = x[0].size // (x[0].shape[0] * cdim)
        y = cuda.empty_like(x[0])
        _fwd_mrelu()(y, x[0], self.W, cdim, rdim)

        return y,

    def backward_gpu(self, x, gy):
        ldim = x[0].shape[0]
        cdim = self.W.size
        rdim = x[0].size // (ldim * cdim)

        masked = cuda.empty_like(x[0])
        cuda.elementwise(
            '''
            float* masked, const float* x, const float* W, int cdim, int rdim
            ''', '''
            int c = i / rdim % cdim;
            if (x[i] >= W[c])
                masked[i] = -1;
            else if (x[i] <= W[c])
                masked[i] = 1;
            else
                masked[i] = 0;
            ''',
            'bwd_mrelu')(masked, x[0], gy[0], cdim, rdim)

        with cuda.using_cumisc():
            rsum = cuda.cumisc.sum(masked.reshape(ldim * cdim, rdim), axis=1)
            gW = cuda.cumisc.sum(rsum.reshape(ldim, cdim), axis=0)
            self.gW += gW.reshape(self.gW.shape)
            del rsum, gW

        gx = masked  # reuse buffer
        _fwd_mrelu()(gx, gy[0], self.W, cdim, rdim)
        return gx,
