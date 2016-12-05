from chainer import cuda
from chainer import function
from chainer.functions.activation import log_softmax
from chainer.utils import type_check

import chainer
import numpy
import six


class SoftmaxCrossEntropy(function.Function):

    """Softmax activation followed by a cross entropy loss."""

    ignore_label = -1
    normalize = True

    def __init__(self, use_cudnn=True, normalize=True, cache_score=True,
                 c=None):
        self.use_cudnn = use_cudnn
        self.normalize = normalize
        self.cache_score = cache_score
        self.c = c
        if c is not None:
            assert self.c.ndim == 1
            assert self.c.dtype.kind == 'f'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype == numpy.int32,
            t_type.ndim == x_type.ndim - 1,

            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2:] == t_type.shape[1:],
        )

    def _check_input_values(self, x, t):
        if not (((0 <= t) &
                 (t < x.shape[1])) |
                (t == self.ignore_label)).all():
            msg = ('Each label `t` need to satisfy '
                   '`0 <= t < x.shape[1] or t == %d`' % self.ignore_label)
            raise ValueError(msg)

    def forward_cpu(self, inputs):
        x, t = inputs
        if chainer.is_debug():
            self._check_input_values(x, t)

        log_y = log_softmax._log_softmax(x, self.use_cudnn)
        if self.cache_score:
            self.y = numpy.exp(log_y)
        if self.c is not None:
            if self.c.shape != x.shape:
                shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
                self.c = numpy.broadcast_to(self.c.reshape(shape), x.shape)
            log_y *= self.c
        log_yd = numpy.rollaxis(log_y, 1)
        log_yd = log_yd.reshape(len(log_yd), -1)
        log_p = log_yd[numpy.maximum(t.ravel(), 0), numpy.arange(t.size)]

        # deal with the case where the SoftmaxCrossEntropy is
        # unpickled from the old version
        if self.normalize:
            count = (t != self.ignore_label).sum()
        else:
            count = len(x)
        self._coeff = 1.0 / max(count, 1)

        y = (log_p * (t.ravel() != self.ignore_label)).sum(keepdims=True) \
            * (-self._coeff)
        return y.reshape(()),

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, t = inputs
        if chainer.is_debug():
            self._check_input_values(x, t)

        log_y = log_softmax._log_softmax(x, self.use_cudnn)
        if self.cache_score:
            self.y = cupy.exp(log_y)
        if self.c is not None:
            if self.c.ndim != x.ndim:
                shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
                self.c = cupy.broadcast_to(self.c.reshape(shape), x.shape)
            log_y *= self.c
        if self.normalize:
            coeff = cupy.maximum(1, (t != self.ignore_label).sum())
        else:
            coeff = max(1, len(t))
        self._coeff = cupy.divide(1.0, coeff, dtype=x.dtype)

        log_y = cupy.rollaxis(log_y, 1, log_y.ndim)
        ret = cuda.reduce(
            'S t, raw T log_y, int32 n_channel, raw T coeff', 'T out',
            't == -1 ? T(0) : log_y[_j * n_channel + t]',
            'a + b', 'out = a * -coeff[0]', '0', 'crossent_fwd'
        )(t, log_y.reduced_view(), log_y.shape[-1], self._coeff)
        return ret,

    def backward_cpu(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        if hasattr(self, 'y'):
            y = self.y.copy()
        else:
            y = log_softmax._log_softmax(x, self.use_cudnn)
            numpy.exp(y, out=y)
        if y.ndim == 2:
            gx = y
            gx[numpy.arange(len(t)), numpy.maximum(t, 0)] -= 1
            if self.c is not None:
                c = self.c[numpy.arange(len(t)), numpy.maximum(t, 0)]
                gx *= numpy.broadcast_to(numpy.expand_dims(c, 1), gx.shape)
            gx *= (t != self.ignore_label).reshape((len(t), 1))
        else:
            # in the case where y.ndim is higher than 2,
            # we think that a current implementation is inefficient
            # because it yields two provisional arrays for indexing.
            n_unit = t.size // len(t)
            gx = y.reshape(y.shape[0], y.shape[1], -1)
            fst_index = numpy.arange(t.size) // n_unit
            trd_index = numpy.arange(t.size) % n_unit
            gx[fst_index, numpy.maximum(t.ravel(), 0), trd_index] -= 1
            if self.c is not None:
                c = self.c.reshape(gx.shape)
                c = c[fst_index, numpy.maximum(t.ravel(), 0), trd_index]
                c = c.reshape(y.shape[0], 1, -1)
                gx *= numpy.broadcast_to(c, gx.shape)
            gx *= (t != self.ignore_label).reshape((len(t), 1, -1))
            gx = gx.reshape(y.shape)
        gx *= gloss * self._coeff
        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        cupy = cuda.cupy
        x, t = inputs
        if hasattr(self, 'y'):
            y = self.y
        else:
            y = log_softmax._log_softmax(x, self.use_cudnn)
            cupy.exp(y, out=y)
        gloss = grad_outputs[0]
        n_unit = t.size // len(t)
        coeff = gloss * self._coeff
        if self.c is None:
            gx = cuda.elementwise(
                'T y, S t, raw T coeff, S n_channel, S n_unit',
                'T gx',
                '''
                    const int c = (i / n_unit % n_channel);
                    gx = (t == -1) ? 0 : (coeff[0] * (y - (c == t)));
                ''',
                'softmax_crossent_bwd')(
                    y, cupy.expand_dims(t, 1), coeff, x.shape[1], n_unit)
        else:
            if not isinstance(self.c, cupy.ndarray):
                self.c = cuda.to_gpu(self.c, device=x.device)
            gx = cuda.elementwise(
                'T y, T w, S t, raw T coeff, S n_channel, S n_unit',
                'T gx',
                '''
                    const int c = (i / n_unit % n_channel);
                    if (t == -1) {
                        gx = 0;
                    } else if (t == c) {
                        gx = coeff[0] * (y - 1) * w;
                    } else {
                        gx = coeff[0] * y;
                    }
                ''',
                'softmax_crossent_bwd')(
                    y, self.c, cupy.expand_dims(t, 1), coeff, x.shape[1],
                    n_unit)
        return gx, None


def softmax_cross_entropy(
        x, t, use_cudnn=True, normalize=True, cache_score=True, c=None):
    """Computes cross entropy loss for pre-softmax activations.

    Args:
        x (~chainer.Variable): Variable holding a multidimensional array whose
            element indicates unnormalized log probability: the first axis of
            the variable represents the number of samples, and the second axis
            represents the number of classes. While this function computes
            a usual softmax cross entropy if the number of dimensions is equal
            to 2, it computes a cross entropy of the replicated softmax if the
            number of dimensions is greater than 2.
        t (~chainer.Variable): Variable holding an int32 vector of ground truth
            labels. If ``t[i] == -1``, corresponding ``x[i]`` is ignored.
        normalize (bool): If ``True``, this function normalizes the cross
            entropy loss across all instances. If ``False``, it only
            normalizes along a batch size.
        cache_score (bool): When it is ``True``, the function stores result
            of forward computation to use it on backward computation. It
            reduces computational cost though consumes more memory.
        c (~numpy.ndarray): An array that contains constant weights that will
            be multiplied to loss values along with the second dimension. The
            shape of this array should be ``(x.shape[1],)``.

    Returns:
        Variable: A variable holding a scalar array of the cross entropy loss.

    .. note::

       This function is differentiable only by ``x``.

    """
    return SoftmaxCrossEntropy(use_cudnn, normalize, cache_score, c)(x, t)
