import numpy
def sliding_window(data, size, stepsize=1, padded=False, axis=0, copy=True):
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = numpy.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = numpy.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides,
    )

    if copy:
        return strided.copy()
    else:
        return strided

def stacking_window(data, size, stepsize=1, padded=False, copy=True):
    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if numpy.where(numpy.array(data.shape) == size)[0][0] != 1:
        raise ValueError(
            "Dim should be Batch * Window * Channel"
        )

    stacked = data[0, :, :]

    tail = data[1:, -stepsize:, :]
    tail = tail.reshape(-1, tail.shape[-1])

    stacked = numpy.append(stacked, tail, axis=0)

    if copy:
        return stacked.copy()
    else:
        return stacked


if __name__ == '__main__':
    input = numpy.random.rand(51221, 8)
    slides = sliding_window(input, 700, 100, axis=0).transpose([0, 2, 1])
    restr = stacking_window(slides, 700, 100)
    print((restr==input[:len(restr)]).all())