from tensorflow.python.ops.init_ops import Initializer, _assert_float_dtype
from tensorflow.python.ops import linalg_ops, array_ops
from tensorflow.python.framework import dtypes


class Identity(Initializer):
    """Initializer that generates the identity matrix.
    Only use for 2D matrices.
    Args:
      gain: Multiplicative factor to apply to the identity matrix.
      dtype: The type of the output.
    """

    def __init__(self, gain=1.0, dtype=dtypes.float32):
        self.gain = gain
        self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))

    def __call__(self, shape, dtype=None, partition_info=None):
        full_shape = shape if partition_info is None else partition_info.full_shape

        if dtype is None:
            dtype = self.dtype
        if len(full_shape) > 2:
            batch_shape = full_shape[:-2]
            rows, cols = full_shape[-2:]
            initializer = linalg_ops.eye(rows, cols, batch_shape=batch_shape, dtype=dtype)
        elif len(full_shape) == 2:
            initializer = linalg_ops.eye(*full_shape, dtype=dtype)
        else:
            raise ValueError(
                "Identity matrix initializer can only be used for shapes with 2 dimensions or more.")
        if partition_info is not None:
            initializer = array_ops.slice(initializer, partition_info.var_offset,
                                          shape)
        return self.gain * initializer

    def get_config(self):
        return {"gain": self.gain, "dtype": self.dtype.name}
