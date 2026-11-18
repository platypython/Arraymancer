==================================
Tutorial: Aggregate and statistics
==================================

``sum``, ``product`` and ``mean`` reduce a whole tensor or a single axis
with the ``axis`` argument.

Floating-point and complex means retain the input dtype by default. Integer
and boolean means require an explicit floating dtype; they do not use integer
division. For real CPU inputs, ``sum``, ``product`` and ``mean`` accept
``dtype = float32`` or ``dtype = float64``. Each value is cast before reduction
without allocating a converted input tensor. Without an explicit dtype,
integer and boolean sums and products promote to wrapping int64.

.. code-block:: nim

   let values = [[1, 2, 3], [4, 5, 6]].toTensor
   doAssert values.mean(dtype = float64) == 3.5
   doAssert values.mean(axis = 0, dtype = float32) ==
            [[2.5'f32, 3.5, 4.5]].toTensor
   doAssert values.sum(dtype = float64) == 21.0
   doAssert values.sum(axis = 0, dtype = float32) ==
            [[5'f32, 7, 9]].toTensor
   doAssert values.product(dtype = float64) == 720.0
   doAssert values.product(axis = 0, dtype = float32) ==
            [[4'f32, 10, 18]].toTensor

Axis reductions keep the reduced dimension with length one. Empty sums produce
zeros, empty products produce ones, and empty floating means produce NaNs;
other empty dimensions remain empty.

Floating products multiply in the requested dtype, including intermediate
overflow and underflow. Packed reductions group terms and can round differently
from scalar products.

``cumsum`` and ``cumprod`` also accept explicit floating dtypes for real CPU
inputs. They retain the input shape, cast each value before accumulating, and
round each prefix from an ordered float64 carry. An overflowed float32 output
does not replace that carry. Negative axes, strided views and ``reverse = true``
are supported; the output is independent of its input.

.. code-block:: nim

   let precise = [16777217'i32, -16777216'i32].toTensor
   doAssert precise.cumsum(dtype = float32) == [16777216'f32, 0].toTensor
   doAssert precise.cumsum(dtype = float64) == [16777217.0, 1].toTensor

Integer output dtypes ``uint8``, ``int8``, ``int16``, ``int32`` and ``int64``
work with ``sum``, ``product``, ``cumsum`` and ``cumprod``. Arithmetic wraps
modulo the output bit width. Boolean sums are logical OR; boolean products
are logical AND.

.. code-block:: nim

   let signed = [129'i64, -130, 3].toTensor
   doAssert signed.sum(dtype = int8) == 2'i8
   doAssert signed.cumsum(dtype = uint8) == [129'u8, 255, 2].toTensor
   doAssert signed.cumprod(dtype = int8) == [-127'i8, 126, 122].toTensor

Floating inputs truncate before integer arithmetic. The truncated value must
fit the requested integer dtype; like PyTorch, out-of-range and nonfinite
float-to-integer casts have no portable result.

Empty sums, products and scans also accept ``uint16``, ``uint32`` and
``uint64``. Nonempty reductions in these dtypes raise ``ValueError``.
Scans with these dtypes or ``bool`` accept scalars and empty tensors only;
even a one-element vector raises ``ValueError``. All means reject an integer
or boolean output dtype, including on empty inputs.

Generic aggregates on the whole tensor or along an axis can be computed
with ``agg`` and ``agg_inplace`` functions.
