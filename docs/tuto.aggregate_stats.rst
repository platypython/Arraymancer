==================================
Tutorial: Aggregate and statistics
==================================

``sum`` and ``mean`` functions are available to compute the sum and mean
of a tensor. ``sum`` and ``mean`` can also be computed along an axis
with the ``axis`` argument.

Floating-point and complex means retain the input dtype by default. Integer
and boolean means require an explicit floating dtype; they do not use integer
division. For real CPU inputs, both ``sum`` and ``mean`` accept
``dtype = float32`` or ``dtype = float64``. Each value is cast before reduction
without allocating a converted input tensor. Without an explicit dtype,
integer and boolean sums promote to wrapping int64.

.. code-block:: nim

   let values = [[1, 2, 3], [4, 5, 6]].toTensor
   doAssert values.mean(dtype = float64) == 3.5
   doAssert values.mean(axis = 0, dtype = float32) ==
            [[2.5'f32, 3.5, 4.5]].toTensor
   doAssert values.sum(dtype = float64) == 21.0
   doAssert values.sum(axis = 0, dtype = float32) ==
            [[5'f32, 7, 9]].toTensor

Axis reductions keep the reduced dimension with length one. Empty sums produce
zeros and empty floating means produce NaNs; other empty dimensions remain empty.

Generic aggregates on the whole tensor or along an axis can be computed
with ``agg`` and ``agg_inplace`` functions.
