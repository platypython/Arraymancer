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

By default, complex ``cumsum`` and ``cumprod`` retain the input dtype and
accumulate in complex128 on CPU, rounding both components only when storing a prefix.
This keeps complex64 prefixes accurate after cancellation or rounded output
overflow and underflow. Vector scans begin with the additive or multiplicative
identity; scalar scans copy the value, including signed-zero components.
Views, negative axes and reverse scans follow the same rules as real inputs.

Pass ``dtype = Complex[float32]`` or ``dtype = Complex[float64]`` to scan real,
boolean or complex inputs into a complex output. Each input component is cast
to the requested width before the complex128 carry; no converted input tensor
is allocated. Real inputs gain a positive-zero imaginary component.

Complex inputs also accept real floating and integer output dtypes. Numeric
casts discard the imaginary component and warn once per process, including on
nonempty unsupported unsigned scans. Invalid axes and empty tensors do not
warn. Integer casts truncate the real component, which must fit the output
dtype before wrapping arithmetic. Boolean scalar copies test both components
and never warn. The scalar/empty dtype exceptions above apply unchanged.

Complex means accept the same ``Complex[float32]`` and ``Complex[float64]``
output dtypes for real, boolean and complex inputs. Unlike scans, they sum in
the requested component width and use complex reciprocal division. Default
complex means retain their input dtype. Packed reductions preserve CPU cascade
and column-tail grouping, which can affect cancellation-sensitive results.

For a complex input, ``mean(dtype = float32)`` or ``mean(dtype = float64)``
discards imaginary components before real arithmetic and shares the
once-per-process cast warning with scans. Integer and boolean mean output
dtypes reject before axis validation or warnings, even for scalars and empties.
Empty complex means have two NaN components; scalar means add the zero identity
before dividing rather than copying signed-zero components.

Complex sums accept ``dtype = Complex[float32]`` or ``Complex[float64]`` for
real, boolean and complex inputs, casting before component-width accumulation.
Default complex sums retain their input dtype and execute the zero identity,
including for scalars. Empty sums produce zeros in the retained output shape.

Complex inputs also support real floating, integer and boolean sum outputs.
Numeric outputs discard imaginary components with the same cast warning;
integer casts truncate before wrapping addition and must be representable in
the output dtype. Boolean sums OR the truth of both components without warning.
Wider unsigned sums accept only empty inputs, rejecting nonempty scalars too.
Those nonempty numeric casts warn before kernel rejection; invalid axes reject
before warnings, while empty inputs remain quiet.

Generic aggregates on the whole tensor or along an axis can be computed
with ``agg`` and ``agg_inplace`` functions.
