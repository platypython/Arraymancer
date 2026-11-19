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

Complex products accept complex output dtypes for real, boolean and complex
inputs, with no conversion buffer. Arithmetic uses the requested component
width, not the widened carries used by scans. Default complex products retain
their input dtype. Packed products fold four independent vectors before
multiplying the scalar tail; columns retain independent ordered chains.
Rounding, overflow and underflow can therefore depend on grouping.

Whole and scalar-axis products multiply the ``1+0i`` identity rather than
copying a scalar's bits. For example, ``-0-0i`` becomes ``+0-0i``.
Empty products produce ones in the retained output shape. Complex-to-real
floating products discard imaginary parts with the shared warning; integer
products truncate real parts before wrapping, within the defined cast range.
Boolean products AND the truth of both components without warning.
Wider unsigned products accept only empty inputs. Numeric casts warn before
nonempty kernel rejection; invalid axes reject first, and empty inputs stay quiet.

``amin`` and ``amax`` return independent tensors without changing the input
dtype. Unlike the older ``min``/``max`` axis helpers, they remove reduced axes
by default. Pass ``keepdim = true`` to retain those dimensions as singletons:

.. code-block:: nim

  let values = [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]].toTensor
  echo values.amin(0)                 # [3.0, 1.0, 2.0], shape [3]
  echo values.amax(1, keepdim = true)  # [[3.0], [6.0]], shape [2, 1]
  echo values.amin([0, -1])           # scalar tensor containing 1.0

Omitting axes, or passing an empty axis list, reduces every dimension.
Negative axes are accepted; duplicate normalized axes reject. Scalars retain
rank zero even with ``keepdim``. NaNs propagate, while signed-zero ties follow
the backend's vector and scalar reduction grouping.

There is no identity for an empty minimum or maximum: a zero-length reduced
axis rejects, but a zero-length output caused by an unreduced axis is valid.
Nonempty complex and wider unsigned CPU inputs reject; those dtypes can still
produce empty outputs when all reduced axes have nonzero lengths.

CPU float32/float64 Variables provide the same ``amin``/``amax`` axis and
``keepdim`` overloads. First-order backward splits each seed equally among
tied extrema, including signed zeros and infinities:

.. code-block:: nim

  let ctx = newContext Tensor[float64]
  let tracked = ctx.variable(
    [[1.0, 1.0, 3.0], [3.0, 3.0, 2.0]].toTensor, requires_grad = true)
  let minima = tracked.amin(1)
  minima.backprop([2.0, -3.0].toTensor)
  echo tracked.grad

Gradients have the input shape and independent storage. Division precedes the
numeric selection mask, so unselected gradients can be negative zero or NaN
with nonfinite seeds; NaN reductions produce NaN gradients. Do not mutate the
input or saved output before backward. Metal extrema gradients and higher-order
extrema differentiation are not provided.

``argmin`` and ``argmax`` return independent ``Tensor[int64]`` indices.
Omitting ``axis`` selects the first extremum in logical row-major order;
an explicit axis returns indices within that dimension. Reduced axes disappear
unless ``keepdim = true``. Scalars remain rank zero, accepting axes 0 and -1.
Equal values, including signed zeros and infinities, select the first index.
Any NaN takes precedence over numeric values; the first NaN wins.

.. code-block:: nim

  let scores = [[3.0, 1.0, 3.0], [2.0, 5.0, 5.0]].toTensor
  echo scores.argmax(1)                  # [0, 1], dtype int64
  echo scores.argmin(0, keepdim = true)   # [[1, 0, 0]], shape [1, 3]
  echo scores.argmax()                   # scalar tensor containing 4

CPU float32/float64, signed integers and uint8 are supported without input
copies, including offset, transposed, reversed and broadcast views.
Bool and complex inputs reject before dimension validation. Wider unsigned
inputs reject except for explicit singleton dimensions on positive-rank
inputs and valid empty outputs. Empty whole reductions and zero-length reduced
axes reject. These index-only operations do not provide autograd or Metal
implementations.

``minWithIndices(values, axis)`` and ``maxWithIndices(values, axis)`` provide
the two-output CPU reductions corresponding to PyTorch's ``min(dim=...)`` and
``max(dim=...)``. The tuple contains ``values`` first and ``indices`` second.
Both tensors are independent outputs, with the input dtype for values and
int64 indices. Pass ``keepdim = true`` to retain the reduced dimension.
The selected first element supplies the value bits, including signed zeros
and NaN payloads.

.. code-block:: nim

  let (maxima, positions) = scores.maxWithIndices(1)
  echo maxima     # [3.0, 5.0]
  echo positions  # [0, 1]

Paired reductions support bool. Wider unsigned dtypes accept scalars and
valid empty outputs, but reject positive-rank singleton dimensions.
Axis and empty-reduced-axis errors take precedence over complex dtype rejection;
complex inputs cannot produce paired outputs, including valid empty outputs.
CPU float32/float64 ``Variable`` overloads return differentiable values and a
plain, nondifferentiable int64 index tensor. Backward scatters each seed to
the first selected index, rather than splitting ties as ``amin``/``amax`` do.
Unselected gradients remain positive zero even with infinite, NaN or negative-zero
seeds. Gradients own dense input-shaped storage for signed and broadcast views.
Do not mutate the returned indices before backward. Retained graphs, explicit
strided seeds and ``no_grad_mode`` follow the usual autograd contracts.
Higher-order gradients, Metal, output-buffer and low-precision overloads remain
unsupported.

Use ``aminmax`` to compute both extrema in one pass, without an input copy:

.. code-block:: nim

  let (minimum, maximum) = scores.aminmax(1)
  let whole = scores.aminmax(keepdim = true)

Both outputs own independent storage and preserve the input dtype. The axis
overload preserves the first equal signed zero and first NaN payload; whole
floating reductions follow the native vector fold, so zero ties and the selected
NaN payload can differ. Scalar inputs retain their bits and stay scalar.
Bool is supported. Wider unsigned inputs support explicit scalar axes only;
whole reductions and positive-rank inputs reject, even with valid empty outputs.
Complex inputs reject before axis or empty-input checks. No autograd, Metal,
output-buffer or low-precision overloads are provided.

Generic aggregates on the whole tensor or along an axis can be computed
with ``agg`` and ``agg_inplace`` functions.

Logarithmic reductions
---------------------

``logsumexp`` computes a stable ``log(sum(exp(x)))`` without materializing
shifted or exponential tensors:

.. code-block:: nim

  let normalizer = scores.logsumexp(1)  # Remove axis 1.
  let total = scores.logsumexp()       # Return a scalar tensor.
  let retained = scores.logsumexp([0, 1], keepdim = true)

Integer and bool inputs promote numerically to float32, including high unsigned
values. Float32/float64 and complex64/complex128 preserve their dtype. Complex
results use the principal logarithm: even a scalar's imaginary phase can wrap.
Signed, offset and broadcast views are read directly; results own independent
storage. Reduced empty axes yield negative infinity, while retained empty axes
produce empty outputs. Duplicate or invalid axes reject.

An explicit empty axis list reduces all axes. Matching the pinned PyTorch
broadcast behavior, nonempty positive-rank inputs require ``keepdim = true``
for that spelling; use the no-axis overload for a dropped whole result.

CPU float32/64 and complex64/128 variables support the same overloads with
first-order seeded gradients:

.. code-block:: nim

  let normalizer = source.logsumexp(1)
  normalizer.backprop(seed)  # Seed shape must exactly match the output.

Backward computes ``seed * conj(exp(input - rounded_forward_output))``.
Rounding the forward result matters for large real inputs; complex derivatives
conjugate the exponential, including its phase. Complex subtraction preserves
PyTorch's default complex-alpha IEEE effects for infinite components.
Gradients own dense input-shaped
storage; signed, offset and broadcast inputs and seeds do not require converted
tensor intermediates. Saved input and forward-output storage must not be mutated
before backward. Retained graphs and no-grad mode follow the usual contracts.
Higher-order gradients, integer/bool differentiation, Metal, output-buffer and
low-precision overloads remain unsupported.

