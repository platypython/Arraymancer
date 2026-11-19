# Copyright 2026 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Exercises unrounded scan carries, phase continuity, views and dtype precedence.
import ../../src/arraymancer
import std/[complex, math, unittest]

proc closeValue[F: SomeFloat](actual, expected: F) =
  let tolerance = when F is float32: 1e-6 else: 1e-12
  if expected != expected: check actual != actual
  elif abs(expected) == F(Inf): check actual == expected
  else: check abs(float64(actual) - float64(expected)) <= tolerance * max(1.0, abs(float64(expected)))

proc checkReal[F: SomeFloat]() =
  ## Rounding every carry to float32 drifts far beyond this long-prefix tolerance.
  let zeros = newTensor[F](30000)
  let prefixes = zeros.logcumsumexp()
  closeValue(prefixes[29999], F(ln(30000.0)))
  let large = [F(1000), F(999), F(-1000)].toTensor.logcumsumexp()
  closeValue(large[0], F(1000))
  closeValue(large[1], F(1000.3132616875182))
  closeValue(large[2], F(1000.3132616875182))
  let signs = [F(-0.0), F(0), F(-0.0)].toTensor.logcumsumexp()
  check signs[0].classify == fcZero
  closeValue(signs[1], F(ln(2.0)))
  closeValue(signs[2], F(ln(3.0)))
  let scalar = newTensorWith[F]([], F(-0.0)).logcumsumexp(-1)
  check scalar.shape.len == 0
  check scalar.atContiguousIndex(0).classify == fcNegZero
  let nonfinite = [F(-Inf), F(-Inf), F(1), F(Inf), F(NaN)].toTensor.logcumsumexp()
  check nonfinite[0] == F(-Inf)
  check nonfinite[1] == F(-Inf)
  check nonfinite[2] == F(1)
  check nonfinite[3] == F(Inf)
  check nonfinite[4].classify == fcNan

proc checkComplex[F: SomeFloat]() =
  ## Unlike logsumexp, the complex scan carries phase rather than wrapping it.
  type C = Complex[F]
  let values = [C(re: F(2), im: F(7)), C(re: F(1), im: F(-8)), C(re: F(-2), im: F(0.5))].toTensor
  let output = values.logcumsumexp()
  let reals = [2.0, 1.7245130162187947, 1.7482277523299061]
  let imaginaries = [7.0, 6.6794339857634455, 6.681873930610002]
  for index in 0 ..< 3:
    closeValue(output[index].re, F(reals[index]))
    closeValue(output[index].im, F(imaginaries[index]))
  let special = C(re: F(-Inf), im: F(Inf))
  let scalar = newTensorWith[C]([], special).logcumsumexp(-1)
  check scalar.atContiguousIndex(0) == special
  let nonfinite = [special, C(re: F(1), im: F(0)), C(re: F(Inf), im: F(-0.0))].toTensor.logcumsumexp()
  check nonfinite[0].re == F(-Inf)
  check nonfinite[0].im.classify == fcZero
  check nonfinite[1] == C(re: F(1), im: F(0))
  check nonfinite[2].re == F(Inf)
  check nonfinite[2].im.classify == fcZero
  let nan = [C(re: F(1), im: F(0)), C(re: F(0), im: F(NaN)), C(re: F(2), im: F(0))].toTensor.logcumsumexp()
  for index in 1 ..< 3:
    check nan[index].re.classify == fcNan
    check nan[index].im.classify == fcNan

proc checkViews[T: SomeFloat or Complex]() =
  ## Independent packed and logical walks must retain the same unrounded carries.
  var storage = newTensorUninit[T](3, 4, 33)
  for index in 0 ..< storage.size:
    when T is Complex:
      type F = typeof(default(T).re)
      storage.unsafe_raw_offset[index] = T(re: F(index mod 7) / F(8), im: F(index mod 11) / F(32))
    else: storage.unsafe_raw_offset[index] = T(index mod 7) / T(8)
  let views = [storage[^1..0|-1, _, ^1..0|-1], storage[1..2, _, 1..32],
               storage[0..0, _, _].broadcast(3, 4, 33), storage.transpose]
  for view in views:
    for axis in 0 ..< view.rank:
      var actual = view.logcumsumexp(axis - view.rank)
      let expected = view.clone.logcumsumexp(axis)
      check actual.shape == view.shape
      check actual.is_C_contiguous
      for index in 0 ..< actual.size:
        when T is Complex:
          closeValue(actual.atContiguousIndex(index).re, expected.atContiguousIndex(index).re)
          closeValue(actual.atContiguousIndex(index).im, expected.atContiguousIndex(index).im)
        else: closeValue(actual.atContiguousIndex(index), expected.atContiguousIndex(index))
      let first = view.atContiguousIndex(0)
      actual.unsafe_raw_offset[0] = default(T)
      check view.atContiguousIndex(0) == first
  let empty = newTensor[T](2, 0, 3)
  for axis in 0 ..< 3:
    let output = empty.logcumsumexp(axis)
    check output.shape == [2, 0, 3]
    check output.size == 0
  expect IndexDefect: discard empty.logcumsumexp(3)

proc checkRejected[T]() =
  ## Dtype rejection follows dimension validation even for scalar and empty inputs.
  for input in [newTensor[T]([]), newTensor[T](2, 0)]:
    expect ValueError: discard input.logcumsumexp(0)
    expect IndexDefect: discard input.logcumsumexp(3)

suite "CPU logarithmic cumulative scans":
  test "Float32 wide carries, scalar bits and nonfinite prefixes": checkReal[float32]()
  test "Float64 wide carries, scalar bits and nonfinite prefixes": checkReal[float64]()
  test "Complex64 continuous phases and infectious NaNs": checkComplex[float32]()
  test "Complex128 continuous phases and infectious NaNs": checkComplex[float64]()
  test "Real signed, offset and broadcast scans":
    checkViews[float32]()
    checkViews[float64]()
  test "Complex signed, offset and broadcast scans":
    checkViews[Complex[float32]]()
    checkViews[Complex[float64]]()
  test "Integer and boolean dispatch follows dimension checks":
    checkRejected[bool]()
    checkRejected[int8]()
    checkRejected[int16]()
    checkRejected[int32]()
    checkRejected[int64]()
    checkRejected[uint8]()
    checkRejected[uint16]()
    checkRejected[uint32]()
    checkRejected[uint64]()
