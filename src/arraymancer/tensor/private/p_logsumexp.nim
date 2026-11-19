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

## Provides cast-on-read logarithmic reductions without intermediate tensors.
import std/[complex, math]
import ../data_structure, ../init_cpu
import ./p_extrema, ./p_accelerate

when defined(macosx) and defined(arm64) and defined(clang):
  # Outlined cast/load fragments break the independent column chains.
  {.localPassC: "-mno-outline".}

template logsumexpDtype*(D: typedesc): typedesc =
  when D is SomeInteger or D is bool: float32
  else: D

template logsumexpReal(D: typedesc): typedesc =
  when D is Complex: typeof(default(D).re)
  else: logsumexpDtype(D)

template logsumexpAccumulator(D: typedesc): typedesc =
  when D is Complex: Complex[float64]
  else: float64

proc realLogsumexpValue[T](value: T): logsumexpReal(T) {.inline.} =
  when T is Complex: value.re
  elif T is bool: float32(ord(value))
  elif T is SomeInteger: float32(value)
  else: value

when defined(macosx):
  proc sineCosine(value: float32; sine, cosine: ptr float32)
    {.importc: "__sincosf", header: "<math.h>".}
  proc sineCosine(value: float64; sine, cosine: ptr float64)
    {.importc: "__sincos", header: "<math.h>".}

proc exponentialComplex*[T: SomeFloat](value: Complex[T]): Complex[T] {.inline.} =
  ## Keep IEEE complex branches that a direct exp*cos/sin formula loses.
  if value.im == T(0):
    return Complex[T](re: exp(value.re), im: value.im)
  if value.re == T(-Inf) and
      (value.im != value.im or abs(value.im) == T(Inf)):
    return Complex[T](re: T(0), im: T(0))
  if value.re == T(Inf) and
      (value.im != value.im or abs(value.im) == T(Inf)):
    return Complex[T](re: T(Inf), im: T(NaN))
  when defined(macosx):
    var sine, cosine: T
    sineCosine(value.im, addr sine, addr cosine)
    let magnitude = exp(value.re)
    Complex[T](re: magnitude * cosine, im: magnitude * sine)
  else:
    complex.exp(value)

proc complexExponentialParts*[F: SomeFloat](real, imaginary, magnitude, sine,
    cosine: F): Complex[F] {.inline.} =
  ## Vector transcendental calls still need the scalar IEEE exceptional branches.
  if imaginary == F(0): return Complex[F](re: magnitude, im: imaginary)
  if imaginary != imaginary or abs(imaginary) == F(Inf):
    if real == F(-Inf): return Complex[F](re: F(0), im: F(0))
    if real == F(Inf): return Complex[F](re: F(Inf), im: F(NaN))
  Complex[F](re: F(magnitude * cosine), im: F(magnitude * sine))

proc accumulateLogsumexp[Input, A, F](total: var A; value: Input;
    maximum: F) {.inline.} =
  when Input is Complex:
    let shifted = Complex[F](re: F(value.re - maximum), im: value.im)
    let term = exponentialComplex(shifted)
    total.re += float64(term.re)
    total.im += float64(term.im)
  else:
    let shifted = F(realLogsumexpValue(value) - maximum)
    total += float64(exp(shifted))

proc finishLogsumexp[Input, A, F](total: A; maximum: F;
    dtype: typedesc[Input]): auto {.inline.} =
  when Input is Complex:
    let rounded = Complex[F](re: F(total.re), im: F(total.im))
    let value = complex.ln(rounded)
    return Complex[F](re: F(value.re + maximum), im: F(value.im + F(0)))
  else:
    return F(ln(F(total)) + maximum)

proc logsumexpBooleanCount(trueCount, count: uint): float32 {.inline.} =
  if trueCount == 0: return ln(float32(count))
  let total = float64(trueCount) +
    float64(count - trueCount) * float64(exp(-1'f32))
  float32(ln(float32(total)) + 1'f32)

proc logsumexpSpan[T; I: SomeInteger](source: ptr UncheckedArray[T]; base: I;
    count: uint; stride: I): logsumexpDtype(T) =
  ## Keep input casting and exponential tiles bounded independently of shape.
  when T is bool:
    var trueCount = 0'u
    for index in 0'u ..< count:
      trueCount += uint(ord(source[base + I(index) * stride]))
    return logsumexpBooleanCount(trueCount, count)
  else:
    type F = logsumexpReal(T)
    var maximum: F
    when T is Complex:
      maximum = F(-Inf)
      for index in 0'u ..< count:
        let value = realLogsumexpValue(source[base + I(index) * stride])
        if maximum == maximum and not (value <= maximum): maximum = value
    else:
      maximum = realLogsumexpValue(extremaSpan(source, base, count, stride, true))
    if abs(maximum) == F(Inf): maximum = F(0)
    var total: logsumexpAccumulator(T)
    when defined(macosx):
      when T is Complex:
        if count < 8:
          for index in 0'u ..< count:
            accumulateLogsumexp(total, source[base + I(index) * stride], maximum)
          return finishLogsumexp(total, maximum, T)
      const Tile = 256'u
      var shifted {.noinit.}, terms {.noinit.}: array[Tile, F]
      when T is Complex:
        var phases {.noinit.}, sines {.noinit.}, cosines {.noinit.}: array[Tile, F]
      var partials: array[4, logsumexpAccumulator(T)]
      var start = 0'u
      while start < count:
        let length = min(Tile, count - start)
        for index in 0'u ..< length:
          shifted[index] = F(realLogsumexpValue(
            source[base + I(start + index) * stride]) - maximum)
          when T is Complex:
            phases[index] = source[base + I(start + index) * stride].im
        var lengthC = cint(length)
        when F is float32: vvexpf(addr terms[0], addr shifted[0], addr lengthC)
        else: vvexp(addr terms[0], addr shifted[0], addr lengthC)
        when T is Complex:
          when F is float32:
            vvsincosf(addr sines[0], addr cosines[0], addr phases[0], addr lengthC)
          else:
            vvsincos(addr sines[0], addr cosines[0], addr phases[0], addr lengthC)
          for index in 0'u ..< length:
            let term = complexExponentialParts(shifted[index], phases[index],
              terms[index], sines[index], cosines[index])
            partials[index and 3].re += float64(term.re)
            partials[index and 3].im += float64(term.im)
        else:
          var index = 0'u
          while index + 4 <= length:
            partials[0] += float64(terms[index])
            partials[1] += float64(terms[index + 1])
            partials[2] += float64(terms[index + 2])
            partials[3] += float64(terms[index + 3])
            index += 4
          while index < length:
            partials[0] += float64(terms[index])
            inc index
        start += length
      total = (partials[0] + partials[1]) + (partials[2] + partials[3])
    else:
      for index in 0'u ..< count:
        accumulateLogsumexp(total, source[base + I(index) * stride], maximum)
    finishLogsumexp(total, maximum, T)

proc logsumexpStrided[T](source: ptr UncheckedArray[T]; base: int;
    shape, strides: Metadata; count: int): logsumexpDtype(T) =
  ## Restart the same logical reduction walk for the stable exponential pass.
  type F = logsumexpReal(T)
  var maximum = F(-Inf)
  var offset = base
  var coordinates: Metadata
  coordinates.len = shape.len
  template advance() =
    for dimension in countdown(shape.len - 1, 0):
      inc coordinates[dimension]
      offset += strides[dimension]
      if coordinates[dimension] < shape[dimension]: break
      coordinates[dimension] = 0
      offset -= shape[dimension] * strides[dimension]
  for index in 0 ..< count:
    let value = realLogsumexpValue(source[offset])
    if maximum == maximum and not (value <= maximum): maximum = value
    advance()
  if abs(maximum) == F(Inf): maximum = F(0)
  coordinates = default(Metadata)
  coordinates.len = shape.len
  offset = base
  var total: logsumexpAccumulator(T)
  for index in 0 ..< count:
    accumulateLogsumexp(total, source[offset], maximum)
    advance()
  finishLogsumexp(total, maximum, T)

when defined(macosx):
  proc logsumexpColumns[T: SomeNumber or bool or Complex](destination:
      ptr UncheckedArray[logsumexpDtype(T)]; source: ptr UncheckedArray[T];
      outer, length, inner: uint) =
    ## Walk adjacent columns together instead of rereading one cache line per row.
    const Columns = 32'u
    const Tile = 256'u
    type F = logsumexpReal(T)
    when T isnot bool:
      var shifted {.noinit.}, terms {.noinit.}: array[Tile, F]
      when T is Complex:
        var phases {.noinit.}, sines {.noinit.}, cosines {.noinit.}: array[Tile, F]
    for blockIndex in 0'u ..< outer:
      let base = blockIndex * length * inner
      var column = 0'u
      while column < inner:
        let width = min(Columns, inner - column)
        when T is bool:
          var counts: array[Columns, uint]
          for row in 0'u ..< length:
            let position = base + row * inner + column
            for lane in 0'u ..< width:
              counts[lane] += uint(ord(source[position + lane]))
          for lane in 0'u ..< width:
            destination[blockIndex * inner + column + lane] =
              logsumexpBooleanCount(counts[lane], length)
        else:
          var maxima {.noinit.}: array[Columns, F]
          var totals: array[Columns, logsumexpAccumulator(T)]
          when T is SomeInteger:
            # Numeric casts are monotone; cast each column maximum only once.
            var integerMaxima {.noinit.}: array[Columns, T]
            for lane in 0'u ..< width:
              integerMaxima[lane] = source[base + column + lane]
            for row in 1'u ..< length:
              let position = base + row * inner + column
              for lane in 0'u ..< width:
                if source[position + lane] > integerMaxima[lane]:
                  integerMaxima[lane] = source[position + lane]
            for lane in 0'u ..< width:
              maxima[lane] = realLogsumexpValue(integerMaxima[lane])
          else:
            for lane in 0'u ..< width:
              maxima[lane] = realLogsumexpValue(source[base + column + lane])
            for row in 1'u ..< length:
              let position = base + row * inner + column
              for lane in 0'u ..< width:
                let value = realLogsumexpValue(source[position + lane])
                if maxima[lane] == maxima[lane] and not (value <= maxima[lane]):
                  maxima[lane] = value
            for lane in 0'u ..< width:
              if abs(maxima[lane]) == F(Inf): maxima[lane] = F(0)
          let rowsPerTile = Tile div width
          var firstRow = 0'u
          while firstRow < length:
            let rows = min(rowsPerTile, length - firstRow)
            for row in 0'u ..< rows:
              let position = base + (firstRow + row) * inner + column
              for lane in 0'u ..< width:
                shifted[row * width + lane] =
                  F(realLogsumexpValue(source[position + lane]) - maxima[lane])
                when T is Complex:
                  phases[row * width + lane] = source[position + lane].im
            var countC = cint(rows * width)
            when F is float32: vvexpf(addr terms[0], addr shifted[0], addr countC)
            else: vvexp(addr terms[0], addr shifted[0], addr countC)
            when T is Complex:
              when F is float32:
                vvsincosf(addr sines[0], addr cosines[0], addr phases[0], addr countC)
              else:
                vvsincos(addr sines[0], addr cosines[0], addr phases[0], addr countC)
            for row in 0'u ..< rows:
              for lane in 0'u ..< width:
                when T is Complex:
                  let index = row * width + lane
                  let term = complexExponentialParts(shifted[index], phases[index],
                    terms[index], sines[index], cosines[index])
                  totals[lane].re += float64(term.re)
                  totals[lane].im += float64(term.im)
                else:
                  totals[lane] += float64(terms[row * width + lane])
            firstRow += rows
          for lane in 0'u ..< width:
            destination[blockIndex * inner + column + lane] =
              finishLogsumexp(totals[lane], maxima[lane], T)
        column += width

proc reduceLogsumexp*[T: SomeNumber or bool or Complex](arg: Tensor[T];
    axes: openArray[int]; whole: static[bool]; keepdim: bool):
    Tensor[logsumexpDtype(T)] =
  ## Plan dimensions on the stack and allocate only the final result.
  type U = logsumexpDtype(T)
  var selected: Metadata
  selected.len = max(arg.rank, 1)
  when whole:
    for dimension in 0 ..< selected.len: selected[dimension] = 1
  else:
    if axes.len == 0:
      if arg.size != 0 and arg.rank != 0 and not keepdim:
        raise newException(ValueError, "logsumexp empty dimensions cannot broadcast a dropped result")
      for dimension in 0 ..< selected.len: selected[dimension] = 1
    else:
      for axis in axes:
        let dimension = normalizedExtremaAxis(arg.rank, axis)
        if selected[dimension] != 0:
          raise newException(ValueError, "logsumexp duplicate reduction axis")
        selected[dimension] = 1
  var outputShape, reducedShape, reducedStrides: Metadata
  var outputCount = 1
  var reducedCount = 1
  var first = arg.rank
  var last = -1
  for dimension in 0 ..< arg.rank:
    if selected[dimension] != 0:
      first = min(first, dimension)
      last = dimension
      reducedCount *= arg.shape[dimension]
      inc reducedShape.len
      inc reducedStrides.len
      reducedShape[reducedShape.len - 1] = arg.shape[dimension]
      reducedStrides[reducedStrides.len - 1] = arg.strides[dimension]
      if keepdim:
        inc outputShape.len
        outputShape[outputShape.len - 1] = 1
    else:
      inc outputShape.len
      outputShape[outputShape.len - 1] = arg.shape[dimension]
      outputCount *= arg.shape[dimension]
  result = newTensorUninit[U](outputShape)
  if outputCount == 0: return
  let destination = cast[ptr UncheckedArray[U]](result.get_offset_ptr)
  if reducedCount == 0:
    when U is Complex:
      type F = logsumexpReal(T)
      for index in 0 ..< outputCount:
        destination[index] = U(re: F(-Inf), im: F(0))
    else:
      for index in 0 ..< outputCount: destination[index] = U(-Inf)
    return
  let source = cast[ptr UncheckedArray[T]](arg.get_offset_ptr)
  if arg.rank == 0:
    destination[0] = logsumexpSpan(source, 0'u, 1'u, 1'u)
    return
  if reducedShape.len == last - first + 1 and arg.is_C_contiguous:
    var outer = 1'u
    var inner = 1'u
    for dimension in 0 ..< first: outer *= uint(arg.shape[dimension])
    for dimension in last + 1 ..< arg.rank: inner *= uint(arg.shape[dimension])
    when defined(macosx):
      if inner > 1:
        logsumexpColumns[T](destination, source, outer, uint(reducedCount), inner)
        return
    for blockIndex in 0'u ..< outer:
      let base = blockIndex * uint(reducedCount) * inner
      for column in 0'u ..< inner:
        destination[blockIndex * inner + column] =
          logsumexpSpan(source, base + column, uint(reducedCount), inner)
    return
  var outputCoordinates: Metadata
  outputCoordinates.len = arg.rank
  var offset = 0
  for outputIndex in 0 ..< outputCount:
    if reducedShape.len == 1:
      destination[outputIndex] = logsumexpSpan(source, offset,
        uint(reducedCount), reducedStrides[0])
    else:
      destination[outputIndex] = logsumexpStrided(source, offset,
        reducedShape, reducedStrides, reducedCount)
    for dimension in countdown(arg.rank - 1, 0):
      if selected[dimension] == 0:
        inc outputCoordinates[dimension]
        offset += arg.strides[dimension]
        if outputCoordinates[dimension] < arg.shape[dimension]: break
        outputCoordinates[dimension] = 0
        offset -= arg.shape[dimension] * arg.strides[dimension]
