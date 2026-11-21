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

## Checked, staged destination planning for cumulative extrema out overloads.
import std/complex
import ../data_structure
import ../../laser/compiler_optim_hints

type
  CumulativeExtremaOutputs*[T] = object
    values*: Tensor[T]
    indices*: Tensor[int64]
    valuesExactInput*, indicesExactInput*: bool

  LayoutProof = enum
    lpDisjoint, lpOverlap, lpUnproven

  SearchResult = enum
    srNone, srFound, srBudget, srUnproven

  ByteSpan = object
    first, past: uint
    isEmpty: bool

  OutputLayout[T] = object
    shape, strides: Metadata
    offset: int
    storageBase: pointer
    capacityElems, logicalNumel: int
    lowestElem, highestElem: int
    span: ByteSpan

  OutputPlan[T] = object
    layout: OutputLayout[T]
    storage: CpuStorage[T]
    replacementElems: int
    needsReplacement: bool

const MaxAliasVariables = 2 * LASER_MAXRANK

proc checkedAddInt(a, b: int; value: var int): bool {.inline.} =
  if b > 0 and a > high(int) - b:
    return false
  if b < 0 and a < low(int) - b:
    return false
  value = a + b
  true

proc checkedSubInt(a, b: int; value: var int): bool {.inline.} =
  if b > 0 and a < low(int) + b:
    return false
  if b < 0 and a > high(int) + b:
    return false
  value = a - b
  true

proc checkedMulInt(a, b: int; value: var int): bool {.inline.} =
  if a == 0 or b == 0:
    value = 0
    return true
  if a == -1:
    if b == low(int):
      return false
    value = -b
    return true
  if b == -1:
    if a == low(int):
      return false
    value = -a
    return true
  if a > 0:
    if b > 0:
      if a > high(int) div b:
        return false
    elif b < low(int) div a:
      return false
  elif b > 0:
    if a < low(int) div b:
      return false
  elif a < high(int) div b:
    return false
  value = a * b
  true

proc checkedNegInt(a: int; value: var int): bool {.inline.} =
  if a == low(int):
    return false
  value = -a
  true

proc checkedAddUInt(a, b: uint; value: var uint): bool {.inline.} =
  if a > high(uint) - b:
    return false
  value = a + b
  true

proc checkedMulUInt(a, b: uint; value: var uint): bool {.inline.} =
  if a == 0 or b == 0:
    value = 0
    return true
  if a > high(uint) div b:
    return false
  value = a * b
  true

proc checkedIntBytes(value, elementSize: int; bytes: var int): bool {.inline.} =
  checkedMulInt(value, elementSize, bytes)

proc validMetadataLength(metadata: Metadata): bool {.inline.} =
  metadata.len >= 0 and metadata.len <= LASER_MAXRANK

proc checkedCumulativeExtremaNumel(shape: Metadata): int =
  if not validMetadataLength(shape):
    raise newException(ValueError, "cumulative extrema shape rank is invalid")
  var hasZero = false
  for axis in 0 ..< shape.len:
    if shape[axis] < 0:
      raise newException(ValueError, "cumulative extrema shape has a negative extent")
    if shape[axis] == 0:
      hasZero = true
  if hasZero:
    return 0
  result = 1
  for axis in 0 ..< shape.len:
    let extent = shape[axis]
    if result > high(int) div extent:
      raise newException(ValueError, "cumulative extrema element count overflows int")
    result *= extent

## Builds canonical C strides, treating empty extents as one and omitting the
## unused final product so representable empty layouts remain representable.
proc cumulativeExtremaResizeStrides(shape: Metadata): Metadata =
  if not validMetadataLength(shape):
    raise newException(ValueError, "cumulative extrema shape rank is invalid")
  result.len = shape.len
  var running = 1
  var axis = shape.len
  while axis > 0:
    dec axis
    let extent = shape[axis]
    if extent < 0:
      raise newException(ValueError, "cumulative extrema shape has a negative extent")
    result[axis] = running
    if axis > 0:
      let strideExtent = max(extent, 1)
      if running > high(int) div strideExtent:
        raise newException(ValueError, "cumulative extrema resize stride overflows int")
      running *= strideExtent

proc checkedReachableBounds(shape, strides: Metadata; offset: int;
    lowest, highest: var int): bool =
  if not validMetadataLength(shape) or not validMetadataLength(strides) or
      shape.len != strides.len or offset < 0:
    return false
  lowest = offset
  highest = offset
  for axis in 0 ..< shape.len:
    let extent = shape[axis]
    if extent < 0:
      return false
    if extent <= 1:
      continue
    var term: int
    if not checkedMulInt(extent - 1, strides[axis], term):
      return false
    if term < 0:
      if not checkedAddInt(lowest, term, lowest):
        return false
    else:
      if not checkedAddInt(highest, term, highest):
        return false
  lowest >= 0 and highest >= lowest

proc validateCapacity(capacityElems: int; highest: int;
    requireKnown: bool) =
  if capacityElems < -1:
    raise newException(ValueError, "cumulative extrema storage capacity metadata is invalid")
  if requireKnown and capacityElems < 0:
    raise newException(ValueError, "cumulative extrema output storage capacity is unknown")
  if capacityElems >= 0 and highest >= capacityElems:
    raise newException(ValueError, "cumulative extrema output exceeds storage capacity")

proc byteSpanFromBounds[T](logicalNumel, lowest, highest: int;
    storageBase: pointer): ByteSpan =
  if logicalNumel == 0:
    result.isEmpty = true
    return
  if storageBase.isNil or lowest < 0 or highest < lowest:
    raise newException(ValueError, "cumulative extrema storage address range is invalid")
  const elementSize = sizeof(T)
  if elementSize <= 0:
    raise newException(ValueError, "cumulative extrema element size is invalid")
  let baseAddress = cast[uint](storageBase)
  var firstOffset, pastOffset: uint
  if not checkedMulUInt(uint(lowest), uint(elementSize), firstOffset):
    raise newException(ValueError, "cumulative extrema byte offset overflows address space")
  var pastElem: uint
  if not checkedAddUInt(uint(highest), 1'u, pastElem) or
      not checkedMulUInt(pastElem, uint(elementSize), pastOffset):
    raise newException(ValueError, "cumulative extrema byte span overflows address space")
  if not checkedAddUInt(baseAddress, firstOffset, result.first) or
      not checkedAddUInt(baseAddress, pastOffset, result.past) or
      result.past <= result.first:
    raise newException(ValueError, "cumulative extrema byte address overflows address space")

proc validateAllocationSize[T](elements: int) =
  if elements <= 0 or sizeof(T) <= 0 or LASER_MEM_ALIGN <= 0:
    raise newException(ValueError, "cumulative extrema replacement allocation size is invalid")
  let overhead = LASER_MEM_ALIGN - 1
  if elements > (high(int) - overhead) div sizeof(T):
    raise newException(ValueError, "cumulative extrema replacement allocation size overflows int")

proc sourceLayout[T](arg: Tensor[T]): OutputLayout[T] =
  if not validMetadataLength(arg.shape) or not validMetadataLength(arg.strides) or
      arg.strides.len != arg.shape.len:
    raise newException(ValueError, "cumulative extrema source stride rank is invalid")
  if arg.offset < 0:
    raise newException(ValueError, "cumulative extrema source offset is negative")
  result.shape = arg.shape
  result.strides = arg.strides
  result.offset = arg.offset
  result.logicalNumel = checkedCumulativeExtremaNumel(arg.shape)
  if result.logicalNumel == 0:
    result.span.isEmpty = true
    return
  if arg.storage.isNil or arg.storage.raw_buffer.isNil:
    raise newException(ValueError, "cumulative extrema source has no storage")
  result.capacityElems = arg.storage.capacityElems
  validateCapacity(result.capacityElems, 0, false)
  if not checkedReachableBounds(result.shape, result.strides, result.offset,
      result.lowestElem, result.highestElem):
    raise newException(ValueError, "cumulative extrema source address range is invalid")
  validateCapacity(result.capacityElems, result.highestElem, false)
  result.storageBase = cast[pointer](arg.storage.raw_buffer)
  result.span = byteSpanFromBounds[T](result.logicalNumel,
      result.lowestElem, result.highestElem, result.storageBase)

proc prepareOutputPlan[T](argShape: Metadata; logicalNumel: int;
    dst: Tensor[T]): OutputPlan[T] =
  if not validMetadataLength(dst.shape) or not validMetadataLength(dst.strides) or
      dst.strides.len != dst.shape.len:
    raise newException(ValueError, "cumulative extrema destination stride rank is invalid")
  if dst.offset < 0:
    raise newException(ValueError, "cumulative extrema destination offset is negative")

  result.storage = dst.storage
  result.layout.logicalNumel = logicalNumel
  let sameShape = dst.shape == argShape
  result.layout.offset = dst.offset
  if sameShape:
    result.layout.shape = dst.shape
    result.layout.strides = dst.strides
  else:
    result.layout.shape = argShape
    result.layout.strides = cumulativeExtremaResizeStrides(argShape)

  if result.layout.logicalNumel == 0:
    result.layout.span.isEmpty = true
    if not result.storage.isNil:
      result.layout.capacityElems = result.storage.capacityElems
      result.layout.storageBase = cast[pointer](result.storage.raw_buffer)
    return

  if not checkedReachableBounds(result.layout.shape, result.layout.strides,
      result.layout.offset, result.layout.lowestElem, result.layout.highestElem):
    raise newException(ValueError, "cumulative extrema destination address range is invalid")
  var requiredElems: int
  if not checkedAddInt(result.layout.highestElem, 1, requiredElems):
    raise newException(ValueError, "cumulative extrema destination capacity overflows int")

  if result.storage.isNil:
    if sameShape and result.layout.shape.len != 0:
      raise newException(ValueError, "same-shaped cumulative extrema output has no storage")
    result.needsReplacement = true
    result.replacementElems = requiredElems
    validateAllocationSize[T](requiredElems)
    return

  let capacity = result.storage.capacityElems
  if capacity < -1:
    raise newException(ValueError, "cumulative extrema storage capacity metadata is invalid")
  let base = cast[pointer](result.storage.raw_buffer)
  if sameShape:
    if base.isNil:
      raise newException(ValueError, "same-shaped cumulative extrema output has no storage")
    validateCapacity(capacity, result.layout.highestElem, true)
    result.layout.storageBase = base
    result.layout.capacityElems = capacity
    result.layout.span = byteSpanFromBounds[T](result.layout.logicalNumel,
        result.layout.lowestElem, result.layout.highestElem, base)
    return

  let fits = not base.isNil and capacity >= 0 and
    result.layout.highestElem < capacity
  if fits:
    result.layout.storageBase = base
    result.layout.capacityElems = capacity
    result.layout.span = byteSpanFromBounds[T](result.layout.logicalNumel,
        result.layout.lowestElem, result.layout.highestElem, base)
    return
  if result.storage.isExternal:
    if base.isNil:
      raise newException(ValueError, "external cumulative extrema output has no storage")
    if capacity < 0:
      raise newException(ValueError, "external cumulative extrema output capacity is unknown")
    raise newException(ValueError, "external cumulative extrema output cannot grow")
  result.needsReplacement = true
  result.replacementElems = requiredElems
  validateAllocationSize[T](requiredElems)

proc stageOutputStorage[T](plan: var OutputPlan[T]) =
  if not plan.needsReplacement:
    return
  var fresh: CpuStorage[T]
  allocCpuStorage(fresh, plan.replacementElems)
  if fresh.isNil or fresh.raw_buffer.isNil:
    raise newException(ValueError, "cumulative extrema replacement allocation has no storage")
  plan.storage = fresh
  plan.layout.storageBase = cast[pointer](fresh.raw_buffer)
  plan.layout.capacityElems = fresh.capacityElems
  validateCapacity(plan.layout.capacityElems, plan.layout.highestElem, true)
  plan.layout.span = byteSpanFromBounds[T](plan.layout.logicalNumel,
      plan.layout.lowestElem, plan.layout.highestElem, plan.layout.storageBase)

proc layoutStartAddress[T](layout: OutputLayout[T]; address: var uint): bool =
  if layout.logicalNumel == 0 or layout.storageBase.isNil or layout.offset < 0:
    return false
  var offsetBytes: uint
  if not checkedMulUInt(uint(layout.offset), uint(sizeof(T)), offsetBytes):
    return false
  checkedAddUInt(cast[uint](layout.storageBase), offsetBytes, address)

proc exactSameIndexMap[A, B](a: OutputLayout[A]; b: OutputLayout[B]): bool =
  if a.logicalNumel == 0 or b.logicalNumel == 0 or a.shape != b.shape:
    return false
  var aStart, bStart: uint
  if not layoutStartAddress(a, aStart) or not layoutStartAddress(b, bStart) or
      aStart != bStart:
    return false
  for axis in 0 ..< a.shape.len:
    if a.shape[axis] <= 1:
      continue
    var aStride, bStride: int
    if not checkedIntBytes(a.strides[axis], sizeof(A), aStride) or
        not checkedIntBytes(b.strides[axis], sizeof(B), bStride):
      return false
    if aStride != bStride:
      return false
  true

proc absIntChecked(value: int; absolute: var int): bool {.inline.} =
  if value == low(int):
    return false
  absolute = if value < 0: -value else: value
  true

proc gcdPositive(a, b: int): int {.inline.} =
  var x = a
  var y = b
  while y != 0:
    let remainder = x mod y
    x = y
    y = remainder
  x

proc floorDivPositive(value, divisor: int; quotient: var int): bool {.inline.} =
  if divisor <= 0:
    return false
  quotient = value div divisor
  let remainder = value mod divisor
  if remainder < 0:
    if quotient == low(int):
      return false
    dec quotient
  true

proc ceilDivPositive(value, divisor: int; quotient: var int): bool {.inline.} =
  if divisor <= 0:
    return false
  quotient = value div divisor
  let remainder = value mod divisor
  if remainder > 0:
    if quotient == high(int):
      return false
    inc quotient
  true

proc containsMultiple(lowValue, highValue, divisor: int): bool =
  if lowValue > highValue:
    return false
  if divisor == 0:
    return lowValue <= 0 and highValue >= 0
  var firstMultiple, lastMultiple: int
  ceilDivPositive(lowValue, divisor, firstMultiple) and
    floorDivPositive(highValue, divisor, lastMultiple) and
    firstMultiple <= lastMultiple

proc spendWork(budget: var int; amount = 1): bool {.inline.} =
  if amount < 0 or budget < amount:
    return false
  budget -= amount
  true

## Extended Euclid gives a particular positive-coefficient solution; the
## caller uses the shared budget while bounding its full solution family.
proc extendedGcd(a, b: int; x, y, divisor: var int;
    budget: var int): SearchResult =
  var oldR = a
  var r = b
  var oldS = 1
  var s = 0
  var oldT = 0
  var t = 1
  while r != 0:
    if not spendWork(budget, 4):
      return srBudget
    let q = oldR div r
    var product, nextR, nextS, nextT: int
    if not checkedMulInt(q, r, product) or not checkedSubInt(oldR, product, nextR) or
        not checkedMulInt(q, s, product) or not checkedSubInt(oldS, product, nextS) or
        not checkedMulInt(q, t, product) or not checkedSubInt(oldT, product, nextT):
      return srUnproven
    oldR = r
    r = nextR
    oldS = s
    s = nextS
    oldT = t
    t = nextT
  x = oldS
  y = oldT
  divisor = oldR
  srFound

## Solves one bounded Diophantine equation exactly; arithmetic or budget
## failure remains inconclusive rather than being treated as disjointness.
proc boundedTwoVariableEquation(a, lowA, highA, b, lowB, highB: int;
    target: int; budget: var int): SearchResult =
  if not spendWork(budget, 4):
    return srBudget
  if a == 0 or b == 0:
    return srUnproven
  var absA, absB: int
  if not absIntChecked(a, absA) or not absIntChecked(b, absB):
    return srUnproven
  var minA = lowA
  var maxA = highA
  var minB = lowB
  var maxB = highB
  if a < 0:
    var negLow, negHigh: int
    if not checkedNegInt(highA, negLow) or not checkedNegInt(lowA, negHigh):
      return srUnproven
    minA = negLow
    maxA = negHigh
  if b < 0:
    var negLow, negHigh: int
    if not checkedNegInt(highB, negLow) or not checkedNegInt(lowB, negHigh):
      return srUnproven
    minB = negLow
    maxB = negHigh

  let divisor = gcdPositive(absA, absB)
  if divisor == 0 or target mod divisor != 0:
    return srNone
  var xCoeff, yCoeff, egcd: int
  let gcdResult = extendedGcd(absA, absB, xCoeff, yCoeff, egcd, budget)
  if gcdResult != srFound:
    return gcdResult
  if egcd != divisor:
    return srUnproven
  let factor = target div divisor
  var x0, y0: int
  if not checkedMulInt(xCoeff, factor, x0) or
      not checkedMulInt(yCoeff, factor, y0):
    return srUnproven
  let stepX = absB div divisor
  let stepY = absA div divisor
  var lowerX, upperX, lowerY, upperY: int
  var numerator: int
  if not spendWork(budget, 8):
    return srBudget
  if not checkedSubInt(minA, x0, numerator) or
      not ceilDivPositive(numerator, stepX, lowerX) or
      not checkedSubInt(maxA, x0, numerator) or
      not floorDivPositive(numerator, stepX, upperX) or
      not checkedSubInt(y0, maxB, numerator) or
      not ceilDivPositive(numerator, stepY, lowerY) or
      not checkedSubInt(y0, minB, numerator) or
      not floorDivPositive(numerator, stepY, upperY):
    return srUnproven
  if max(lowerX, lowerY) <= min(upperX, upperY):
    srFound
  else:
    srNone

## Enumerates the small byte-overlap target interval with exact bounded solves.
proc boundedTwoVariableInterval(a, lowA, highA, b, lowB, highB: int;
    targetLow, targetHigh: int; budget: var int): SearchResult =
  if targetLow > targetHigh:
    return srNone
  var target = targetLow
  while true:
    if not spendWork(budget):
      return srBudget
    let one = boundedTwoVariableEquation(a, lowA, highA, b, lowB, highB,
        target, budget)
    if one == srFound:
      return srFound
    if one != srNone:
      return one
    if target == targetHigh:
      break
    if target == high(int):
      return srUnproven
    inc target
  srNone

proc boundedFinalVariable(coefficient, lowerBound, upperBound: int;
    targetLow, targetHigh: int; requireNonzero: bool;
    hasSolution: var bool): SearchResult

proc twoVariableHasNonzeroZeroSolution(a, lowA, highA, b, lowB, highB: int;
    budget: var int): SearchResult

## Fixed-stack depth-first bounded search with suffix min/max/gcd pruning.
## Each node and candidate spends the shared budget; exhaustion is inconclusive.
proc boundedLinearSearch(coefficients, lowerBounds, upperBounds: openArray[int];
    count: int; targetLow, targetHigh: int; requireNonzero: bool;
    budget: var int): SearchResult =
  if count < 0 or count > MaxAliasVariables or
      coefficients.len < count or lowerBounds.len < count or
      upperBounds.len < count:
    return srUnproven
  var suffixLow, suffixHigh: array[MaxAliasVariables + 1, int]
  var suffixGcd: array[MaxAliasVariables + 1, int]
  suffixLow[count] = 0
  suffixHigh[count] = 0
  suffixGcd[count] = 0
  var axis = count
  while axis > 0:
    dec axis
    if not spendWork(budget, 3):
      return srBudget
    if lowerBounds[axis] > upperBounds[axis]:
      return srUnproven
    var lowProduct, highProduct: int
    if not checkedMulInt(coefficients[axis], lowerBounds[axis], lowProduct) or
        not checkedMulInt(coefficients[axis], upperBounds[axis], highProduct):
      return srUnproven
    let contributionLow = min(lowProduct, highProduct)
    let contributionHigh = max(lowProduct, highProduct)
    if not checkedAddInt(contributionLow, suffixLow[axis + 1], suffixLow[axis]) or
        not checkedAddInt(contributionHigh, suffixHigh[axis + 1], suffixHigh[axis]):
      return srUnproven
    var absCoefficient: int
    if not absIntChecked(coefficients[axis], absCoefficient):
      return srUnproven
    suffixGcd[axis] = gcdPositive(absCoefficient, suffixGcd[axis + 1])

  var rootLow, rootHigh: int
  if not checkedSubInt(targetLow, 0, rootLow) or
      not checkedSubInt(targetHigh, 0, rootHigh):
    return srUnproven
  let reachableLow = max(rootLow, suffixLow[0])
  let reachableHigh = min(rootHigh, suffixHigh[0])
  if not spendWork(budget):
    return srBudget
  if not containsMultiple(reachableLow, reachableHigh, suffixGcd[0]):
    return srNone

  var partialAt, nextCandidate: array[MaxAliasVariables + 1, int]
  var hadNonzeroAt: array[MaxAliasVariables + 1, bool]
  var frameState: array[MaxAliasVariables + 1, int]
  var depth = 0
  while true:
    if not spendWork(budget, 2):
      return srBudget
    if depth == count:
      if partialAt[depth] >= targetLow and partialAt[depth] <= targetHigh and
          (not requireNonzero or hadNonzeroAt[depth]):
        return srFound
      if depth == 0:
        return srNone
      dec depth
      continue

    var needLow, needHigh: int
    if not checkedSubInt(targetLow, partialAt[depth], needLow) or
        not checkedSubInt(targetHigh, partialAt[depth], needHigh):
      return srUnproven
    let possibleLow = max(needLow, suffixLow[depth])
    let possibleHigh = min(needHigh, suffixHigh[depth])
    if not spendWork(budget):
      return srBudget
    if not containsMultiple(possibleLow, possibleHigh, suffixGcd[depth]):
      if depth == 0:
        return srNone
      dec depth
      continue

    let remaining = count - depth
    if remaining == 1:
      if not spendWork(budget, 2):
        return srBudget
      var hasSolution = false
      let finalResult = boundedFinalVariable(coefficients[depth],
          lowerBounds[depth], upperBounds[depth], needLow, needHigh,
          requireNonzero and not hadNonzeroAt[depth], hasSolution)
      if finalResult == srFound and hasSolution:
        return srFound
      if finalResult != srNone:
        return finalResult
      if depth == 0:
        return srNone
      dec depth
      continue

    if remaining == 2:
      if not spendWork(budget, 2):
        return srBudget
      let pairResult = boundedTwoVariableInterval(coefficients[depth],
          lowerBounds[depth], upperBounds[depth], coefficients[depth + 1],
          lowerBounds[depth + 1], upperBounds[depth + 1], needLow, needHigh, budget)
      if pairResult != srFound:
        if pairResult != srNone:
          return pairResult
        if depth == 0:
          return srNone
        dec depth
        continue
      if not requireNonzero or hadNonzeroAt[depth]:
        return srFound

      # If a nonzero residual is possible, some coordinate must be nonzero.
      if needLow < 0:
        let belowZero = boundedTwoVariableInterval(coefficients[depth],
            lowerBounds[depth], upperBounds[depth], coefficients[depth + 1],
            lowerBounds[depth + 1], upperBounds[depth + 1], needLow,
            min(needHigh, -1), budget)
        if belowZero == srFound:
          return srFound
        if belowZero != srNone:
          return belowZero
      if needHigh > 0:
        let aboveZero = boundedTwoVariableInterval(coefficients[depth],
            lowerBounds[depth], upperBounds[depth], coefficients[depth + 1],
            lowerBounds[depth + 1], upperBounds[depth + 1], max(needLow, 1),
            needHigh, budget)
        if aboveZero == srFound:
          return srFound
        if aboveZero != srNone:
          return aboveZero
      if needLow <= 0 and needHigh >= 0:
        let zeroPair = twoVariableHasNonzeroZeroSolution(coefficients[depth],
            lowerBounds[depth], upperBounds[depth], coefficients[depth + 1],
            lowerBounds[depth + 1], upperBounds[depth + 1], budget)
        if zeroPair == srFound:
          return srFound
        if zeroPair != srNone:
          return zeroPair
      if depth == 0:
        return srNone
      dec depth
      continue

    if frameState[depth] == 2:
      if depth == 0:
        return srNone
      dec depth
      continue
    if frameState[depth] == 0:
      nextCandidate[depth] = lowerBounds[depth]
      frameState[depth] = 1
    let candidate = nextCandidate[depth]
    if not spendWork(budget):
      return srBudget
    var term, nextPartial: int
    if not checkedMulInt(coefficients[depth], candidate, term) or
        not checkedAddInt(partialAt[depth], term, nextPartial):
      return srUnproven
    if candidate == upperBounds[depth]:
      frameState[depth] = 2
    else:
      if candidate == high(int):
        return srUnproven
      nextCandidate[depth] = candidate + 1
    partialAt[depth + 1] = nextPartial
    hadNonzeroAt[depth + 1] = hadNonzeroAt[depth] or candidate != 0
    frameState[depth + 1] = 0
    inc depth

## Computes feasible values of the final variable without enumerating it.
proc boundedFinalVariable(coefficient, lowerBound, upperBound: int;
    targetLow, targetHigh: int; requireNonzero: bool;
    hasSolution: var bool): SearchResult =
  if coefficient == 0:
    if targetLow <= 0 and targetHigh >= 0:
      hasSolution = not requireNonzero or lowerBound < 0 or upperBound > 0
      if hasSolution:
        return srFound
    return srNone
  var absCoefficient: int
  if not absIntChecked(coefficient, absCoefficient):
    return srUnproven
  var normalizedLow, normalizedHigh: int
  if coefficient > 0:
    normalizedLow = targetLow
    normalizedHigh = targetHigh
  else:
    if not checkedNegInt(targetHigh, normalizedLow) or
        not checkedNegInt(targetLow, normalizedHigh):
      return srUnproven
  var feasibleLow, feasibleHigh: int
  if not ceilDivPositive(normalizedLow, absCoefficient, feasibleLow) or
      not floorDivPositive(normalizedHigh, absCoefficient, feasibleHigh):
    return srUnproven
  feasibleLow = max(feasibleLow, lowerBound)
  feasibleHigh = min(feasibleHigh, upperBound)
  if feasibleLow <= feasibleHigh and
      (not requireNonzero or feasibleLow < 0 or feasibleHigh > 0):
    hasSolution = true
    return srFound
  srNone
## A two-axis null vector exists iff both symmetric coordinate bounds include
## the primitive Euclidean solution.
proc twoVariableHasNonzeroZeroSolution(a, lowA, highA, b, lowB, highB: int;
    budget: var int): SearchResult =
  if not spendWork(budget, 4):
    return srBudget
  var absA, absB: int
  if not absIntChecked(a, absA) or not absIntChecked(b, absB) or
      absA == 0 or absB == 0:
    return srUnproven
  let divisor = gcdPositive(absA, absB)
  let minAbsA = absB div divisor
  let minAbsB = absA div divisor
  if lowA <= -minAbsA and highA >= minAbsA and
      lowB <= -minAbsB and highB >= minAbsB:
    return srFound
  srNone

## Uses O(rank) separated-stride proofs first, then exact bounded search when
## the common layouts cannot establish injectivity.
proc proveInternallyNonOverlapping(shape, strides: Metadata; logicalNumel: int;
    budget: var int): LayoutProof =
  if logicalNumel <= 1:
    return lpDisjoint
  if shape.len != strides.len or not validMetadataLength(shape):
    return lpUnproven
  for axis in 0 ..< shape.len:
    if shape[axis] > 1 and strides[axis] == 0:
      return lpOverlap

  for reverseOrder in [false, true]:
    var span = 0
    var separated = true
    for step in 0 ..< shape.len:
      let axis = if reverseOrder: shape.len - step - 1 else: step
      if shape[axis] <= 1:
        continue
      var absoluteStride: int
      if not absIntChecked(strides[axis], absoluteStride) or
          absoluteStride <= span:
        separated = false
        break
      var contribution: int
      if not checkedMulInt(shape[axis] - 1, absoluteStride, contribution) or
          not checkedAddInt(span, contribution, span):
        separated = false
        break
    if separated:
      return lpDisjoint

  var coefficients: array[MaxAliasVariables, int]
  var lowerBounds, upperBounds: array[MaxAliasVariables, int]
  var count = 0
  for axis in 0 ..< shape.len:
    if shape[axis] <= 1:
      continue
    coefficients[count] = strides[axis]
    let bound = shape[axis] - 1
    lowerBounds[count] = -bound
    upperBounds[count] = bound
    inc count
  if count <= 1:
    return lpDisjoint
  if count == 2:
    let nonzeroPair = twoVariableHasNonzeroZeroSolution(coefficients[0],
        lowerBounds[0], upperBounds[0], coefficients[1], lowerBounds[1],
        upperBounds[1], budget)
    case nonzeroPair
    of srFound: return lpOverlap
    of srNone: return lpDisjoint
    of srBudget, srUnproven: return lpUnproven
  case boundedLinearSearch(coefficients, lowerBounds, upperBounds, count,
      0, 0, true, budget)
  of srFound: lpOverlap
  of srNone: lpDisjoint
  of srBudget, srUnproven: lpUnproven

proc addressDifference(a, b: uint; difference: var int): bool {.inline.} =
  if a >= b:
    let delta = a - b
    if delta > uint(high(int)):
      return false
    difference = int(delta)
    return true
  let delta = b - a
  let minMagnitude = uint(high(int)) + 1'u
  if delta > minMagnitude:
    return false
  if delta == minMagnitude:
    difference = low(int)
  else:
    difference = -int(delta)
  true

proc appendLayoutCoefficients[A, B](a: OutputLayout[A]; b: OutputLayout[B];
    coefficients: var array[MaxAliasVariables, int];
    lowerBounds, upperBounds: var array[MaxAliasVariables, int];
    count: var int; budget: var int): bool =
  for axis in 0 ..< a.shape.len:
    if a.shape[axis] <= 1:
      continue
    if not spendWork(budget):
      return false
    var coefficient: int
    if not checkedIntBytes(a.strides[axis], sizeof(A), coefficient):
      return false
    if coefficient != 0:
      if count >= MaxAliasVariables:
        return false
      coefficients[count] = coefficient
      lowerBounds[count] = 0
      upperBounds[count] = a.shape[axis] - 1
      inc count
  for axis in 0 ..< b.shape.len:
    if b.shape[axis] <= 1:
      continue
    if not spendWork(budget):
      return false
    var coefficient: int
    if not checkedIntBytes(b.strides[axis], sizeof(B), coefficient) or
        not checkedNegInt(coefficient, coefficient):
      return false
    if coefficient != 0:
      if count >= MaxAliasVariables:
        return false
      coefficients[count] = coefficient
      lowerBounds[count] = 0
      upperBounds[count] = b.shape[axis] - 1
      inc count
  true

## Intersecting byte spans only trigger an exact bounded affine search; they
## are never taken as proof of overlap by themselves.
proc proveLayoutsDisjoint[A, B](a: OutputLayout[A]; b: OutputLayout[B];
    budget: var int): LayoutProof =
  if a.logicalNumel == 0 or b.logicalNumel == 0 or a.span.isEmpty or b.span.isEmpty:
    return lpDisjoint
  if a.span.first >= b.span.past or b.span.first >= a.span.past:
    return lpDisjoint
  var aStart, bStart: uint
  if not layoutStartAddress(a, aStart) or not layoutStartAddress(b, bStart):
    return lpUnproven
  var difference: int
  if not addressDifference(aStart, bStart, difference):
    return lpUnproven
  var targetLow, targetHigh: int
  if not checkedSubInt(-(sizeof(A) - 1), difference, targetLow) or
      not checkedSubInt(sizeof(B) - 1, difference, targetHigh):
    return lpUnproven
  var coefficients: array[MaxAliasVariables, int]
  var lowerBounds, upperBounds: array[MaxAliasVariables, int]
  var count = 0
  if not appendLayoutCoefficients(a, b, coefficients, lowerBounds,
      upperBounds, count, budget):
    return lpUnproven
  let search = boundedLinearSearch(coefficients, lowerBounds, upperBounds,
      count, targetLow, targetHigh, false, budget)
  case search
  of srFound: lpOverlap
  of srNone: lpDisjoint
  of srBudget, srUnproven: lpUnproven

proc requireNonOverlappingLayout(proof: LayoutProof; message: string) =
  case proof
  of lpDisjoint:
    discard
  of lpOverlap:
    raise newException(ValueError, message & " overlaps itself")
  of lpUnproven:
    raise newException(ValueError, message & " internal overlap could not be proven")

proc requireLayoutsDisjoint(proof: LayoutProof; message: string) =
  case proof
  of lpDisjoint:
    discard
  of lpOverlap:
    raise newException(ValueError, message & " overlap in occupied bytes")
  of lpUnproven:
    raise newException(ValueError, message & " disjointness could not be proven")

proc cumulativeExtremaAliasWorkBudget(argNumel, valuesNumel, indicesNumel,
    rank: int): int =
  var total = 1
  for value in [argNumel, valuesNumel, indicesNumel, rank]:
    if value < 0 or not checkedAddInt(total, value, total):
      raise newException(ValueError, "cumulative extrema alias work budget overflows int")
  if not checkedMulInt(total, 64, result):
    raise newException(ValueError, "cumulative extrema alias work budget overflows int")

proc prepareCumulativeExtremaOutputs*[T: SomeNumber or bool or Complex](
    arg: Tensor[T]; values: Tensor[T]; indices: Tensor[int64]): CumulativeExtremaOutputs[T] =
  let source = sourceLayout(arg)
  var valuesPlan = prepareOutputPlan(arg.shape, source.logicalNumel, values)
  var indicesPlan = prepareOutputPlan(arg.shape, source.logicalNumel, indices)

  var workBudget = cumulativeExtremaAliasWorkBudget(source.logicalNumel,
      valuesPlan.layout.logicalNumel, indicesPlan.layout.logicalNumel,
      arg.shape.len)

  requireNonOverlappingLayout(proveInternallyNonOverlapping(
      valuesPlan.layout.shape, valuesPlan.layout.strides,
      valuesPlan.layout.logicalNumel, workBudget), "values output")
  requireNonOverlappingLayout(proveInternallyNonOverlapping(
      indicesPlan.layout.shape, indicesPlan.layout.strides,
      indicesPlan.layout.logicalNumel, workBudget), "indices output")

  # Only byte-map checks need the final staged allocation addresses.
  stageOutputStorage(valuesPlan)
  stageOutputStorage(indicesPlan)
  requireLayoutsDisjoint(proveLayoutsDisjoint(valuesPlan.layout,
      indicesPlan.layout, workBudget), "values and indices outputs")

  if source.logicalNumel > 0:
    result.valuesExactInput = exactSameIndexMap(source, valuesPlan.layout)
    when sizeof(T) == sizeof(int64):
      result.indicesExactInput = exactSameIndexMap(source, indicesPlan.layout)
    else:
      when T is Complex:
        if source.shape.len == 0 and sizeof(T) > sizeof(int64):
          var sourceStart, indexStart: uint
          result.indicesExactInput = layoutStartAddress(source, sourceStart) and
            layoutStartAddress(indicesPlan.layout, indexStart) and
            sourceStart == indexStart

    if not result.valuesExactInput:
      requireLayoutsDisjoint(proveLayoutsDisjoint(source,
          valuesPlan.layout, workBudget), "values output and source")
    if not result.indicesExactInput:
      requireLayoutsDisjoint(proveLayoutsDisjoint(source,
          indicesPlan.layout, workBudget), "indices output and source")

  result.values = values
  result.values.shape = valuesPlan.layout.shape
  result.values.strides = valuesPlan.layout.strides
  result.values.offset = valuesPlan.layout.offset
  result.values.storage = valuesPlan.storage
  result.indices = indices
  result.indices.shape = indicesPlan.layout.shape
  result.indices.strides = indicesPlan.layout.strides
  result.indices.offset = indicesPlan.layout.offset
  result.indices.storage = indicesPlan.storage
