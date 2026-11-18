# Copyright 2017 the Arraymancer contributors
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

import  ./backend/openmp,
        ./private/p_checks,
        ./private/p_inner_spans,
        ./private/p_accelerate,
        ./data_structure, ./init_cpu, ./accessors,
        ../laser/compiler_optim_hints

import std / [sugar, math]
# Valid tensor spans already fit their backing storage. Wrapping index arithmetic
# avoids per-element overflow branches without disabling checks in the user's op.

# ####################################################################
# Mapping over tensors

when not defined(nimHasEffectsOf):
  {.pragma: effectsOf.}

template apply_inline*[T: KnownSupportsCopyMem](t: var Tensor[T], op: untyped): untyped =
  # TODO, if t is a result of a function
  # how to ensure that it is not called twice
  omp_parallel_blocks(block_offset, block_size, t.size):
    for x {.inject.} in t.mitems(block_offset, block_size):
      x = op

template apply_inline*[T: not KnownSupportsCopyMem](t: var Tensor[T], op: untyped): untyped =
  # TODO, if t is a result of a function
  # how to ensure that it is not called twice
  for x {.inject.} in t.mitems():
    x = op

template apply2_inline*[T: KnownSupportsCopyMem; U](dest: var Tensor[T],
                                                   src: Tensor[U],
                                                   op: untyped): untyped =
  let output = addr dest
  let input = src
  when compileOption("boundChecks"):
    check_elementwise(output[], input)
  when U is KnownSupportsCopyMem:
    if output[].is_C_contiguous and input.is_C_contiguous and
        disjointOwnedStorage(output[], input):
      let destination = output[].storage.raw_buffer
      let source = input.storage.raw_buffer
      let destinationOffset = output[].offset
      let sourceOffset = input.offset
      omp_parallel_blocks(block_offset, block_size, output[].size):
        for cursor in uint(block_offset) ..< uint(block_offset + block_size):
          let i = cast[int](cursor)
          template x: untyped {.inject.} = destination[destinationOffset +% i]
          let y {.inject, used.} = source[sourceOffset +% i]
          x = op
    elif output[].is_C_contiguous and disjointOwnedStorage(output[], input) and output[].shape == input.shape:
      let strides = [output[].strides, input.strides]
      let span = innerSpan(output[].shape, strides)
      let origins = [output[].offset, input.offset]
      let destination = output[].storage.raw_buffer
      let source = input.storage.raw_buffer
      if span.length > 1 or output[].size == 1:
        for _, offsets in spanOffsets(output[].shape, strides, origins,
            span.outerRank, output[].size div span.length):
          withSpanStep(span.steps[1], sourceStep):
            for cursor in 0'u ..< uint(span.length):
              let i = cast[int](cursor)
              template x: untyped {.inject.} = destination[offsets[0] +% i]
              let y {.inject, used.} = source[offsets[1] +% (i *% sourceStep)]
              x = op
      else:
        omp_parallel_blocks(block_offset, block_size, output[].size):
          for x {.inject.}, y {.inject.} in mzip(output[], input, block_offset, block_size):
            x = op
    else:
      omp_parallel_blocks(block_offset, block_size, output[].size):
        for x {.inject.}, y {.inject.} in mzip(output[], input, block_offset, block_size):
          x = op
  else:
    omp_parallel_blocks(block_offset, block_size, output[].size):
      for x {.inject.}, y {.inject.} in mzip(output[], input, block_offset, block_size):
        x = op

template apply2_inline*[T: not KnownSupportsCopyMem; U](dest: var Tensor[T],
                                                   src: Tensor[U],
                                                   op: untyped): untyped =
  let output = addr dest
  let input = src
  for x {.inject.}, y {.inject.} in mzip(output[], input):
    x = op

template apply3_inline*[T: KnownSupportsCopyMem;U,V](dest: var Tensor[T],
                                                     src1: Tensor[U],
                                                     src2: Tensor[V],
                                                     op: untyped): untyped =
  let output = addr dest
  let input1 = src1
  let input2 = src2
  when compileOption("boundChecks"):
    check_elementwise(output[], input1)
    check_elementwise(output[], input2)
  when U is KnownSupportsCopyMem and V is KnownSupportsCopyMem:
    if output[].is_C_contiguous and disjointOwnedStorage(output[], input1) and
        disjointOwnedStorage(output[], input2) and output[].shape == input1.shape and input1.shape == input2.shape:
      let strides = [output[].strides, input1.strides, input2.strides]
      let span = innerSpan(output[].shape, strides)
      let origins = [output[].offset, input1.offset, input2.offset]
      withCompilerOptimHints()
      # The checked storage separation also excludes loop-carried output aliases.
      let destination {.restrict.} = output[].storage.raw_buffer
      let source1 = input1.storage.raw_buffer
      let source2 = input2.storage.raw_buffer
      if span.length > 1 or output[].size == 1:
        for _, offsets in spanOffsets(output[].shape, strides, origins,
            span.outerRank, output[].size div span.length):
          withSpanStep(span.steps[1], firstStep):
            withSpanStep(span.steps[2], secondStep):
              for cursor in 0'u ..< uint(span.length):
                let i = cast[int](cursor)
                template x: untyped {.inject.} = destination[offsets[0] +% i]
                let y {.inject, used.} = source1[offsets[1] +% (i *% firstStep)]
                let z {.inject, used.} = source2[offsets[2] +% (i *% secondStep)]
                x = op
      else:
        omp_parallel_blocks(block_offset, block_size, output[].size):
          for x {.inject.}, y {.inject.}, z {.inject.} in mzip(output[], input1, input2, block_offset, block_size):
            x = op
    else:
      omp_parallel_blocks(block_offset, block_size, output[].size):
        for x {.inject.}, y {.inject.}, z {.inject.} in mzip(output[], input1, input2, block_offset, block_size):
          x = op
  else:
    omp_parallel_blocks(block_offset, block_size, output[].size):
      for x {.inject.}, y {.inject.}, z {.inject.} in mzip(output[], input1, input2, block_offset, block_size):
        x = op

type UnaryMathOp* = enum
  unaryExp, unaryLn, unarySqrt

proc applyUnary*(dest: var Tensor[float32]; src: Tensor[float32];
                 operation: static[UnaryMathOp]) =
  ## Write elementwise exp, ln or sqrt into an equally shaped destination.
  ## Reuse destination storage; aliases and strided views use scalar traversal.
  ## Disjoint contiguous macOS buffers use bulk math without input copies.
  if dest.shape != src.shape:
    raise newException(ValueError, "Unary input and output shapes must agree")
  let count = src.size
  if count == 0: return
  when defined(macosx):
    if dest.is_C_contiguous and src.is_C_contiguous and disjointOwnedStorage(dest, src):
      when operation == unarySqrt:
        sqrtFloat32(addr dest.storage.raw_buffer[dest.offset],
            addr src.storage.raw_buffer[src.offset], count)
      else:
        var offset = 0
        while offset < count:
          var chunk = min(count - offset, high(cint).int).cint
          let destination = addr dest.storage.raw_buffer[dest.offset + offset]
          let source = addr src.storage.raw_buffer[src.offset + offset]
          when operation == unaryExp:
            vvexpf(destination, source, addr chunk)
          else:
            vvlogf(destination, source, addr chunk)
          offset += chunk.int
      return
  when operation == unaryExp:
    dest.apply2_inline(src, math.exp(y))
  elif operation == unaryLn:
    dest.apply2_inline(src, math.ln(y))
  else:
    dest.apply2_inline(src, math.sqrt(y))

template map_inline*[T](t: Tensor[T], op:untyped): untyped =

  let z = t # ensure that if t is the result of a function it is not called multiple times

  type outType = type((
    block:
      var x{.inject, used.}: type(items(z));
      op
  ))

  when outType is KnownSupportsCopyMem:
    var dest = newTensorUninit[outType](z.shape)
    let data = dest.storage.raw_buffer
    when T is KnownSupportsCopyMem:
      if z.is_C_contiguous:
        let source = z.storage.raw_buffer
        let offset = z.offset
        omp_parallel_blocks(block_offset, block_size, dest.size):
          for cursor in uint(block_offset) ..< uint(block_offset + block_size):
            let i = cast[int](cursor)
            let x {.inject, used.} = source[offset +% i]
            data[i] = op
      else:
        omp_parallel_blocks(block_offset, block_size, dest.size):
          for i, x {.inject.} in enumerate(z, block_offset, block_size):
            data[i] = op
    else:
      omp_parallel_blocks(block_offset, block_size, dest.size):
        for i, x {.inject.} in enumerate(z, block_offset, block_size):
          data[i] = op
  else:
    var dest = newTensorUninit[outType](z.shape)
    for i, x {.inject.} in enumerate(z):
      ## NOTE: we assign to the raw buffer directly as enumerate yields the indices as contiguous
      dest.storage.raw_buffer[i] = op
  dest

template map2_inline*[T, U](t1: Tensor[T], t2: Tensor[U], op:untyped): untyped =

  let
    z1 = t1 # ensure that if t1 is the result of a function it is not called multiple times
    z2 = t2

  when compileOption("boundChecks"):
    check_elementwise(z1,z2)

  type outType = type((
    block:
      var x{.inject.}: type(items(z1));
      var y{.inject.}: type(items(z2));
      op
  ))


  when outType is KnownSupportsCopyMem:
    var dest = newTensorUninit[outType](z1.shape)
    withCompilerOptimHints()
    # Fresh output storage cannot alias either input, even when the inputs overlap.
    let data {.restrict.} = dest.storage.raw_buffer
    when T is KnownSupportsCopyMem and U is KnownSupportsCopyMem:
      let strides = [z1.strides, z2.strides]
      let span = innerSpan(z1.shape, strides)
      let origins = [z1.offset, z2.offset]
      let source1 = z1.storage.raw_buffer
      let source2 = z2.storage.raw_buffer
      if z1.is_C_contiguous and z2.is_C_contiguous:
        omp_parallel_blocks(block_offset, block_size, z1.size):
          for cursor in uint(block_offset) ..< uint(block_offset + block_size):
            let i = cast[int](cursor)
            let x {.inject, used.} = source1[z1.offset +% i]
            let y {.inject, used.} = source2[z2.offset +% i]
            data[i] = op
      elif span.length > 1 or z1.size == 1:
        for outer, offsets in spanOffsets(z1.shape, strides, origins,
            span.outerRank, z1.size div span.length):
          let firstOffset = offsets[0]
          let secondOffset = offsets[1]
          let outputOffset = outer *% span.length
          let count = uint(span.length)
          withSpanStep(span.steps[0], firstStep):
            withSpanStep(span.steps[1], secondStep):
              for cursor in 0'u ..< count:
                let i = cast[int](cursor)
                let x {.inject, used.} = source1[firstOffset +% (i *% firstStep)]
                let y {.inject, used.} = source2[secondOffset +% (i *% secondStep)]
                data[outputOffset +% i] = op
      else:
        omp_parallel_blocks(block_offset, block_size, z1.size):
          for i, x {.inject.}, y {.inject.} in enumerateZip(z1, z2, block_offset, block_size):
            data[i] = op
    else:
      omp_parallel_blocks(block_offset, block_size, z1.size):
        for i, x {.inject.}, y {.inject.} in enumerateZip(z1, z2, block_offset, block_size):
          data[i] = op
  else:
    var dest = newTensorUninit[outType](z1.shape)
    for i, x {.inject.}, y {.inject.} in enumerateZip(z1, z2):
      dest.storage.raw_buffer[i] = op
  dest

template map3_inline*[T, U, V](t1: Tensor[T], t2: Tensor[U], t3: Tensor[V], op:untyped): untyped =
  let
    z1 = t1 # ensure that if t1 is the result of a function it is not called multiple times
    z2 = t2
    z3 = t3

  when compileOption("boundChecks"):
    check_elementwise(z1,z2)
    check_elementwise(z1,z3)

  type outType = type((
    block:
      var x{.inject.}: type(items(z1));
      var y{.inject.}: type(items(z2));
      var z{.inject.}: type(items(z3));
      op
  ))
  when outType is KnownSupportsCopyMem:
    var dest = newTensorUninit[outType](z1.shape)
    let data = dest.storage.raw_buffer
    when T is KnownSupportsCopyMem and U is KnownSupportsCopyMem and V is KnownSupportsCopyMem:
      let strides = [z1.strides, z2.strides, z3.strides]
      let span = innerSpan(z1.shape, strides)
      let origins = [z1.offset, z2.offset, z3.offset]
      let source1 = z1.storage.raw_buffer
      let source2 = z2.storage.raw_buffer
      let source3 = z3.storage.raw_buffer
      if z1.is_C_contiguous and z2.is_C_contiguous and z3.is_C_contiguous:
        omp_parallel_blocks(block_offset, block_size, z1.size):
          for cursor in uint(block_offset) ..< uint(block_offset + block_size):
            let i = cast[int](cursor)
            let x {.inject, used.} = source1[z1.offset +% i]
            let y {.inject, used.} = source2[z2.offset +% i]
            let z {.inject, used.} = source3[z3.offset +% i]
            data[i] = op
      elif span.length > 1 or z1.size == 1:
        for outer, offsets in spanOffsets(z1.shape, strides, origins,
            span.outerRank, z1.size div span.length):
          let firstOffset = offsets[0]
          let secondOffset = offsets[1]
          let thirdOffset = offsets[2]
          let outputOffset = outer *% span.length
          let count = uint(span.length)
          withSpanStep(span.steps[0], firstStep):
            withSpanStep(span.steps[1], secondStep):
              withSpanStep(span.steps[2], thirdStep):
                for cursor in 0'u ..< count:
                  let i = cast[int](cursor)
                  let x {.inject, used.} = source1[firstOffset +% (i *% firstStep)]
                  let y {.inject, used.} = source2[secondOffset +% (i *% secondStep)]
                  let z {.inject, used.} = source3[thirdOffset +% (i *% thirdStep)]
                  data[outputOffset +% i] = op
      else:
        omp_parallel_blocks(block_offset, block_size, z1.size):
          for i, x {.inject.}, y {.inject.}, z {.inject.} in enumerateZip(z1, z2, z3, block_offset, block_size):
            data[i] = op
    else:
      omp_parallel_blocks(block_offset, block_size, z1.size):
        for i, x {.inject.}, y {.inject.}, z {.inject.} in enumerateZip(z1, z2, z3, block_offset, block_size):
          data[i] = op
  else:
    var dest = newTensorUninit[outType](z1.shape)
    for i, x {.inject.}, y {.inject.}, z {.inject.} in enumerateZip(z1, z2, z3):
      dest.storage.raw_buffer[i] = op
  dest

proc map*[T; U](t: Tensor[T], f: T -> U): Tensor[U] {.noinit, effectsOf: f.} =
  ## Apply a unary function in an element-wise manner on Tensor[T], returning a new Tensor.
  ## Usage with Nim's ``sugar`` module:
  ##  .. code:: nim
  ##     a.map(x => x+1) # Map the anonymous function x => x+1
  ## Usage with named functions:
  ##  .. code:: nim
  ##     proc plusone[T](x: T): T =
  ##       x + 1
  ##     a.map(plusone) # Map the function plusone
  ## Note:
  ##   for basic operation, you can use implicit broadcasting instead
  ##   with operators prefixed by a dot :
  ##  .. code:: nim
  ##     a +. 1
  ## ``map`` is especially useful to do multiple element-wise operations on a tensor in a single loop over the data.
  ##
  ## For types that are not mem-copyable types (ref, string, etc.) a non OpenMP accelerated version of
  ## `apply2_inline` is used internally!
  result = newTensorUninit[U](t.shape)
  result.apply2_inline(t, f(y))

proc apply*[T](t: var Tensor[T], f: T -> T) {.effectsOf: f.} =
  ## Apply a unary function in an element-wise manner on Tensor[T], in-place.
  ##
  ## Input:
  ##   - a var Tensor
  ##   - A function or anonymous function that ``returns a value``
  ## Result:
  ##   - Nothing, the ``var`` Tensor is modified in-place
  ##
  ## Usage with Nim's ``sugar`` module:
  ##  .. code:: nim
  ##     var a = newTensor([5,5], int) # a must be ``var``
  ##     a.apply(x => x+1) # Map the anonymous function x => x+1
  ## Usage with named functions:
  ##  .. code:: nim
  ##     proc plusone[T](x: T): T =
  ##       x + 1
  ##     a.apply(plusone) # Apply the function plusone in-place
  t.apply_inline(f(x))

proc apply*[T: KnownSupportsCopyMem](t: var Tensor[T], f: proc(x:var T)) {.effectsOf: f.} =
  ## Apply a unary function in an element-wise manner on Tensor[T], in-place.
  ##
  ## Input:
  ##   - a var Tensor
  ##   - An in-place function that ``returns no value``
  ## Result:
  ##   - Nothing, the ``var`` Tensor is modified in-place
  ##
  ## Usage with Nim's ``sugar`` module:
  ##   - The sugar module has a functional programming paradigm,
  ##     anonymous functions cannot mutate the arguments
  ## Usage with named functions:
  ##  .. code:: nim
  ##     proc pluseqone[T](x: var T) =
  ##       x += 1
  ##     a.apply(pluseqone) # Apply the in-place function pluseqone
  ## ``apply`` is especially useful to do multiple element-wise operations
  ## on a tensor in a single loop over the data.

  omp_parallel_blocks(block_offset, block_size, t.size):
    for x in t.mitems(block_offset, block_size):
      f(x)

proc map2*[T, U; V: KnownSupportsCopyMem](t1: Tensor[T],
                                          f: (T,U) -> V,
                                          t2: Tensor[U]): Tensor[V] {.noinit, effectsOf: f.} =
  ## Apply a binary function in an element-wise manner on two Tensor[T], returning a new Tensor.
  ##
  ## The function is applied on the elements with the same coordinates.
  ##
  ## Input:
  ##   - A tensor
  ##   - A function
  ##   - A tensor
  ## Result:
  ##   - A new tensor
  ## Usage with named functions:
  ##  .. code:: nim
  ##     proc `**`[T](x, y: T): T = # We create a new power `**` function that works on 2 scalars
  ##       pow(x, y)
  ##     a.map2(`**`, b)
  ##     # Or
  ##     map2(a, `**`, b)
  ## ``map2`` is especially useful to do multiple element-wise operations on a two tensors in a single loop over the data.
  ## for example ``alpha * sin(A) + B``
  ##
  ## For OpenMP compatibility, this ``map2`` doesn't allow ref types as result like seq or string
  when compileOption("boundChecks"):
    check_elementwise(t1,t2)

  result = newTensorUninit[V](t1.shape)
  result.apply3_inline(t1, t2, f(y,z))

proc map2*[T, U; V: not KnownSupportsCopyMem](t1: Tensor[T],
                                              f: (T,U) -> V,
                                              t2: Tensor[U]): Tensor[V] {.noinit,noSideEffect, effectsOf: f.}=
  ## Apply a binary function in an element-wise manner on two Tensor[T], returning a new Tensor.
  ##
  ##
  ## This is a fallback for ref types as
  ## OpenMP will not work with if the results allocate memory managed by GC.
  when compileOption("boundChecks"):
    check_elementwise(t1,t2)

  result = newTensorUninit[V](t1.shape)
  for r, a, b in mzip(result, t1, t2):
    r = f(t1, t2)

proc apply2*[T: KnownSupportsCopyMem, U](a: var Tensor[T],
                                         f: proc(x:var T, y:T), # We can't use the nice sugar syntax here
                                         b: Tensor[U]) {.effectsOf: f.} =
  ## Apply a binary in-place function in an element-wise manner on two Tensor[T], returning a new Tensor.
  ## Overload for types that are not mem-copyable.
  ##
  ## The function is applied on the elements with the same coordinates.
  ##
  ## Input:
  ##   - A var tensor
  ##   - A function
  ##   - A tensor
  ## Result:
  ##   - Nothing, the ``var``Tensor is modified in-place
  ## Usage with named functions:
  ##  .. code:: nim
  ##     proc `**=`[T](x, y: T) = # We create a new in-place power `**=` function that works on 2 scalars
  ##       x = pow(x, y)
  ##     a.apply2(`**=`, b)
  ##     # Or
  ##     apply2(a, `**=`, b)
  ##
  ## ``apply2`` is especially useful to do multiple element-wise operations on a two tensors in a single loop over the data.
  ## for example ``A += alpha * sin(A) + B``
  when compileOption("boundChecks"):
    check_elementwise(a,b)

  omp_parallel_blocks(block_offset, block_size, a.size):
    for x, y in mzip(a, b, block_offset, block_size):
      f(x, y)

proc apply2*[T: not KnownSupportsCopyMem; U](a: var Tensor[T],
                                             f: proc(x:var T, y: T), # We can't use the nice sugar syntax here
                                             b: Tensor[U]) {.effectsOf: f.} =
  ## Apply a binary in-place function in an element-wise manner on two Tensor[T], returning a new Tensor.
  ## Overload for types that are not mem-copyable.
  ##
  ## The function is applied on the elements with the same coordinates.
  ##
  ## Input:
  ##   - A var tensor
  ##   - A function
  ##   - A tensor
  ## Result:
  ##   - Nothing, the ``var``Tensor is modified in-place
  ## Usage with named functions:
  ##  .. code:: nim
  ##     proc `**=`[T](x, y: T) = # We create a new in-place power `**=` function that works on 2 scalars
  ##       x = pow(x, y)
  ##     a.apply2(`**=`, b)
  ##     # Or
  ##     apply2(a, `**=`, b)
  ## ``apply2`` is especially useful to do multiple element-wise operations on a two tensors in a single loop over the data.
  ## for example ``A += alpha * sin(A) + B``
  when compileOption("boundChecks"):
    check_elementwise(a,b)
  for x, y in mzip(a, b):
    f(x, y)
