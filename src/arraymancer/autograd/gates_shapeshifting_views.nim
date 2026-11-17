# Copyright 2017-2018 Mamy André-Ratsimbazafy & the Arraymancer contributors
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

import  macros,
        ../tensor,
        ../private/ast_utils,
        ../tensor/private/[p_accessors_macros_desugar, p_accessors_macros_read,
                           p_accessors],
        ./autograd_common

type SliceGate[TT] = ref object of Gate[TT]
  input: Variable[TT]
  shape, strides: Metadata
  offset: int

proc slice_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let gate = SliceGate[TT](self)
  result = newDiffs[TT](1)
  result[0] = gate.input.value.zeros_like
  var selected = result[0]
  selected.shape = gate.shape
  selected.strides = gate.strides
  selected.offset = gate.offset
  selected += payload.variable.grad

proc sliceVariable[TT](input: Variable[TT], value, gradientView: TT): Variable[TT] =
  ## Record a slice in logical input coordinates, never aliasing gradient storage.
  result = input.context.variable(value)
  if input.is_grad_needed:
    let gate = SliceGate[TT](input: input, shape: gradientView.shape,
                             strides: gradientView.strides, offset: gradientView.offset)
    result.requires_grad = true
    result.grad = value.zeros_like
    register_node("Slice", gate, slice_backward_ag[TT], result, input)

proc pointVariable[T](input: Variable[Tensor[T]], indices: varargs[int]): Variable[Tensor[T]] =
  var scalar = newTensorUninit[T]([])
  scalar.unsafe_raw_offset[0] = input.value.atIndex(indices)
  var gradientView = input.grad
  if input.is_grad_needed:
    gradientView.offset = gradientView.getIndex(indices)
    gradientView.shape = Metadata()
    gradientView.strides = Metadata()
  sliceVariable(input, scalar, gradientView)

type SelectGate[T] = ref object of Gate[Tensor[T]]
  input: Variable[Tensor[T]]
  indices: seq[int]
  axis: int # -1 means flattened full-mask selection.

proc select_backward_ag[T](self: Gate[Tensor[T]], payload: Payload[Tensor[T]]): SmallDiffs[Tensor[T]] =
  let gate = SelectGate[T](self)
  result = newDiffs[Tensor[T]](1)
  result[0] = gate.input.value.zeros_like
  let gradient = payload.variable.grad
  for i, index in gate.indices:
    if gate.axis < 0:
      result[0].atContiguousIndex(index) += gradient.atContiguousIndex(i)
    else:
      var destination = result[0].atAxisIndex(gate.axis, index)
      destination += gradient.atAxisIndex(gate.axis, i)

proc selectedVariable[T](input: Variable[Tensor[T]], value: Tensor[T],
                         indices: seq[int], axis: int): Variable[Tensor[T]] =
  result = input.context.variable(value)
  if input.is_grad_needed:
    let gate = SelectGate[T](input: input, indices: indices, axis: axis)
    result.requires_grad = true
    result.grad = value.zeros_like
    register_node("Select", gate, select_backward_ag[T], result, input)

proc indexVariable[T, I](input: Variable[Tensor[T]], axis: int, indices: I): Variable[Tensor[T]] =
  ## Snapshot selectors so later caller mutation cannot redirect backward.
  if axis < 0 or axis >= input.value.rank:
    raise newException(IndexDefect, "Selection axis is outside the input rank")
  when I is Tensor:
    if indices.rank != 1:
      raise newException(ValueError, "Index selectors must have rank one")
  when I is Tensor:
    type IndexType = I.T
  else:
    type IndexType = typeof(indices[0])
  when IndexType isnot (byte or char or SomeInteger):
    {.error: "Index selectors must contain integers".}
  var saved: seq[int]
  for index in indices:
    var idx = int(index)
    if idx < 0: idx += input.value.shape[axis]
    if idx < 0 or idx >= input.value.shape[axis]:
      raise newException(IndexDefect, "Selection index is outside the input axis")
    saved.add(idx)
  var shape = input.value.shape
  shape[axis] = saved.len
  var value = newTensorUninit[T](shape)
  for i, index in saved:
    var destination = value.atAxisIndex(axis, i)
    destination.copyFrom(input.value.atAxisIndex(axis, index))
  selectedVariable(input, value, saved, axis)

proc maskVariable[T](input: Variable[Tensor[T]], mask: Tensor[bool], axis: int): Variable[Tensor[T]] =
  var saved: seq[int]
  if axis < 0:
    if mask.shape != input.value.shape:
      raise newException(ValueError, "Full mask shape must match the input")
  else:
    if axis >= input.value.rank or mask.rank != 1 or mask.size != input.value.shape[axis]:
      raise newException(ValueError, "Axis mask length must match the input axis")
  for i, take in enumerate(mask):
    if take: saved.add(i)
  if axis >= 0:
    return indexVariable(input, axis, saved)
  var value = newTensorUninit[T](saved.len)
  var next = 0
  for i, element in enumerate(input.value):
    if next < saved.len and saved[next] == i:
      value.unsafe_raw_offset[next] = element
      inc next
  selectedVariable(input, value, saved, -1)

macro variableSliceDispatch(input: typed, args: varargs[typed]): untyped =
  checkValidSliceType(args)
  if isAllInt(args):
    result = newCall(bindSym"pointVariable", input)
    for arg in args: result.add(arg)
    return
  var selector: NimNode
  var axis: int
  let fancy = getFancySelector(args, axis, selector)
  if fancy != FancyNone:
    # Keep typed symbols intact; the tensor dispatch helper erases gensym identity.
    selector = args[if args[0].sameType(bindSym"Ellipsis"): args.len - axis else: axis]
    let z = genSym(nskLet, "input")
    let selection = genSym(nskLet, "selector")
    let axisNode =
      if args[0].sameType(bindSym"Ellipsis"):
        quote do: `z`.value.rank - `axis`
      else: newLit(axis)
    let maskAxis = if fancy in {FancyMaskFull, FancyUnknownFull}: newLit(-1) else: axisNode
    result = quote do:
      block:
        let `z` = `input`
        let `selection` = `selector`
        when typeof(`selection`) is Tensor[bool]:
          maskVariable(`z`, `selection`, `maskAxis`)
        elif typeof(`selection`) is Tensor:
          indexVariable(`z`, `axisNode`, `selection`)
        elif typeof(toTensor(`selection`)) is Tensor[bool]:
          maskVariable(`z`, toTensor(`selection`), `maskAxis`)
        else:
          indexVariable(`z`, `axisNode`, `selection`)
    return
  let z = genSym(nskLet, "input")
  var bindings = newStmtList()
  var evaluated = newNimNode(nnkArgList)
  for arg in args:
    let name = genSym(nskLet, "slice")
    bindings.add newLetStmt(name, arg)
    evaluated.add(name)
  let forward = newCall(bindSym"slice_typed_dispatch", newDotExpr(z, ident"value"))
  let gradient = newCall(bindSym"slice_typed_dispatch", newDotExpr(z, ident"grad"))
  for arg in evaluated:
    forward.add(arg)
    gradient.add(arg)
  result = quote do:
    block:
      let `z` = `input`
      `bindings`
      let value = `forward`
      if `z`.is_grad_needed:
        sliceVariable(`z`, value, `gradient`)
      else:
        `z`.context.variable(value)

macro `[]`*[TT](v: Variable[TT], args: varargs[untyped]): untyped =
  ## Index or slice a variable; backward scatters into the logical input shape.
  ## Basic slices share values, but gradients use independent storage.
  let normalized = getAst(desugar(args))
  let input = genSym(nskLet, "input")
  var body = newStmtList(newLetStmt(input, v))
  let call = newCall(bindSym"variableSliceDispatch", input)
  for arg in normalized:
    if arg.kind == nnkBracket:
      let selector = genSym(nskLet, "selector")
      body.add newLetStmt(selector, arg)
      call.add(selector)
    else:
      call.add(arg)
  body.add(call)
  result = newTree(nnkBlockStmt, newEmptyNode(), body)

# #############################################

type ReshapeGate*[TT] {.final.} = ref object of Gate[TT]
  cached_input_shape: Metadata

proc reshape_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = ReshapeGate[TT](self)
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)
  result[0] = gradient.reshape(self.cached_input_shape)

proc reshape_cache[TT](result: Variable[TT], a: Variable[TT]) =
  # Gate
  var gate: ReshapeGate[TT]
  new gate
  gate.cached_input_shape = a.value.shape

  # Result setup
  result.grad = zeros_like(result.value)
  result.requires_grad = true

  # Add to graph
  register_node(
    "Reshape",
    gate,
    reshape_backward_ag[TT],
    result,
    a
  )

proc reshapeImpl[TT](a: Variable[TT], shape: Metadata): Variable[TT] =
  # Resulting var
  new result
  result.context = a.context
  result.value = a.value.reshape(shape)

  # Caching for backprop
  if a.is_grad_needed:
    result.reshape_cache(a)

proc reshape*[TT](a: Variable[TT], shape: varargs[int]): Variable[TT] =
  ## Input:
  ##   - A variable
  ##   - A shape
  reshapeImpl(a, shape.toMetadata)

proc reshape*[TT](a: Variable[TT], shape: Metadata): Variable[TT] =
  ## Input:
  ##   - A variable
  ##   - A shape
  reshapeImpl(a, shape)

proc flatten*[TT](a: Variable[TT]): Variable[TT] =
  ## Input:
  ##   - A variable
  reshapeImpl(a, [a.value.shape[0], a.value.size div a.value.shape[0]].toMetadata)

# ############################################################
#
#                   Squeeze / Unsqueeze
#
# ############################################################

type SingletonViewGate[TT] {.final.} = ref object of Gate[TT]
  input_shape: Metadata

proc singletonViewBackward[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = SingletonViewGate[TT](self)
  result = newDiffs[TT](1)
  result[0] = payload.variable.grad.reshape(self.input_shape)

proc singletonView[TT](v: Variable[TT], value: TT): Variable[TT] =
  ## Restoring the input shape also handles squeeze axes that were non-singleton no-ops.
  new result
  result.context = v.context
  result.value = value
  if v.is_grad_needed:
    let gate = SingletonViewGate[TT](input_shape: v.value.shape)
    result.requires_grad = true
    result.grad = result.value.zeros_like
    register_node("SingletonView", gate, singletonViewBackward[TT], result, v)

proc squeeze*[TT](v: Variable[TT]): Variable[TT] =
  ## Remove every singleton axis as a storage-sharing view; all-singleton inputs become scalar.
  singletonView(v, v.value.squeeze)

proc squeeze*[TT](v: Variable[TT], axis: int): Variable[TT] =
  ## Remove a singleton axis, accepting negative axes; non-singleton axes are unchanged.
  ## Backward always restores the original input shape.
  singletonView(v, v.value.squeeze(axis))

proc unsqueeze*[TT](v: Variable[TT], axis: int): Variable[TT] =
  ## Insert a storage-sharing singleton axis in ``[-rank-1, rank]``, including into scalars.
  singletonView(v, v.value.unsqueeze(axis))

# ############################################################
#
#                   Transpose2D
#
# ############################################################

type Transpose2DGate*[TT] {.final.} = ref object of Gate[TT]

proc transpose2d_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  result = newDiffs[TT](1)
  result[0] = payload.variable.grad.transpose2d

proc transpose2d_cache[TT](result: Variable[TT], a: Variable[TT]) =
  var gate: Transpose2DGate[TT]
  new gate
  result.grad = zeros_like result.value
  result.requires_grad = true
  register_node(
    "Transpose2D",
    gate,
    transpose2d_backward_ag[TT],
    result,
    a
  )

proc transpose2d*[TT](a: Variable[TT]): Variable[TT] =
  ## Transpose the last two dimensions of a Variable.
  ## For 2D matrices, this is identical to standard transpose.
  new result
  result.context = a.context
  result.value = a.value.transpose2d
  if a.is_grad_needed:
    result.transpose2d_cache(a)

# ############################################################
#
#                   Permute
#
# ############################################################

type PermuteGate*[TT] {.final.} = ref object of Gate[TT]
  inv_dims: seq[int]

proc permute_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = PermuteGate[TT](self)
  result = newDiffs[TT](1)
  result[0] = payload.variable.grad.permute(self.inv_dims)

proc permute_cache[TT](result: Variable[TT], a: Variable[TT], dims: openArray[int]) =
  var gate: PermuteGate[TT]
  new gate
  gate.inv_dims = newSeq[int](dims.len)
  for i, p in dims:
    gate.inv_dims[p] = i

  result.grad = zeros_like result.value
  result.requires_grad = true
  register_node(
    "Permute",
    gate,
    permute_backward_ag[TT],
    result,
    a
  )

proc permute*[TT](a: Variable[TT], dims: varargs[int]): Variable[TT] =
  ## Permute dimensions of a Variable
  new result
  result.context = a.context
  result.value = a.value.permute(dims)
  if a.is_grad_needed:
    result.permute_cache(a, dims)

