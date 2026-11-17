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

import std / [typetraits, macros, sets]

# ############################################################
#
#                        Datatypes
#
# ############################################################

# Design tradeoff:
#
# Nim default GC is deferred reference counting, meaning it doesn't deal well with
# cyclic references where A refers to B and B refers to A.
#
# To solve that it must use the dreaded mark-and-sweep, i.e. stop the world
# copy all live objects into a new location, everything that was not copied
# was unused.
#
# Deep learning is already extremely memory-bandwidth intensive.
# We can't use a direct pointer to the object as the GC can move it (https://github.com/mratsim/Arraymancer/pull/329)
# so we need a pointer to the ref the GC is holding to break the ref cycle.
#
# The cost is that using objects through that requires a double pointer indirection,
# so much more cache misses.
# Though it's certainly much less costly than mark and sweep.
#
# This is pending https://github.com/nim-lang/Nim/issues/9974

type
  Context*[TT] = ref object # {.acyclic.}
    ## An autograd context is a record of operations or layers.
    ## It holds the following fields:
    ##   - ``nodes``: This records the list of operations(``Node``) applied in the context
    ##   - ``no_grad``: This disable tracing the list of operations altogether.
    ##     This is useful to save memory when you don't need the gradient
    ##     (for validation or prediction for example)
    ##
    ## A context is also called a tape or a Wengert list.
    ##
    ## Backpropagation releases traversed operations unless ``retain_graph`` is set.
    nodes: seq[Node[TT]]
    no_grad: bool

  Variable*[TT] = ref object # {.acyclic.}
    ## A variable is a wrapper for Tensors that tracks operations applied to it.
    ## It consists of:
    ##    - A weak reference to a record of operations ``context``
    ##    - The tensor being tracked ``value``
    ##    - The gradient of the tensor ``grad``
    ##    - a flag that indicates if gradient is needed
    context*: Context[TT]
      # Variables shouldn't own their Context
    value*: TT
    grad*: TT
    requires_grad*: bool
    has_grad_fn: bool
    graph_freed: bool

  Gate*[TT] = ref object of RootObj # {.acyclic.}
    ## Base operator or layer. You can describe your custom operations or layers
    ## by inheriting from Gate and add a forward and optionally a backward method.
    ## Each operations should set the number of gradients produced during backpropagation.
    ## Additional fields specific to the operations like weights or inputs cache should be added too.

  PayloadKind* = enum
    pkVar, pkSeq
  Payload*[TT] = object
    case kind*: PayloadKind
    of pkVar: variable*: Variable[TT]
    of pkSeq: sequence*: seq[Variable[TT]]

  Backward*[TT] = proc(self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] {.nimcall.}
    ## ⚠️ Warning: make sure the identifier is not overloaded
    ## https://github.com/nim-lang/Nim/issues/9997

  Node[TT] = object
    ## A node consist of:
    ##   - The description of the operator or layer (``gate``)
    ##   - The corresponding proc that handles backpropagation
    ##   - A weak reference to the ``parents`` VariableObj
    ##   - The actual value of the node (``payload``)
    gate: Gate[TT]
    backward: Backward[TT]
    parents: Parents[TT]
    payload: Payload[TT]
    when defined(debug):
      name: string

  Parents[TT] = seq[Variable[TT]]
    # Children nodes shouldn't own their parents
  SmallDiffs*[TT] = seq[TT]

# ############################################################
#
#                      Debugging
#
# ############################################################

func debug[TT](ctx: Context[TT]) =
  ## Debug the autograd context

  debugEcho "\n######"
  for i, node in ctx.nodes:
    # strformat doesn't work in generics
    # var s = &"Node {i:>4}: {node.name:>12} - "
    var s = "Node " & $i & ": " & node.name & " - "
    if node.parents.len <= 1:
      s &= $node.parents[0].value.shape
    else:
      s &= '('
      for p, parent in node.parents:
        if p != 0:
          s &= ", "
        s &= $parent.value.shape
      s &= ')'
    s &= " ===>> "
    case node.payload.kind
    of pkVar: s &= $node.payload.variable.value.shape
    of pkSeq:
      s &= "( "
      for p, payload in node.payload.sequence:
        if p != 0:
          s &= ", "
        s &= $payload.value.shape
      s &= ")"
    debugEcho s

# ############################################################
#
#                Autograd procedure
#
# ############################################################

func newContext*(TT: typedesc): Context[TT] =
  ## Initialize a context
  new result
  result.nodes = newSeq[Node[TT]]()

proc variable*[TT](ctx: Context[TT], value: TT, requires_grad = false): Variable[TT] =
  ## Wrap a variable to the context
  ## T is a Tensor[T, CudaTensor[T] or scalar T
  new result
  result.context = ctx
  result.value = value
  if requires_grad:
    result.requires_grad = true
    result.grad = value.zeros_like


template push[TT](ctx: Context[TT], node: Node[TT]) =
  ## Append a new operation to the context
  ctx.nodes.add(node)


func register_node*[TT](
        name: static string,
        gate: Gate[TT],
        backward: Backward[TT],
        result: Variable[TT] or seq[Variable[TT]],
        parents: varargs[Variable[TT]]) =
  ## Add an operation / gate as a new node in the computation graph
  var node: Node[TT]

  node.gate = gate
  node.backward = backward
  node.parents = @parents

  when result is Variable:
    node.payload = Payload[TT](kind: pkVar, variable: result)
    result.has_grad_fn = true
  else:
    node.payload = Payload[TT](kind: pkSeq, sequence: result)
    for variable in result:
      variable.has_grad_fn = true

  when defined(debug):
    node.name = name

  parents[0].context.push node

template no_grad_mode*(ctx: Context, body: untyped): untyped =
  ## Within this block, the context will not track the operations applied
  ## to each Variable.
  ##
  ## This should be used for validation or prediction to optimize memory.
  ## The previous tracking mode is restored on every exit, including exceptions.
  let context = ctx
  let prev_state = context.no_grad
  context.no_grad = true
  try:
    body
  finally:
    context.no_grad = prev_state

func is_grad_needed*(v: Variable): bool {.inline.} =
  ## Depending on the input variable and its context no_grad_mode,
  ## returns true if gradient computation is needed and false otherwise
  v.requires_grad and not v.context.no_grad

func check_ctx*(a, b: Variable) {.inline.} =
  if unlikely(a.context != b.context):
    raise newException(ValueError, "You cannot combine variable from different contexts")

proc resetIntermediate[TT](v: Variable[TT]) =
  ## Reuse owned gradient buffers between retained traversals.
  when compiles(v.grad.setZero()):
    v.grad.setZero()
  elif compiles(v.grad.fill(getSubType(TT)(0))):
    v.grad.fill(getSubType(TT)(0))
  else:
    v.grad = v.value.zeros_like

proc backpropagate[TT](v: Variable[TT], gradient: TT, retain_graph: bool) =
  ## Traverse only output ancestors, leaving independent graphs intact.
  if unlikely(not v.requires_grad):
    raise newException(ValueError, "Operations leading to this variable were not fully traced.\nDid you forget to set a `requires_grad` or to disable the context `no_grad_mode`?")
  if v.graph_freed:
    raise newException(ValueError, "The backward graph has been released; use retain_graph on the preceding backward call")
  if not v.has_grad_fn:
    v.grad += gradient
    return

  var needed = initHashSet[pointer]()
  needed.incl(cast[pointer](v))
  var execution: seq[int]
  for i in countdown(v.context.nodes.high, 0):
    template node: untyped = v.context.nodes[i]
    var reachable = false
    case node.payload.kind
    of pkVar:
      reachable = cast[pointer](node.payload.variable) in needed
    of pkSeq:
      for output in node.payload.sequence:
        if cast[pointer](output) in needed:
          reachable = true
          break
    if reachable:
      execution.add(i)
      for parent in node.parents:
        if parent.requires_grad:
          if parent.graph_freed:
            raise newException(ValueError, "A required backward graph has already been released")
          needed.incl(cast[pointer](parent))

  # Validate the entire traversal before changing any gradients.
  for i in execution:
    case v.context.nodes[i].payload.kind
    of pkVar:
      let output = v.context.nodes[i].payload.variable
      if output != v: output.resetIntermediate()
    of pkSeq:
      for output in v.context.nodes[i].payload.sequence:
        if output != v: output.resetIntermediate()
  v.grad = gradient

  for i in execution:
    template node: untyped = v.context.nodes[i]
    let diffs = node.backward(node.gate, node.payload)
    for j, diff in diffs:
      let parent = node.parents[j]
      if parent.requires_grad:
        parent.grad += diff

  if not retain_graph:
    # Compact once; indices in execution are in descending tape order.
    var next = execution.high
    var destination = 0
    for i in 0 ..< v.context.nodes.len:
      if next >= 0 and execution[next] == i:
        case v.context.nodes[i].payload.kind
        of pkVar: v.context.nodes[i].payload.variable.graph_freed = true
        of pkSeq:
          for output in v.context.nodes[i].payload.sequence:
            output.graph_freed = true
        dec next
      else:
        if destination != i:
          v.context.nodes[destination] = move(v.context.nodes[i])
        inc destination
    v.context.nodes.setLen(destination)

proc backprop*[TT](v: Variable[TT], gradient: TT, retain_graph = false) =
  ## Compute a vector-Jacobian product with an explicit output gradient.
  ##
  ## ``gradient`` must have exactly the output shape; broadcasting is not allowed.
  ## Its values are copied so accumulating gradients cannot mutate the caller's
  ## tensor. Leaf gradients accumulate; intermediate gradients are recomputed.
  ## ``retain_graph`` preserves traversed operations for another backward call.
  ## Otherwise those operations are released and cannot be traversed again.
  ## Independent graphs in the same context are left intact.
  if gradient.shape != v.value.shape:
    raise newException(ValueError, "backprop gradient shape must match the output shape")
  v.backpropagate(gradient.clone, retain_graph)

proc backprop*[TT](v: Variable[TT], retain_graph = false) =
  ## Differentiate the sum of output elements using an implicit gradient of ones.
  ##
  ## Unlike PyTorch, non-scalar outputs remain supported for compatibility.
  ## Use the explicit-gradient overload for a vector-Jacobian product.
  ## Leaf gradients accumulate; intermediate gradients are recomputed.
  ## ``retain_graph`` preserves traversed operations for another backward call;
  ## independent graphs in the same context are always left intact.
  v.backpropagate(v.value.ones_like, retain_graph)

func newParents*[TT](num: Natural): Parents[TT] {.inline.} =
  newSeq[Variable[TT]](num)

func newDiffs*[TT](num: Natural): SmallDiffs[TT] {.inline.} =
  newSeq[TT](num)
