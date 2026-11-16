# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises real event dependencies, queue-local fences and retained pending tensor owners.
import std/[unittest, locks]
import ../../src/arraymancer

type ThreadSelection = object
  lock: Lock
  condition: Cond
  ready, released, valid: bool
  identity: uint64

proc threadProbe(state: ptr ThreadSelection) {.thread.} =
  try:
    let own = newMetalStream()
    setMetalStream(own)
    withLock state.lock:
      state.identity = own.identity
      state.ready = true
      signal(state.condition)
      while not state.released: wait(state.condition, state.lock)
    let output = [1'f32].toTensor.metal + 1'f32
    metalSynchronize(own)
    var evidence: AMEvidence
    checkMetal metalApi().evidence(output.storage.owner.handle, addr evidence)
    state.valid = currentMetalStream().identity == own.identity and
      evidence.completed == 1 and evidence.kind == 2
  except CatchableError:
    withLock state.lock:
      state.valid = false
      state.ready = true
      signal(state.condition)

suite "Metal streams and pending owners":
  test "Selection is thread-local and restored across exceptions":
    let previous = currentMetalStream()
    defer: setMetalStream(previous)
    let main = newMetalStream()
    check main.identity != defaultMetalStream().identity
    check defaultMetalStream().identity == defaultMetalStream().identity
    setMetalStream(main)
    let nested = newMetalStream()
    expect ValueError:
      withMetalStream(nested):
        check currentMetalStream().identity == nested.identity
        raise newException(ValueError, "intentional stream scope exit")
    check currentMetalStream().identity == main.identity
    let mainGate = newMetalEvent()
    wait(main, mainGate, 1)
    defer:
      signalMetalEvent(mainGate, 1)
      metalSynchronize(main)
    var state: ThreadSelection
    initLock(state.lock)
    initCond(state.condition)
    var worker: Thread[ptr ThreadSelection]
    createThread(worker, threadProbe, addr state)
    try:
      withLock state.lock:
        while not state.ready: wait(state.condition, state.lock)
      check currentMetalStream().identity == main.identity
      check state.identity != main.identity
      withMetalStream(nested):
        check currentMetalStream().identity == nested.identity
    finally:
      withLock state.lock:
        state.released = true
        signal(state.condition)
      joinThread(worker)
      deinitCond(state.condition)
      deinitLock(state.lock)
    check state.valid
    expect MetalError: setMetalStream(nil)
  test "Blocked producer and explicit event handoff retain dropped inputs":
    let producer = newMetalStream()
    let consumer = newMetalStream()
    let blocker = newMetalEvent()
    let ready = newMetalEvent()
    var released = false
    defer:
      if not released: signalMetalEvent(blocker, 1)
      metalSynchronize(producer)
      metalSynchronize(consumer)
    wait(producer, blocker, 1)
    var alias: MetalTensor[float32]
    block:
      withMetalStream(producer):
        let original = [1'f32, 2].toTensor.metal
        let generated = original + 1'f32
        alias = generated.metalView([2].toMetadata, [1].toMetadata, 0)
        record(ready, producer)
    var output: MetalTensor[float32]
    withMetalStream(consumer):
      wait(consumer, ready)
      output = alias * 2'f32
    metalSynchronize(defaultMetalStream())
    var pending: AMEvidence
    checkMetal metalApi().evidence(alias.storage.owner.handle, addr pending)
    check pending.completed == 0
    checkMetal metalApi().evidence(output.storage.owner.handle, addr pending)
    check pending.completed == 0
    check not metalStreamIsComplete(producer)
    check not metalStreamIsComplete(consumer)
    signalMetalEvent(blocker, 1)
    released = true
    metalSynchronize(consumer)
    check output.cpu == [4'f32, 6].toTensor
    check alias.cpu == [2'f32, 3].toTensor
    metalSynchronize(producer)
    check metalStreamIsComplete(producer)
    check metalStreamIsComplete(consumer)
  test "A stream fence leaves an unrelated future-event queue blocked":
    let blocked = newMetalStream()
    let active = newMetalStream()
    let event = newMetalEvent()
    var released = false
    defer:
      if not released: signalMetalEvent(event, 1)
      metalSynchronize(blocked)
    wait(blocked, event, 1)
    var output: MetalTensor[float32]
    withMetalStream(active):
      expect ValueError:
        withMetalStream(blocked):
          raise newException(ValueError, "restore a nondefault native stream")
      output = [3'f32, 4].toTensor.metal * 2'f32
    metalSynchronize(active)
    check not metalStreamIsComplete(blocked)
    var evidence: AMEvidence
    checkMetal metalApi().evidence(output.storage.owner.handle, addr evidence)
    check evidence.completed == 1
    signalMetalEvent(event, 1)
    released = true
    metalSynchronize(metalDevice())
    check output.cpu == [6'f32, 8].toTensor
    check metalStreamIsComplete(blocked)
    check metalStreamIsComplete(active)
    expect MetalError: signalMetalEvent(event, 1)
    expect MetalError: signalMetalEvent(event, 0)
    let unrecorded = newMetalEvent()
    expect MetalError: wait(active, unrecorded)
  test "Implicit event waits capture the recorded timeline value":
    let producer = newMetalStream()
    let consumer = newMetalStream()
    let firstGate = newMetalEvent()
    let laterGate = newMetalEvent()
    let timeline = newMetalEvent()
    var firstReleased, laterReleased: bool
    defer:
      if not firstReleased: signalMetalEvent(firstGate, 1)
      if not laterReleased: signalMetalEvent(laterGate, 1)
      metalSynchronize(producer)
      metalSynchronize(consumer)
    wait(producer, firstGate, 1)
    check record(timeline, producer) == 1
    wait(consumer, timeline)
    wait(producer, laterGate, 1)
    check record(timeline, producer) == 2
    signalMetalEvent(firstGate, 1)
    firstReleased = true
    metalSynchronize(consumer)
    check not metalStreamIsComplete(producer)
    signalMetalEvent(laterGate, 1)
    laterReleased = true
    metalSynchronize(producer)
    signalMetalEvent(timeline, 4)
    check record(timeline, producer) == 5
    metalSynchronize(producer)
    expect MetalError: signalMetalEvent(timeline, 4)
