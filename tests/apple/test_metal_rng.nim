# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Exercises native Philox known answers, reservations and same-backend replay.
import std/unittest
import ../../src/arraymancer

proc uniforms(words: array[4, uint32]): Tensor[float32] =
  # Random123's words become exact float32 uniforms; this is not another RNG.
  var values: array[4, float32]
  for i, word in words: values[i] = float32(word shr 8) * (1'f32 / 16777216'f32)
  values.toTensor

suite "Metal Philox random generation":
  test "Published zero and nonzero key/counter known answers run on GPU":
    # https://github.com/DEShawResearch/random123/blob/9545ff6413f258be2f04c1d319d99aaef7521150/tests/kat_vectors
    let generator = newMetalGenerator(0)
    let zero = randMetal([4].toMetadata, generator)
    check zero.cpu == uniforms([0x6627e8d5'u32, 0xe169c58d'u32, 0xbc57ac4c'u32, 0x9b00dbd8'u32])
    let pi = MetalRngState(version: 1, seed: 0x299f31d0a4093822'u64,
      counter: [0x243f6a88'u32, 0x85a308d3'u32, 0x13198a2e'u32, 0x03707344'u32])
    setRngState(generator, pi)
    let nonzero = randMetal([4].toMetadata, generator)
    check nonzero.cpu == uniforms([0xd16cfe09'u32, 0x94fdcceb'u32, 0x5001e420'u32, 0x24126ea1'u32])
    check nonzero.storage.device.registryId == generator.device.registryId
    var evidence: AMEvidence
    checkMetal metalApi().evidence(nonzero.storage.owner.handle, addr evidence)
    check evidence.completed == 1 and evidence.kind == 2 and evidence.kernelBits == 200
    check evidence.transferBytes == 16
    check evidence.cpuFallbackCount == 0

  test "Restored state redraws exact values into distinct native storage":
    let generator = newMetalGenerator(0x123456789abcdef0'u64)
    let saved = getRngState(generator)
    let first = randMetal([3, 3].toMetadata, generator)
    check getRngState(generator).counter == [3'u32, 0, 0, 0]
    setRngState(generator, saved)
    let second = randMetal(3, 3, generator = generator)
    check first.storage != second.storage
    check first.shape == second.shape
    let values = first.cpu
    check values == second.cpu
    for value in values:
      check value >= 0'f32 and value < 1'f32

  test "Pending reservations are visible without waiting for a blocked queue":
    let generator = newMetalGenerator(42)
    let saved = getRngState(generator)
    let stream = newMetalStream(generator.device)
    let gate = newMetalEvent(generator.device)
    var released = false
    defer:
      if not released: signalMetalEvent(gate, 1)
      metalSynchronize(stream)
    wait(stream, gate, 1)
    var first: MetalTensor[float32]
    withMetalStream(stream):
      first = randMetal([5].toMetadata, generator)
      check getRngState(generator).counter == [2'u32, 0, 0, 0]
      discard randMetal([4].toMetadata, generator)
    check getRngState(generator).counter == [3'u32, 0, 0, 0]
    check not metalStreamIsComplete(stream)
    var pending: AMEvidence
    checkMetal metalApi().evidence(first.storage.owner.handle, addr pending)
    check pending.completed == 0
    setRngState(generator, saved)
    let replay = randMetal([5].toMetadata, generator)
    let replayValues = replay.cpu
    check not metalStreamIsComplete(stream)
    signalMetalEvent(gate, 1)
    released = true
    metalSynchronize(stream)
    check first.cpu == replayValues

  test "Scalar and empty draws reserve one and zero blocks respectively":
    let generator = newMetalGenerator(7)
    var scalarShape: Metadata
    let scalar = randMetal(scalarShape, generator)
    check scalar.rank == 0 and scalar.size == 1
    check getRngState(generator).counter == [1'u32, 0, 0, 0]
    let saved = getRngState(generator)
    let empty = randMetal([2, 0, 3].toMetadata, generator)
    check empty.shape == [2, 0, 3].toMetadata and empty.size == 0
    check getRngState(generator) == saved
    check empty.cpu.size == 0
    metalSynchronize(generator.device)

  test "Counter carry is preserved and overflow or incompatible states error":
    let generator = newMetalGenerator(1)
    let carried = MetalRngState(version: 1, seed: 1,
      counter: [0'u32, 0, 6, 7])
    var state = carried
    state.counter = [high(uint32), high(uint32), 5'u32, 7'u32]
    setRngState(generator, state)
    let across = randMetal([8].toMetadata, generator)
    check getRngState(generator).counter == [1'u32, 0, 6, 7]
    setRngState(generator, carried)
    let afterCarry = randMetal([4].toMetadata, generator)
    check across.metalView([4].toMetadata, [1].toMetadata, 4).cpu == afterCarry.cpu
    state = getRngState(generator)
    var invalid = state
    invalid.version = 2
    expect MetalRngStateError: setRngState(generator, invalid)
    check getRngState(generator) == state
    state.counter = [high(uint32) - 1, high(uint32), high(uint32), high(uint32)]
    setRngState(generator, state)
    expect MetalRngStateError: discard randMetal([8].toMetadata, generator)
    check getRngState(generator) == state

  test "Default generator is shared and entropy reseeding resets reservation state":
    let first = defaultMetalGenerator()
    let second = defaultMetalGenerator()
    let previous = getRngState(first)
    defer: setRngState(first, previous)
    check first.owner.handle == second.owner.handle
    manualSeed(first, 19)
    check getRngState(second).seed == 19
    discard randMetal([4].toMetadata, first)
    let entropySeed = seed(second)
    let reseeded = getRngState(first)
    check reseeded.seed == entropySeed
    check reseeded.counter == [0'u32, 0, 0, 0]
    metalSynchronize(first.device)
