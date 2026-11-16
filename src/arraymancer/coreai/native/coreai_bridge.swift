// Copyright 2026 Arraymancer contributors. Apache-2.0.
// Core AI objects cross the C ABI only as retained opaque handles. No Swift
// storage view or caller-owned host pointer survives its synchronous closure.
import ArraymancerAppleABI
import CoreAI
import CryptoKit
import Darwin
import Foundation
import Synchronization

private enum ABIError: Error {
  case failure(Int32, String)

  var code: Int32 {
    if case let .failure(code, _) = self { return code }
    return 103
  }

  var message: String {
    if case let .failure(_, message) = self { return message }
    return "Core AI native operation failed"
  }
}

private func abiError(_ code: Int32, _ message: String) -> ABIError {
  .failure(code, message)
}

private func statusOK() -> AMStatus {
  AMStatus(code: 0, message: nil)
}

private func statusFailure(_ code: Int32, _ message: String) -> AMStatus {
  AMStatus(code: code, message: strdup(message))
}

private func statusFailure(_ error: Error, fallback: Int32 = 103) -> AMStatus {
  if let error = error as? ABIError {
    return statusFailure(error.code, error.message)
  }
  return statusFailure(fallback, String(describing: error))
}

private func ownedCString(_ value: String) -> UnsafeMutablePointer<CChar>? {
  strdup(value)
}

private func retainedHandle(_ value: AnyObject) -> UnsafeMutableRawPointer {
  Unmanaged.passRetained(value).toOpaque()
}

private func objectHandle<T: AnyObject>(
  _ pointer: UnsafeMutableRawPointer?, as type: T.Type
) throws -> T {
  guard let pointer else {
    throw abiError(103, "Core AI handle is nil")
  }
  let object = Unmanaged<AnyObject>.fromOpaque(pointer).takeUnretainedValue()
  guard let value = object as? T else {
    throw abiError(103, "Core AI handle has the wrong native type")
  }
  return value
}

private let invocationCounter = Mutex<UInt64>(1)

private func nextInvocationID() -> UInt64 {
  invocationCounter.withLock { value in
    let result = value
    value = value == UInt64.max ? 1 : value + 1
    return result
  }
}

@available(macOS 27.0, *)
private func unitMask(_ units: Set<ComputeUnitKind>) -> UInt32 {
  var mask: UInt32 = 0
  if units.contains(.cpu) { mask |= 1 }
  if units.contains(.gpu) { mask |= 2 }
  if units.contains(.neuralEngine) { mask |= 4 }
  return mask
}

@available(macOS 27.0, *)
private func optionsForPolicy(_ policy: Int32) throws -> SpecializationOptions {
  switch policy {
  case 0: return .default
  case 1: return .cpuOnly
  case 2: return SpecializationOptions(preferredComputeUnitKind: .gpu)
  case 3: return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
  default: throw abiError(102, "unknown Core AI compute policy \(policy)")
  }
}

@available(macOS 27.0, *)
private func unitIndex(_ unit: ComputeUnitKind?) -> Int32 {
  guard let unit else { return -1 }
  switch unit {
  case .cpu: return 0
  case .gpu: return 1
  case .neuralEngine: return 2
  @unknown default: return -1
  }
}

@available(macOS 27.0, *)
private struct EvidenceContext: Sendable {
  let invocationID: UInt64
  let kind: Int32
  let requestedPolicy: Int32
  let availableMask: UInt32
  let allowedMask: UInt32
  let preferredUnit: Int32
  let functionName: String
  var assetDigest: String
  var architecture: String
}

@available(macOS 27.0, *)
private struct JobFailure: Sendable {
  let code: Int32
  let message: String
}

@available(macOS 27.0, *)
private final class CoreModelHandle: @unchecked Sendable {
  let model: AIModel
  let url: URL
  let options: SpecializationOptions
  let requestedPolicy: Int32
  let assetDigest: String
  let architecture: String
  let availableMask: UInt32
  let allowedMask: UInt32
  let preferredUnit: Int32

  init(
    model: AIModel,
    url: URL,
    options: SpecializationOptions,
    requestedPolicy: Int32,
    assetDigest: String,
    architecture: String,
    availableMask: UInt32,
    allowedMask: UInt32,
    preferredUnit: Int32
  ) {
    self.model = model
    self.url = url
    self.options = options
    self.requestedPolicy = requestedPolicy
    self.assetDigest = assetDigest
    self.architecture = architecture
    self.availableMask = availableMask
    self.allowedMask = allowedMask
    self.preferredUnit = preferredUnit
  }
}

@available(macOS 27.0, *)
private final class CoreFunctionHandle: @unchecked Sendable {
  let model: CoreModelHandle
  let function: InferenceFunction
  let name: String

  init(model: CoreModelHandle, function: InferenceFunction, name: String) {
    self.model = model
    self.function = function
    self.name = name
  }
}

@available(macOS 27.0, *)
private final class CoreArrayHandle: @unchecked Sendable {
  let array: NDArray
  let name: String
  let minimumBytes: UInt64
  private let hostCopyCounter = Mutex<UInt64>(0)

  init(array: NDArray, name: String = "", minimumBytes: UInt64) {
    self.array = array
    self.name = name
    self.minimumBytes = minimumBytes
  }

  func recordHostCopy(_ bytes: UInt64) {
    hostCopyCounter.withLock { total in
      let (next, overflow) = total.addingReportingOverflow(bytes)
      total = overflow ? UInt64.max : next
    }
  }

  var hostCopyBytes: UInt64 {
    hostCopyCounter.withLock { $0 }
  }
}

@available(macOS 27.0, *)
private enum JobOutcome: Sendable {
  case model(CoreModelHandle)
  case outputs([String: CoreArrayHandle])
  case failure(JobFailure)
}

@available(macOS 27.0, *)
private final class CoreJobHandle: @unchecked Sendable {
  private let condition = NSCondition()
  private var outcome: JobOutcome?
  private var context: EvidenceContext

  init(context: EvidenceContext) {
    self.context = context
  }

  func finish(
    _ outcome: JobOutcome,
    assetDigest: String? = nil,
    architecture: String? = nil
  ) {
    condition.lock()
    self.outcome = outcome
    if let assetDigest { context.assetDigest = assetDigest }
    if let architecture { context.architecture = architecture }
    condition.broadcast()
    condition.unlock()
  }

  func wait(timeoutMilliseconds: UInt32) -> JobOutcome? {
    condition.lock()
    defer { condition.unlock() }
    if outcome == nil {
      let deadline = Date(timeIntervalSinceNow: Double(timeoutMilliseconds) / 1_000.0)
      while outcome == nil {
        if !condition.wait(until: deadline) { break }
      }
    }
    return outcome
  }

  func snapshot() -> (JobOutcome?, EvidenceContext) {
    condition.lock()
    defer { condition.unlock() }
    return (outcome, context)
  }
}

private struct HostTensorLayout {
  let shape: [Int64]
  let strides: [Int64]
  let offset: Int64
  let capacity: UInt64
  let elementCount: Int
  let elementWidth: Int
  let byteCount: UInt64
}

private func int64Tuple(_ values: [Int64]) -> (Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64) {
  func value(_ index: Int) -> Int64 {
    index < values.count ? values[index] : 0
  }
  return (value(0), value(1), value(2), value(3), value(4), value(5), value(6), value(7))
}

private func int64Values(
  _ tuple: (Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64)
) -> [Int64] {
  [tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5, tuple.6, tuple.7]
}

private func descriptorShape(_ descriptor: AMTensorDescriptor) -> [Int64] {
  int64Values(descriptor.shape)
}

private func descriptorStrides(_ descriptor: AMTensorDescriptor) -> [Int64] {
  int64Values(descriptor.strides)
}

private func checkedAdd(_ left: Int64, _ right: Int64) throws -> Int64 {
  let (value, overflow) = left.addingReportingOverflow(right)
  guard !overflow else { throw abiError(102, "tensor offset arithmetic overflow") }
  return value
}

private func checkedMultiply(_ left: Int64, _ right: Int64) throws -> Int64 {
  let (value, overflow) = left.multipliedReportingOverflow(by: right)
  guard !overflow else { throw abiError(102, "tensor offset arithmetic overflow") }
  return value
}

private func elementCount(_ shape: [Int64]) throws -> Int {
  guard shape.allSatisfy({ $0 >= 0 }) else {
    throw abiError(102, "tensor dimensions must be nonnegative")
  }
  if shape.contains(0) { return 0 }
  var count: UInt64 = 1
  for dimension in shape {
    let (next, overflow) = count.multipliedReportingOverflow(by: UInt64(dimension))
    guard !overflow, next <= UInt64(Int.max) else {
      throw abiError(102, "tensor element count overflows addressable memory")
    }
    count = next
  }
  return Int(count)
}

private func byteCount(elements: Int, width: Int) throws -> UInt64 {
  guard elements >= 0, width > 0 else {
    throw abiError(102, "invalid tensor byte count")
  }
  let (bytes, overflow) = UInt64(elements).multipliedReportingOverflow(by: UInt64(width))
  guard !overflow, bytes <= UInt64(Int.max) else {
    throw abiError(102, "tensor byte count exceeds addressable memory")
  }
  return bytes
}

private func abiElementWidth(_ dtype: Int32) throws -> Int {
  switch dtype {
  case 1, 6: return MemoryLayout<UInt32>.size
  default: throw abiError(102, "Core AI host arrays support only float32 and int32")
  }
}

@available(macOS 27.0, *)
private func scalarTypeForABI(_ dtype: Int32) throws -> NDArray.ScalarType {
  switch dtype {
  case 1: return .float32
  case 6: return .int32
  default: throw abiError(102, "Core AI host arrays support only float32 and int32")
  }
}

@available(macOS 27.0, *)
private func abiType(for scalarType: NDArray.ScalarType) -> Int32? {
  switch scalarType {
  case .float32: return 1
  case .int32: return 6
  default: return nil
  }
}

private func validateHostLayout(_ descriptor: AMTensorDescriptor) throws -> HostTensorLayout {
  guard (0...8).contains(descriptor.rank) else {
    throw abiError(102, "tensor rank must be between zero and eight")
  }
  guard descriptor.capacity <= UInt64(Int64.max) else {
    throw abiError(102, "tensor capacity exceeds signed addressable range")
  }
  let rank = Int(descriptor.rank)
  let shape = Array(descriptorShape(descriptor).prefix(rank))
  let strides = Array(descriptorStrides(descriptor).prefix(rank))
  let count = try elementCount(shape)
  let width = try abiElementWidth(descriptor.dtype)
  let bytes = try byteCount(elements: count, width: width)
  let capacity = descriptor.capacity

  if count == 0 {
    let start = try byteOffset(descriptor.offset, width: width)
    guard UInt64(start) <= capacity else {
      throw abiError(102, "empty tensor offset is outside storage capacity")
    }
  } else {
    var minimum = descriptor.offset
    var maximum = descriptor.offset
    for axis in 0..<rank {
      let extent = try checkedMultiply(shape[axis] - 1, strides[axis])
      if extent < 0 {
        minimum = try checkedAdd(minimum, extent)
      } else {
        maximum = try checkedAdd(maximum, extent)
      }
    }
    guard minimum >= 0, maximum >= minimum else {
      throw abiError(102, "tensor view reaches outside its bounded storage")
    }
    let end = try byteRangeEnd(maximum, width: width)
    guard UInt64(end) <= capacity else {
      throw abiError(102, "tensor view reaches outside its bounded storage")
    }
  }
  return HostTensorLayout(
    shape: shape,
    strides: strides,
    offset: descriptor.offset,
    capacity: capacity,
    elementCount: count,
    elementWidth: width,
    byteCount: bytes
  )
}

private func byteOffset(_ elementOffset: Int64, width: Int) throws -> Int {
  guard elementOffset >= 0 else {
    throw abiError(102, "tensor view contains a negative storage address")
  }
  let (byteOffset, overflow) = elementOffset.multipliedReportingOverflow(by: Int64(width))
  guard !overflow, let result = Int(exactly: byteOffset) else {
    throw abiError(102, "tensor byte offset exceeds addressable memory")
  }
  return result
}

private func byteRangeEnd(_ elementOffset: Int64, width: Int) throws -> Int {
  let start = try byteOffset(elementOffset, width: width)
  let (end, overflow) = start.addingReportingOverflow(width)
  guard !overflow else {
    throw abiError(102, "tensor byte range exceeds addressable memory")
  }
  return end
}

private func logicalElementOffset(
  _ linearIndex: Int,
  shape: [Int],
  strides: [Int64],
  baseOffset: Int64
) throws -> Int64 {
  guard shape.count == strides.count else {
    throw abiError(102, "tensor shape and stride ranks differ")
  }
  var remainder = linearIndex
  var offset = baseOffset
  if !shape.isEmpty {
    for axis in stride(from: shape.count - 1, through: 0, by: -1) {
      let dimension = shape[axis]
      guard dimension > 0 else { throw abiError(102, "zero-sized tensor was indexed") }
      let coordinate = remainder % dimension
      remainder /= dimension
      offset = try checkedAdd(offset, checkedMultiply(Int64(coordinate), strides[axis]))
    }
  }
  return offset
}

private func isCContiguous(_ shape: [Int], _ strides: [Int64]) -> Bool {
  var expected: Int64 = 1
  for axis in shape.indices.reversed() {
    if shape[axis] > 1 && strides[axis] != expected { return false }
    let (next, overflow) = expected.multipliedReportingOverflow(by: Int64(shape[axis]))
    if overflow { return false }
    expected = next
  }
  return true
}

private func intShape(_ shape: [Int64]) throws -> [Int] {
  try shape.map { dimension in
    guard let value = Int(exactly: dimension), value >= 0 else {
      throw abiError(102, "tensor dimension is not addressable")
    }
    return value
  }
}

@available(macOS 27.0, *)
private func sdkArrayMinimumBytes(_ array: NDArray) throws -> UInt64 {
  guard abiType(for: array.scalarType) != nil else { return 0 }
  let (_, _, byteCount) = try validateSdkArrayLayout(array.shape, array.strides)
  return UInt64(byteCount)
}

@available(macOS 27.0, *)
private func makeHostTensor(_ host: UnsafeRawPointer?, descriptor: AMTensorDescriptor) throws -> CoreArrayHandle {
  let layout = try validateHostLayout(descriptor)
  if layout.elementCount > 0, host == nil {
    throw abiError(102, "nonempty host tensor has a nil data pointer")
  }
  let shape = try intShape(layout.shape)
  var array = NDArray(shape: shape, scalarType: try scalarTypeForABI(descriptor.dtype))
  guard array.interleaveLayout == nil else {
    throw abiError(102, "interleaved Core AI host-import layouts are unsupported")
  }
  let (_, sdkStrides, _) = try validateSdkArrayLayout(shape, array.strides)
  let mutableView = array.mutableRawView()
  try mutableView.withUnsafeMutableBytes { destination, destinationShape, destinationStrides in
    guard destinationShape.count == shape.count, destinationStrides.count == sdkStrides.count else {
      throw abiError(102, "Core AI allocated an unexpected host-import layout")
    }
    for axis in shape.indices {
      guard destinationShape[axis] == shape[axis], Int64(destinationStrides[axis]) == sdkStrides[axis] else {
        throw abiError(102, "Core AI allocated an unexpected host-import layout")
      }
    }
    if layout.elementCount == 0 { return }
    guard let host else { return }
    let destinationBase = destination
    if isCContiguous(shape, layout.strides) && isCContiguous(shape, sdkStrides) {
      let sourceBytes = try byteOffset(layout.offset, width: layout.elementWidth)
      memcpy(destinationBase, host.advanced(by: sourceBytes), Int(layout.byteCount))
      return
    }
    for linearIndex in 0..<layout.elementCount {
      let sourceElement = try logicalElementOffset(
        linearIndex,
        shape: shape,
        strides: layout.strides,
        baseOffset: layout.offset
      )
      let targetElement = try logicalElementOffset(
        linearIndex,
        shape: shape,
        strides: sdkStrides,
        baseOffset: 0
      )
      let sourceBytes = try byteOffset(sourceElement, width: layout.elementWidth)
      let targetBytes = try byteOffset(targetElement, width: layout.elementWidth)
      memcpy(destinationBase.advanced(by: targetBytes), host.advanced(by: sourceBytes), layout.elementWidth)
    }
  }
  let result = CoreArrayHandle(array: array, minimumBytes: try sdkArrayMinimumBytes(array))
  result.recordHostCopy(layout.byteCount)
  return result
}

@available(macOS 27.0, *)
private func validateSdkArrayLayout(
  _ shape: [Int],
  _ strides: [Int]
) throws -> (shape: [Int], strides: [Int64], byteCount: Int) {
  guard shape.count == strides.count, shape.count <= 8, shape.allSatisfy({ $0 >= 0 }) else {
    throw abiError(102, "Core AI returned an invalid NDArray layout")
  }
  let count = try elementCount(shape.map(Int64.init))
  let strideValues = strides.map(Int64.init)
  var byteCount = 0
  if count > 0 {
    var minimum: Int64 = 0
    var maximum: Int64 = 0
    for axis in shape.indices {
      let extent = try checkedMultiply(Int64(shape[axis] - 1), strideValues[axis])
      if extent < 0 { minimum = try checkedAdd(minimum, extent) }
      else { maximum = try checkedAdd(maximum, extent) }
    }
    guard minimum >= 0 else {
      throw abiError(102, "Core AI NDArray has a negative base-relative address")
    }
    byteCount = try byteRangeEnd(maximum, width: MemoryLayout<UInt32>.size)
  }
  return (shape, strideValues, byteCount)
}

@available(macOS 27.0, *)
private func copySdkArray(
  _ handle: CoreArrayHandle,
  to host: UnsafeMutableRawPointer?,
  descriptor: AMTensorDescriptor
) throws -> UInt64 {
  let layout = try validateHostLayout(descriptor)
  let array = handle.array
  let actualShape = array.shape
  let actualStrides = array.strides
  guard actualShape.map(Int64.init) == layout.shape else {
    throw abiError(102, "host output shape does not match the Core AI NDArray")
  }
  guard abiType(for: array.scalarType) == descriptor.dtype else {
    throw abiError(102, "host output dtype does not exactly match the Core AI NDArray")
  }
  if layout.elementCount > 0, host == nil {
    throw abiError(102, "nonempty host output has a nil data pointer")
  }
  let (sdkShape, sdkStrides, _) = try validateSdkArrayLayout(actualShape, actualStrides)
  guard array.interleaveLayout == nil else {
    throw abiError(102, "interleaved Core AI host-output layouts are unsupported")
  }
  let view = array.rawView()
  try view.withUnsafeBytes { source, viewShape, viewStrides in
    guard viewShape.count == sdkShape.count, viewStrides.count == sdkStrides.count else {
      throw abiError(102, "Core AI raw view buffer does not match its NDArray metadata")
    }
    for axis in sdkShape.indices {
      guard viewShape[axis] == sdkShape[axis], Int64(viewStrides[axis]) == sdkStrides[axis] else {
        throw abiError(102, "Core AI raw view buffer does not match its NDArray metadata")
      }
    }
    if layout.elementCount == 0 { return }
    guard let host else { return }
    let sourceBase = source
    if isCContiguous(sdkShape, sdkStrides) && isCContiguous(sdkShape, layout.strides) {
      let targetBytes = try byteOffset(layout.offset, width: layout.elementWidth)
      memcpy(host.advanced(by: targetBytes), sourceBase, Int(layout.byteCount))
      return
    }
    for linearIndex in 0..<layout.elementCount {
      let sourceElement = try logicalElementOffset(
        linearIndex,
        shape: sdkShape,
        strides: sdkStrides,
        baseOffset: 0
      )
      let targetElement = try logicalElementOffset(
        linearIndex,
        shape: sdkShape,
        strides: layout.strides,
        baseOffset: layout.offset
      )
      let sourceBytes = try byteOffset(sourceElement, width: layout.elementWidth)
      let targetBytes = try byteOffset(targetElement, width: layout.elementWidth)
      memcpy(host.advanced(by: targetBytes), sourceBase.advanced(by: sourceBytes), layout.elementWidth)
    }
  }
  handle.recordHostCopy(layout.byteCount)
  return layout.byteCount
}

@available(macOS 27.0, *)
private func makeFunctionDescriptor(
  function: CoreFunctionHandle,
  kind: Int32,
  index: UInt32
) throws -> ACArrayDescriptor {
  let descriptor = function.function.descriptor
  let names: [String]
  let valueDescriptor: (String) -> InferenceValue.Descriptor?
  switch kind {
  case 0:
    names = descriptor.inputNames
    valueDescriptor = { descriptor.inputDescriptor(of: $0) }
  case 1:
    names = descriptor.outputNames
    valueDescriptor = { descriptor.outputDescriptor(of: $0) }
  case 2:
    names = descriptor.stateNames
    valueDescriptor = { descriptor.stateDescriptor(of: $0) }
  default:
    throw abiError(102, "descriptor kind must be input, output, or state")
  }
  guard Int(index) < names.count else { throw abiError(102, "descriptor index is out of range") }
  let name = names[Int(index)]
  guard let value = valueDescriptor(name) else {
    throw abiError(102, "Core AI omitted descriptor for named value \(name)")
  }
  guard case let .ndArray(arrayDescriptor) = value else {
    throw abiError(102, "Core AI image descriptors are not supported by the NDArray ABI")
  }
  let shape = arrayDescriptor.shape
  guard shape.count <= 8 else { throw abiError(102, "Core AI descriptor rank exceeds eight") }
  let shape64 = try shape.map { dimension -> Int64 in
    guard let converted = Int64(exactly: dimension) else {
      throw abiError(102, "Core AI descriptor dimension is not representable")
    }
    return converted
  }

  var result = ACArrayDescriptor()
  result.rank = Int32(shape.count)
  result.dynamic_shape = arrayDescriptor.hasDynamicShape ? 1 : 0
  result.resolved = arrayDescriptor.hasDynamicShape ? 0 : 1
  result.shape = int64Tuple(shape64)
  if !arrayDescriptor.hasDynamicShape {
    let strides = arrayDescriptor.preferredStrides
    guard strides.count == shape.count else {
      throw abiError(102, "Core AI returned an invalid preferred descriptor layout")
    }
    let strides64 = try strides.map { stride -> Int64 in
      guard let converted = Int64(exactly: stride) else {
        throw abiError(102, "Core AI descriptor stride is not representable")
      }
      return converted
    }
    guard arrayDescriptor.minimumByteCount >= 0 else {
      throw abiError(102, "Core AI returned a negative minimum byte count")
    }
    result.strides = int64Tuple(strides64)
    result.minimum_bytes = UInt64(arrayDescriptor.minimumByteCount)
  }
  result.name = ownedCString(name)
  result.dtype = ownedCString(String(describing: arrayDescriptor.scalarType))
  return result
}

@available(macOS 27.0, *)
private func validateFunctionInputs(
  function: CoreFunctionHandle,
  bindings: [String: CoreArrayHandle]
) throws -> [String: NDArray] {
  let descriptor = function.function.descriptor
  guard descriptor.stateNames.isEmpty else {
    throw abiError(102, "function declares state but this ABI does not accept state bindings")
  }
  let expectedNames = descriptor.inputNames
  guard Set(expectedNames).count == expectedNames.count,
        Set(bindings.keys) == Set(expectedNames), bindings.count == expectedNames.count else {
    throw abiError(102, "inference inputs must exactly match the model's named inputs")
  }

  var inputArrays: [String: NDArray] = [:]
  inputArrays.reserveCapacity(expectedNames.count)
  for name in expectedNames {
    guard let expected = descriptor.inputDescriptor(of: name),
          case let .ndArray(expectedArray) = expected else {
      throw abiError(102, "Core AI input \(name) is not a named NDArray")
    }
    guard let actual = bindings[name] else {
      throw abiError(102, "missing Core AI input \(name)")
    }
    guard actual.array.scalarType == expectedArray.scalarType else {
      throw abiError(102, "dtype mismatch for Core AI input \(name)")
    }
    let actualShape = actual.array.shape
    let expectedShape = expectedArray.shape
    guard actualShape.count == expectedShape.count else {
      throw abiError(102, "rank mismatch for Core AI input \(name)")
    }
    for axis in expectedShape.indices {
      let expectedDimension = expectedShape[axis]
      let actualDimension = actualShape[axis]
      guard expectedDimension == -1 || expectedDimension == actualDimension else {
        throw abiError(102, "static dimension mismatch for Core AI input \(name)")
      }
      guard actualDimension >= 0 else {
        throw abiError(102, "Core AI input \(name) has a negative dimension")
      }
    }
    guard abiType(for: actual.array.scalarType) != nil else {
      throw abiError(102, "host inference inputs support only float32 and int32")
    }
    inputArrays[name] = actual.array
  }
  return inputArrays
}

@available(macOS 27.0, *)
private func modelEvidenceContext(
  invocationID: UInt64,
  policy: Int32,
  options: SpecializationOptions,
  functionName: String = "",
  assetDigest: String = "",
  architecture: String = AIModel.deviceArchitectureName
) -> EvidenceContext {
  EvidenceContext(
    invocationID: invocationID,
    kind: functionName.isEmpty ? 1 : 3,
    requestedPolicy: policy,
    availableMask: unitMask(ComputeUnitKind.availableKinds),
    allowedMask: unitMask(options.allowedComputeUnitKinds),
    preferredUnit: unitIndex(options.preferredComputeUnitKind),
    functionName: functionName,
    assetDigest: assetDigest,
    architecture: architecture
  )
}

private func hexDigest(_ digest: SHA256.Digest) -> String {
  digest.map { String(format: "%02x", $0) }.joined()
}

private func relativePath(_ file: URL, under root: URL) throws -> String {
  let rootPath = root.standardizedFileURL.path
  let filePath = file.standardizedFileURL.path
  let prefix = rootPath.hasSuffix("/") ? rootPath : rootPath + "/"
  guard filePath.hasPrefix(prefix) else {
    throw abiError(101, "asset enumerator returned a path outside its root")
  }
  return String(filePath.dropFirst(prefix.count))
}

private func fileSHA256(_ url: URL) throws -> String {
  let data = try Data(contentsOf: url, options: [.mappedIfSafe])
  return hexDigest(SHA256.hash(data: data))
}

private func assetSHA256(at url: URL) throws -> String {
  let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey])
  guard values.isSymbolicLink != true else {
    throw abiError(101, "Core AI model asset cannot be a symbolic link")
  }
  if values.isRegularFile == true {
    return try fileSHA256(url)
  }
  guard values.isDirectory == true else {
    throw abiError(101, "Core AI model asset is neither a regular file nor a directory")
  }
  guard let enumerator = FileManager.default.enumerator(
    at: url,
    includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey],
    options: []
  ) else {
    throw abiError(101, "unable to enumerate Core AI model asset directory")
  }
  var files: [URL] = []
  for case let file as URL in enumerator {
    let fileValues = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
    if fileValues.isRegularFile == true && fileValues.isSymbolicLink != true {
      files.append(file)
    }
  }
  var filesWithPaths: [(url: URL, path: String)] = try files.map { file in
    (url: file, path: try relativePath(file, under: url))
  }
  filesWithPaths.sort { left, right in
    left.path.utf8.lexicographicallyPrecedes(right.path.utf8)
  }
  var hasher = SHA256()
  for (file, path) in filesWithPaths {
    let pathData = Data(path.utf8)
    let fileDigest = Data(try fileSHA256(file).utf8)
    hasher.update(data: pathData)
    hasher.update(data: Data([0]))
    hasher.update(data: fileDigest)
    hasher.update(data: Data([10]))
  }
  return hexDigest(hasher.finalize())
}

@c @implementation
public func ac_message_free(_ message: UnsafeMutablePointer<CChar>?) {
  free(message)
}

@c @implementation
public func ac_retain(_ handle: UnsafeMutableRawPointer?) {
  guard let handle else { return }
  _ = Unmanaged<AnyObject>.fromOpaque(handle).retain()
}

@c @implementation
public func ac_release(_ handle: UnsafeMutableRawPointer?) {
  guard let handle else { return }
  Unmanaged<AnyObject>.fromOpaque(handle).release()
}

@c @implementation
public func ac_architecture(
  _ out: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> AMStatus {
  guard let out else { return statusFailure(103, "architecture output pointer is nil") }
  out.pointee = nil
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  out.pointee = ownedCString(AIModel.deviceArchitectureName)
  return statusOK()
}

@c @implementation
public func ac_available_units(_ mask: UnsafeMutablePointer<UInt32>?) -> AMStatus {
  guard let mask else { return statusFailure(103, "compute-unit output pointer is nil") }
  mask.pointee = 0
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  mask.pointee = unitMask(ComputeUnitKind.availableKinds)
  return statusOK()
}

@c @implementation
public func ac_model_open(
  _ path: UnsafePointer<CChar>?,
  _ policy: Int32,
  _ jobOut: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
) -> AMStatus {
  guard let jobOut else { return statusFailure(103, "model job output pointer is nil") }
  jobOut.pointee = nil
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  guard let path, path.pointee != 0 else {
    return statusFailure(101, "Core AI model path is empty")
  }
  do {
    let options = try optionsForPolicy(policy)
    let url = URL(fileURLWithPath: String(cString: path)).standardizedFileURL
    let architecture = AIModel.deviceArchitectureName
    let context = modelEvidenceContext(
      invocationID: nextInvocationID(),
      policy: policy,
      options: options,
      architecture: architecture
    )
    let job = CoreJobHandle(context: context)
    jobOut.pointee = retainedHandle(job)
    Task.detached(priority: .userInitiated) {
      do {
        let digest = try assetSHA256(at: url)
        let model = try await AIModel(contentsOf: url, options: options)
        let handle = CoreModelHandle(
          model: model,
          url: url,
          options: options,
          requestedPolicy: policy,
          assetDigest: digest,
          architecture: AIModel.deviceArchitectureName,
          availableMask: unitMask(ComputeUnitKind.availableKinds),
          allowedMask: unitMask(options.allowedComputeUnitKinds),
          preferredUnit: unitIndex(options.preferredComputeUnitKind)
        )
        job.finish(.model(handle), assetDigest: digest, architecture: handle.architecture)
      } catch {
        let failure = error as? ABIError
        job.finish(
          .failure(JobFailure(
            code: failure?.code ?? 101,
            message: failure?.message ?? String(describing: error)
          ))
        )
      }
    }
    return statusOK()
  } catch {
    return statusFailure(error, fallback: 101)
  }
}

@c @implementation
public func ac_job_wait(
  _ jobPointer: UnsafeMutableRawPointer?,
  _ timeoutMilliseconds: UInt32
) -> AMStatus {
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  do {
    let job = try objectHandle(jobPointer, as: CoreJobHandle.self)
    guard let outcome = job.wait(timeoutMilliseconds: timeoutMilliseconds) else {
      return statusFailure(104, "Core AI operation timed out")
    }
    if case let .failure(failure) = outcome {
      return statusFailure(failure.code, failure.message)
    }
    return statusOK()
  } catch {
    return statusFailure(error)
  }
}

@c @implementation
public func ac_job_model(
  _ jobPointer: UnsafeMutableRawPointer?,
  _ modelOut: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
) -> AMStatus {
  guard let modelOut else { return statusFailure(103, "model output pointer is nil") }
  modelOut.pointee = nil
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  do {
    let job = try objectHandle(jobPointer, as: CoreJobHandle.self)
    let (outcome, _) = job.snapshot()
    guard let outcome else { throw abiError(103, "Core AI model job has not completed") }
    guard case let .model(model) = outcome else {
      if case let .failure(failure) = outcome { throw abiError(failure.code, failure.message) }
      throw abiError(103, "job does not contain a loaded Core AI model")
    }
    modelOut.pointee = retainedHandle(model)
    return statusOK()
  } catch {
    return statusFailure(error)
  }
}

@c @implementation
public func ac_function_count(
  _ modelPointer: UnsafeMutableRawPointer?,
  _ count: UnsafeMutablePointer<UInt32>?
) -> AMStatus {
  guard let count else { return statusFailure(103, "function count output pointer is nil") }
  count.pointee = 0
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  do {
    let model = try objectHandle(modelPointer, as: CoreModelHandle.self)
    guard let value = UInt32(exactly: model.model.functionNames.count) else {
      throw abiError(103, "Core AI function count exceeds the C ABI range")
    }
    count.pointee = value
    return statusOK()
  } catch {
    return statusFailure(error)
  }
}

@c @implementation
public func ac_function_name(
  _ modelPointer: UnsafeMutableRawPointer?,
  _ index: UInt32,
  _ nameOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> AMStatus {
  guard let nameOut else { return statusFailure(103, "function name output pointer is nil") }
  nameOut.pointee = nil
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  do {
    let model = try objectHandle(modelPointer, as: CoreModelHandle.self)
    let names = model.model.functionNames
    guard Int(index) < names.count else { throw abiError(102, "function index is out of range") }
    nameOut.pointee = ownedCString(names[Int(index)])
    return statusOK()
  } catch {
    return statusFailure(error)
  }
}

@c @implementation
public func ac_function_load(
  _ modelPointer: UnsafeMutableRawPointer?,
  _ name: UnsafePointer<CChar>?,
  _ functionOut: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
) -> AMStatus {
  guard let functionOut else { return statusFailure(103, "function output pointer is nil") }
  functionOut.pointee = nil
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  guard let name, name.pointee != 0 else {
    return statusFailure(102, "Core AI function name is empty")
  }
  do {
    let model = try objectHandle(modelPointer, as: CoreModelHandle.self)
    let functionName = String(cString: name)
    guard let function = try model.model.loadFunction(named: functionName) else {
      return statusOK()
    }
    functionOut.pointee = retainedHandle(CoreFunctionHandle(
      model: model,
      function: function,
      name: functionName
    ))
    return statusOK()
  } catch {
    return statusFailure(error)
  }
}

@c @implementation
public func ac_descriptor_count(
  _ functionPointer: UnsafeMutableRawPointer?,
  _ kind: Int32,
  _ count: UnsafeMutablePointer<UInt32>?
) -> AMStatus {
  guard let count else { return statusFailure(103, "descriptor count output pointer is nil") }
  count.pointee = 0
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  do {
    let function = try objectHandle(functionPointer, as: CoreFunctionHandle.self)
    let descriptor = function.function.descriptor
    let valueCount: Int
    switch kind {
    case 0: valueCount = descriptor.inputNames.count
    case 1: valueCount = descriptor.outputNames.count
    case 2: valueCount = descriptor.stateNames.count
    default: throw abiError(102, "descriptor kind must be input, output, or state")
    }
    guard let result = UInt32(exactly: valueCount) else {
      throw abiError(103, "descriptor count exceeds the C ABI range")
    }
    count.pointee = result
    return statusOK()
  } catch {
    return statusFailure(error)
  }
}

@c @implementation
public func ac_descriptor(
  _ functionPointer: UnsafeMutableRawPointer?,
  _ kind: Int32,
  _ index: UInt32,
  _ out: UnsafeMutablePointer<ACArrayDescriptor>?
) -> AMStatus {
  guard let out else { return statusFailure(103, "descriptor output pointer is nil") }
  out.pointee = ACArrayDescriptor()
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  do {
    let function = try objectHandle(functionPointer, as: CoreFunctionHandle.self)
    out.pointee = try makeFunctionDescriptor(function: function, kind: kind, index: index)
    return statusOK()
  } catch {
    return statusFailure(error)
  }
}

@c @implementation
public func ac_array_import(
  _ host: UnsafeRawPointer?,
  _ descriptor: AMTensorDescriptor,
  _ arrayOut: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
) -> AMStatus {
  guard let arrayOut else { return statusFailure(103, "array output pointer is nil") }
  arrayOut.pointee = nil
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  do {
    let array = try makeHostTensor(host, descriptor: descriptor)
    arrayOut.pointee = retainedHandle(array)
    return statusOK()
  } catch {
    return statusFailure(error)
  }
}

@c @implementation
public func ac_array_copy(
  _ arrayPointer: UnsafeMutableRawPointer?,
  _ host: UnsafeMutableRawPointer?,
  _ descriptor: AMTensorDescriptor
) -> AMStatus {
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  do {
    let array = try objectHandle(arrayPointer, as: CoreArrayHandle.self)
    _ = try copySdkArray(array, to: host, descriptor: descriptor)
    return statusOK()
  } catch {
    return statusFailure(error)
  }
}

@available(macOS 27.0, *)
private func fillActualArrayDescriptor(_ handle: CoreArrayHandle) throws -> ACArrayDescriptor {
  let shape = handle.array.shape
  let strides = handle.array.strides
  guard shape.count <= 8, shape.count == strides.count else {
    throw abiError(102, "Core AI NDArray rank exceeds the C ABI descriptor")
  }
  let shape64 = try shape.map { value -> Int64 in
    guard let converted = Int64(exactly: value) else {
      throw abiError(102, "Core AI NDArray dimension is not representable")
    }
    return converted
  }
  let strides64 = try strides.map { value -> Int64 in
    guard let converted = Int64(exactly: value) else {
      throw abiError(102, "Core AI NDArray stride is not representable")
    }
    return converted
  }
  var result = ACArrayDescriptor()
  result.name = ownedCString(handle.name)
  result.dtype = ownedCString(String(describing: handle.array.scalarType))
  result.rank = Int32(shape.count)
  result.dynamic_shape = 0
  result.resolved = 1
  result.shape = int64Tuple(shape64)
  result.strides = int64Tuple(strides64)
  result.minimum_bytes = handle.minimumBytes
  result.host_copy_bytes = handle.hostCopyBytes
  return result
}

@c @implementation
public func ac_array_descriptor(
  _ arrayPointer: UnsafeMutableRawPointer?,
  _ out: UnsafeMutablePointer<ACArrayDescriptor>?
) -> AMStatus {
  guard let out else { return statusFailure(103, "array descriptor output pointer is nil") }
  out.pointee = ACArrayDescriptor()
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  do {
    let array = try objectHandle(arrayPointer, as: CoreArrayHandle.self)
    out.pointee = try fillActualArrayDescriptor(array)
    return statusOK()
  } catch {
    return statusFailure(error)
  }
}

@c @implementation
public func ac_submit(
  _ functionPointer: UnsafeMutableRawPointer?,
  _ bindingsPointer: UnsafePointer<ACBinding>?,
  _ count: UInt32,
  _ jobOut: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
) -> AMStatus {
  guard let jobOut else { return statusFailure(103, "inference job output pointer is nil") }
  jobOut.pointee = nil
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  do {
    let function = try objectHandle(functionPointer, as: CoreFunctionHandle.self)
    guard count == 0 || bindingsPointer != nil else {
      throw abiError(102, "nonempty inference bindings have a nil pointer")
    }
    var bindings: [String: CoreArrayHandle] = [:]
    bindings.reserveCapacity(Int(count))
    if count > 0, let bindingsPointer {
      for index in 0..<Int(count) {
        let binding = bindingsPointer[index]
        guard let name = binding.name, name.pointee != 0 else {
          throw abiError(102, "inference binding name is empty")
        }
        let bindingName = String(cString: name)
        guard bindings[bindingName] == nil else {
          throw abiError(102, "duplicate inference input name \(bindingName)")
        }
        let array = try objectHandle(binding.array, as: CoreArrayHandle.self)
        bindings[bindingName] = array
      }
    }
    // Validate all names and metadata before constructing SDK input views.
    let inputArrays = try validateFunctionInputs(function: function, bindings: bindings)
    let model = function.model
    let context = modelEvidenceContext(
      invocationID: nextInvocationID(),
      policy: model.requestedPolicy,
      options: model.options,
      functionName: function.name,
      assetDigest: model.assetDigest,
      architecture: model.architecture
    )
    let job = CoreJobHandle(context: context)
    jobOut.pointee = retainedHandle(job)
    Task.detached(priority: .userInitiated) {
      defer { withExtendedLifetime(bindings) {} }
      do {
        var sdkOutputs = try await function.function.run(inputs: inputArrays)
        let expectedNames = function.function.descriptor.outputNames
        let reportedNames = Array(sdkOutputs.names)
        guard Set(reportedNames) == Set(expectedNames),
              reportedNames.count == expectedNames.count else {
          throw abiError(102, "Core AI returned output names inconsistent with its function descriptor")
        }
        var outputs: [String: CoreArrayHandle] = [:]
        outputs.reserveCapacity(expectedNames.count)
        for name in expectedNames {
          guard let output = sdkOutputs.remove(name)?.ndArray else {
            throw abiError(102, "Core AI output \(name) is not an NDArray")
          }
          let minimumBytes = try sdkArrayMinimumBytes(output)
          outputs[name] = CoreArrayHandle(
            array: output,
            name: name,
            minimumBytes: minimumBytes
          )
        }
        job.finish(.outputs(outputs))
      } catch {
        let failure = error as? ABIError
        job.finish(.failure(JobFailure(
          code: failure?.code ?? 103,
          message: failure?.message ?? String(describing: error)
        )))
      }
    }
    return statusOK()
  } catch {
    return statusFailure(error)
  }
}

@c @implementation
public func ac_job_output_count(
  _ jobPointer: UnsafeMutableRawPointer?,
  _ count: UnsafeMutablePointer<UInt32>?
) -> AMStatus {
  guard let count else { return statusFailure(103, "output count pointer is nil") }
  count.pointee = 0
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  do {
    let job = try objectHandle(jobPointer, as: CoreJobHandle.self)
    let (outcome, _) = job.snapshot()
    guard let outcome else { throw abiError(103, "inference job has not completed") }
    guard case let .outputs(outputs) = outcome else {
      if case let .failure(failure) = outcome { throw abiError(failure.code, failure.message) }
      throw abiError(103, "job is a model-loading operation, not inference")
    }
    guard let value = UInt32(exactly: outputs.count) else {
      throw abiError(103, "Core AI output count exceeds the C ABI range")
    }
    count.pointee = value
    return statusOK()
  } catch {
    return statusFailure(error)
  }
}

@c @implementation
public func ac_job_output(
  _ jobPointer: UnsafeMutableRawPointer?,
  _ index: UInt32,
  _ nameOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
  _ arrayOut: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
) -> AMStatus {
  guard let nameOut, let arrayOut else {
    return statusFailure(103, "inference output pointer is nil")
  }
  nameOut.pointee = nil
  arrayOut.pointee = nil
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  do {
    let job = try objectHandle(jobPointer, as: CoreJobHandle.self)
    let (outcome, _) = job.snapshot()
    guard let outcome else { throw abiError(103, "inference job has not completed") }
    guard case let .outputs(outputs) = outcome else {
      if case let .failure(failure) = outcome { throw abiError(failure.code, failure.message) }
      throw abiError(103, "job is a model-loading operation, not inference")
    }
    let names = outputs.keys.sorted()
    guard Int(index) < names.count, let array = outputs[names[Int(index)]] else {
      throw abiError(102, "inference output index is out of range")
    }
    nameOut.pointee = ownedCString(names[Int(index)])
    arrayOut.pointee = retainedHandle(array)
    return statusOK()
  } catch {
    return statusFailure(error)
  }
}

@c @implementation
public func ac_job_evidence(
  _ jobPointer: UnsafeMutableRawPointer?,
  _ out: UnsafeMutablePointer<ACEvidence>?
) -> AMStatus {
  guard let out else { return statusFailure(103, "evidence output pointer is nil") }
  out.pointee = ACEvidence()
  guard #available(macOS 27.0, *) else {
    return statusFailure(100, "Core AI requires macOS 27 or later")
  }
  do {
    let job = try objectHandle(jobPointer, as: CoreJobHandle.self)
    let (outcome, context) = job.snapshot()
    var evidence = ACEvidence()
    evidence.invocation_id = context.invocationID
    if let outcome {
      if case .failure = outcome { evidence.completed = 0 }
      else { evidence.completed = 1 }
    }
    evidence.kind = context.kind
    evidence.requested_policy = context.requestedPolicy
    evidence.actual_placement = -1
    evidence.available_mask = context.availableMask
    evidence.allowed_mask = context.allowedMask
    evidence.preferred_unit = context.preferredUnit
    evidence.host_copy_bytes = 0
    evidence.materialization_bytes = 0
    evidence.cpu_fallback_count = 0
    evidence.asset_digest = ownedCString(context.assetDigest)
    evidence.function_name = ownedCString(context.functionName)
    evidence.architecture = ownedCString(context.architecture)
    out.pointee = evidence
    return statusOK()
  } catch {
    return statusFailure(error)
  }
}
