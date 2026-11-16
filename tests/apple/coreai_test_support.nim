# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Independently hashes authentic Core AI assets with the platform SHA-256 implementation.
import std/[algorithm, json, os]
import ../../src/arraymancer

const coreAISourceAsset* = "tests/apple/models/parity.aimodel"
const coreAIManifestPath* = "tests/apple/models/manifest.json"

proc ccSha256(input: pointer; length: uint32; digest: ptr uint8): ptr uint8
  {.importc: "CC_SHA256", header: "<CommonCrypto/CommonDigest.h>".}

proc sha256Bytes*(data: string): string =
  if uint64(data.len) > uint64(high(uint32)):
    raise newException(ValueError, "Test asset exceeds the SHA-256 single-call size limit")
  var digest: array[32, uint8]
  let input = if data.len == 0: nil else: cast[pointer](unsafeAddr data[0])
  discard ccSha256(input, uint32(data.len), addr digest[0])
  const digits = "0123456789abcdef"
  result = newString(64)
  for i, value in digest:
    result[2 * i] = digits[int(value shr 4)]
    result[2 * i + 1] = digits[int(value and 15)]

proc hashCoreAIAsset*(path: string): tuple[digest: string,
    files: seq[tuple[path, sha256: string, bytes: int]]] =
  ## Directory digests bind file names as well as bytes, independently of the native bridge.
  if fileExists(path):
    let data = readFile(path)
    let digest = sha256Bytes(data)
    return (digest, @[(path.extractFilename, digest, data.len)])
  if not dirExists(path):
    raise newException(IOError, "Missing Core AI asset: " & path)
  var paths: seq[string]
  for file in walkDirRec(path): paths.add relativePath(file, path)
  paths.sort()
  var canonical: string
  for file in paths:
    let data = readFile(path / file)
    let digest = sha256Bytes(data)
    result.files.add((file, digest, data.len))
    canonical.add(file & '\0' & digest & '\n')
  result.digest = sha256Bytes(canonical)

proc coreAIManifestObservation*(): JsonNode =
  let manifest = parseFile(coreAIManifestPath)
  let actual = hashCoreAIAsset(coreAISourceAsset)
  var files = newJArray()
  for file in actual.files:
    files.add %*{"path": file.path, "sha256": file.sha256, "bytes": file.bytes}
  let versions = manifest["versions"]
  let pinsMatch = versions["python"].getStr == "3.13.18" and
    versions["torch"].getStr == "2.15.0" and
    versions["coreai_torch"].getStr == "0.4.3" and
    versions["coreai_core"].getStr == "1.0.0b3"
  let matches = manifest["format"].getInt == 1 and pinsMatch and
    manifest["asset"]["path"].getStr == coreAISourceAsset.extractFilename and
    manifest["asset"]["sha256"].getStr == actual.digest and
    manifest["asset"]["files"] == files
  %*{"asset_digest": actual.digest, "files": files,
    "versions": versions, "matches_manifest": matches}

proc coreAIModelPaths*(): array[2, string] =
  ## coreai-build preserves the source basename and appends the SDK architecture.
  let compiled = arraymancerAppleLibDir / "models/compiled" /
    ("parity." & coreAIArchitecture() & ".aimodelc")
  if not dirExists(compiled):
    raise newException(IOError, "Missing SDK-matching compiled Core AI asset: " & compiled)
  [coreAISourceAsset, compiled]
