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

## Inventories immutable PyTorch source objects without importing the reference runtime.
import std/[json, os, osproc, streams, strutils, parseopt, sets, algorithm]

const root = "tests/parity/"
const usage = "Usage: pytorch_inventory [--source=PATH] [--output=PATH | --check] [--help]"

proc fail(message: string) {.noreturn.} =
  raise newException(ValueError, message)

proc git(source: string; args: seq[string]): string =
  let p = startProcess("git", args = @["-C", source] & args,
                       options = {poUsePath, poStdErrToStdOut})
  defer: close(p)
  result = p.outputStream.readAll()
  if p.waitForExit() != 0: fail(source & ": " & result.strip())

proc reference(): JsonNode =
  result = parseFile(root & "reference.json")
  let required = ["format", "torch", "source_commit", "python", "apple"]
  if result.kind != JObject or result.len != required.len: fail("reference: fields")
  for key in required:
    if not result.hasKey(key): fail("reference: missing " & key)
  if result["format"] != %1 or result["torch"] != %"2.15.0" or
      result["python"] != %"3.13.18" or
      result["source_commit"] != %"31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457":
    fail("reference: unsupported identity")
  let apple = %*{"xcode": "27.1", "xcode_build": "27A9275", "macos_sdk": "27.0",
    "swift": "6.4", "minimum_macos": "27.0", "architecture": "arm64", "metal_api": "4"}
  if result["apple"] != apple: fail("reference: Apple baseline")

proc inventory(source: string): JsonNode =
  let refData = reference()
  let commit = refData["source_commit"].getStr()
  discard git(source, @["cat-file", "-e", commit & "^{commit}"])
  var records: seq[JsonNode]
  var ids = initHashSet[string]()
  for item in [("api_area", "docs/source/pytorch-api.md"),
               ("aten_schema", "aten/src/ATen/native/native_functions.yaml"),
               ("autograd_rule", "tools/autograd/derivatives.yaml"),
               ("mps_export", "torch/mps/__init__.py")]:
    let (kind, path) = item
    let blob = git(source, @["rev-parse", commit & ":" & path]).strip()
    let text = git(source, @["cat-file", "blob", blob])
    var active = false
    var foundBlock = false
    var caption = ""
    var count = 0
    var lineNumber = 0
    for raw in text.splitLines():
      let index = lineNumber
      inc lineNumber
      let line = raw.strip()
      var declaration, identity: string
      if kind == "api_area":
        if line == "```{toctree}":
          active = true
          caption = ""
          continue
        if active and line == "```":
          active = false
          continue
        if not active or line.len == 0: continue
        if line.startsWith(":caption:"): caption = line[10..^1].strip()
        if line.startsWith(":"): continue
        declaration = line
        var target = line
        if line.endsWith(">") and line.contains(" <"):
          target = line[line.rfind(" <") + 2 .. ^2]
        identity = "api:" & target
      elif kind == "mps_export":
        if line == "__all__ = [":
          if foundBlock: fail(path & ": duplicate __all__")
          active = true
          foundBlock = true
          continue
        if not active: continue
        if line == "]":
          active = false
          continue
        var entry = line
        if entry.endsWith(","): entry.setLen(entry.len - 1)
        if entry.len < 3 or entry[0] notin {'\'', '"'} or entry[^1] != entry[0]:
          fail(path & ": malformed export at " & $(index + 1))
        let name = entry[1..^2]
        if name.len == 0 or name[0] notin {'a'..'z', 'A'..'Z', '_'}:
          fail(path & ": malformed identifier")
        for c in name:
          if c notin {'a'..'z', 'A'..'Z', '0'..'9', '_'}: fail(path & ": export expression")
        declaration = "torch.mps." & name
        identity = "mps:" & declaration
      else:
        let prefix = if kind == "aten_schema": "- func:" else: "- name:"
        if not raw.startsWith(prefix): continue
        declaration = raw[prefix.len..^1].strip()
        if declaration.len == 0 or declaration[0] in {'\'', '"', '|', '>'} or
            not declaration.contains('(') or not declaration.contains(')') or
            not declaration.contains("->"):
          fail(path & ": unsupported declaration at " & $(index + 1))
        identity = (if kind == "aten_schema": "aten:" else: "autograd:") & declaration
      if identity in ids: fail(path & ": duplicate " & identity)
      ids.incl(identity)
      var record = %*{"kind": kind, "id": identity, "declaration": declaration,
        "source": {"path": path, "line": index + 1, "blob": blob}}
      if kind == "api_area":
        record["caption"] = %caption
        record["target"] = %identity[4..^1]
      records.add(record)
      inc count
    if count == 0 or active: fail(path & ": empty or unterminated inventory")
  records.sort(proc(a, b: JsonNode): int =
    result = cmp(a["kind"].getStr(), b["kind"].getStr())
    if result == 0: result = cmp(a["id"].getStr(), b["id"].getStr()))
  result = %*{"format": 1, "reference": refData,
    "granularity": "api_areas_aten_schemas_autograd_rules_and_mps_exports", "records": records}

proc main(): int =
  var source = "ref/pytorch"
  var output = ""
  var check = false
  if paramCount() == 0: stderr.writeLine(usage)
  if commandLineParams() == @["--help"]:
    echo usage
    return 0
  for kind, key, value in getopt():
    case kind
    of cmdLongOption:
      case key
      of "source": source = value
      of "output":
        if value.len == 0: fail("--output requires a path")
        output = value
      of "check":
        if value.len != 0: fail("--check takes no value")
        check = true
      else: fail("Unknown option: " & key)
    else: fail("Unexpected argument: " & key)
  if check and output.len > 0: fail("--check and --output conflict")
  let generated = inventory(source).pretty() & "\n"
  if check:
    return (if readFile(root & "inventory.json") == generated: 0 else: 1)
  if output.len > 0: writeFile(output, generated)
  else: stdout.write(generated)

when isMainModule:
  try: quit(main())
  except CatchableError:
    stderr.writeLine(getCurrentExceptionMsg())
    quit(2)
