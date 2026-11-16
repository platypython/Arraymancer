# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides a stdlib-only builder for the optional native Apple adapters.
import std/[os, osproc, strutils, streams]
const help = """Usage: build_apple [--backend=all|metal|coreai] [--output=PATH] [--help]
Build native Apple libraries and Core AI test models in build/apple.
No arguments print this help and build both backends.
"""
proc run(executable: string; args: openArray[string]): string =
  ## Use argument arrays so SDK and output paths never pass through a shell.
  let child = startProcess(executable, args = args, options = {poUsePath, poStdErrToStdOut})
  defer: child.close()
  result = child.outputStream.readAll()
  let code = child.waitForExit()
  if code != 0: raise newException(IOError, executable & " failed: " & result)
proc buildMetallib(source, name, output: string) =
  ## Tests load a separately compiled library, not the runtime compiler's output.
  echo run("xcrun", ["-sdk", "macosx", "metal", "-std=metal3.2", "-fno-fast-math",
    "-mmacosx-version-min=27.0", "-c", source, "-o", output / (name & ".air")])
  echo run("xcrun", ["-sdk", "macosx", "metallib", output / (name & ".air"),
    "-o", output / (name & ".metallib")])

proc buildMetal(root, output, sdk: string) =
  let clang = run("xcrun", ["--find", "clang++"]).strip
  buildMetallib(root / "src/arraymancer/tensor/backend/native/metal_kernels.metal",
    "arraymancer", output)
  let testShader = root / "tests/apple/kernels/double.metal"
  if fileExists(testShader):
    buildMetallib(testShader, "arraymancer_test", output)
  let library = absolutePath(output / "libarraymancer_metal.dylib")
  echo run(clang, ["-std=c++17", "-fobjc-arc", "-dynamiclib", "-arch", "arm64",
    "-target", "arm64-apple-macos27.0", "-isysroot", sdk, "-framework", "Metal",
    "-framework", "Foundation", "-install_name", "@rpath/libarraymancer_metal.dylib",
    root / "src/arraymancer/tensor/backend/native/metal_bridge.mm", "-o", library])
  echo "Built " & library

proc buildCoreAI(root, output, sdk: string) =
  let swift = run("xcrun", ["--find", "swiftc"]).strip
  let library = absolutePath(output / "libarraymancer_coreai.dylib")
  echo run(swift, ["-swift-version", "6", "-emit-library",
    "-target", "arm64-apple-macos27.0", "-sdk", sdk,
    "-I", root / "src/arraymancer/tensor/backend/native",
    "-L", absolutePath(output), "-larraymancer_metal",
    "-Xlinker", "-rpath", "-Xlinker", "@loader_path",
    "-Xlinker", "-install_name", "-Xlinker", "@rpath/libarraymancer_coreai.dylib",
    root / "src/arraymancer/coreai/native/coreai_bridge.swift", "-o", library])
  echo "Built " & library
  let model = root / "tests/apple/models/parity.aimodel"
  if not (fileExists(model) or dirExists(model)):
    raise newException(IOError, "Missing authentic Core AI test asset: " & model)
  let models = output / "models"
  createDir(models)
  echo run("xcrun", ["coreai-build", "compile", model, "--platform", "macOS",
    "--min-deployment-version", "27.0", "--output", models / "compiled"])

when isMainModule:
  when not (defined(macosx) and defined(arm64)):
    quit("Apple backends require macOS arm64", 1)
  var output = "build/apple"
  var backend = "all"
  if paramCount() == 0: echo help
  for arg in commandLineParams():
    if arg == "--help": echo help; quit(0)
    elif arg.startsWith("--backend="):
      backend = arg[10 .. ^1]
      if backend notin ["all", "metal", "coreai"]:
        quit("Unsupported backend: " & backend & "\n" & help, 2)
    elif arg.startsWith("--output="): output = arg[9 .. ^1]
    else: quit("Unsupported argument: " & arg & "\n" & help, 2)
  let sdk = run("xcrun", ["--sdk", "macosx", "--show-sdk-path"]).strip
  let root = currentSourcePath().parentDir.parentDir
  createDir(output)
  if backend in ["all", "metal"] or
      not fileExists(output / "libarraymancer_metal.dylib"):
    buildMetal(root, output, sdk)
  if backend in ["all", "coreai"]:
    buildCoreAI(root, output, sdk)
