# Copyright 2026 Arraymancer contributors. Apache-2.0.
## Provides a stdlib-only builder for the optional native Apple adapters.
import std/[os, osproc, strutils, streams]
const help = """Usage: build_apple [--backend=metal] [--output=PATH] [--help]
Build native Apple libraries; defaults to Metal in build/apple.
No arguments print this help and build the implemented backend.
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

when isMainModule:
  when not (defined(macosx) and defined(arm64)):
    quit("Apple backends require macOS arm64", 1)
  var output = "build/apple"
  if paramCount() == 0: echo help
  for arg in commandLineParams():
    if arg == "--help": echo help; quit(0)
    elif arg == "--backend=metal": discard
    elif arg.startsWith("--output="): output = arg[9 .. ^1]
    else: quit("Unsupported argument: " & arg & "\n" & help, 2)
  let sdk = run("xcrun", ["--sdk", "macosx", "--show-sdk-path"]).strip
  let clang = run("xcrun", ["--find", "clang++"]).strip
  let root = currentSourcePath().parentDir.parentDir
  createDir(output)
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
