#!/usr/bin/env python3
"""Generate the authentic Core AI parity model asset from pinned PyTorch exports.

This developer-only generator requires CPython 3.13.18, PyTorch 2.15.0,
coreai-torch 0.4.3, and coreai-core 1.0.0b3. It never participates in native
Arraymancer execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PYTHON_VERSION = "3.13.18"
TORCH_VERSION = "2.15.0"
COREAI_TORCH_VERSION = "0.4.3"
COREAI_CORE_VERSION = "1.0.0b3"
ASSET_NAME = "parity.aimodel"


def _versions(torch: Any) -> dict[str, str]:
    if sys.implementation.name != "cpython":
        raise ValueError(f"CPython required; found {sys.implementation.name}")
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    if python_version != PYTHON_VERSION:
        raise ValueError(f"CPython {PYTHON_VERSION} required; found {python_version}")

    torch_version = str(torch.__version__)
    normalized_torch = torch_version.split("+", 1)[0]
    if normalized_torch != TORCH_VERSION or any(
        marker in normalized_torch for marker in ("a", "b", "rc")
    ):
        raise ValueError(f"stable torch {TORCH_VERSION} required; found {torch_version}")
    torch_git_commit = str(torch.version.git_version or "")
    if not torch_git_commit:
        raise ValueError("installed torch does not report torch.version.git_version")

    coreai_torch_version = importlib.metadata.version("coreai-torch")
    if coreai_torch_version != COREAI_TORCH_VERSION:
        raise ValueError(
            f"coreai-torch {COREAI_TORCH_VERSION} required; found {coreai_torch_version}"
        )
    coreai_core_version = importlib.metadata.version("coreai-core")
    if coreai_core_version != COREAI_CORE_VERSION:
        raise ValueError(
            f"coreai-core {COREAI_CORE_VERSION} required; found {coreai_core_version}"
        )

    return {
        "python": python_version,
        "torch": torch_version,
        "torch_git_commit": torch_git_commit,
        "coreai_torch": coreai_torch_version,
        "coreai_core": coreai_core_version,
    }


def _asset_hashes(asset_path: Path) -> tuple[list[dict[str, Any]], str]:
    files: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(asset_path.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative_path = path.relative_to(asset_path).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "path": relative_path,
                "sha256": file_digest,
                "bytes": path.stat().st_size,
            }
        )
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    if not files:
        raise ValueError(f"Core AI converter produced no regular files in {asset_path}")
    return files, digest.hexdigest()

def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _increment_entrypoint_manifest(exported_program: Any) -> dict[str, Any]:
    """Validate and describe the exported mutable scalar counter contract."""
    graph_signature = exported_program.graph_signature
    user_inputs = list(graph_signature.user_inputs)
    user_outputs = list(graph_signature.user_outputs)
    buffers_to_mutate = dict(graph_signature.buffers_to_mutate)
    if len(user_inputs) != 1 or user_inputs[0] != "delta":
        raise ValueError(f"increment export must have only the delta input; got {user_inputs!r}")
    if len(user_outputs) != 1:
        raise ValueError(f"increment export must have one user output; got {user_outputs!r}")
    if len(buffers_to_mutate) != 1 or "counter" not in buffers_to_mutate.values():
        raise ValueError(
            "increment export must mutate exactly the registered counter buffer; "
            f"got {buffers_to_mutate!r}"
        )
    return {
        "graph_signature": {
            "user_inputs": user_inputs,
            "user_outputs": user_outputs,
            "buffers_to_mutate": buffers_to_mutate,
            "registered_inputs": ["delta"],
            "registered_outputs": ["y"],
            "registered_states": ["counter"],
        },
        "inputs": [
            {"name": "delta", "dtype": "float32", "shape": [], "dynamic_dimensions": []}
        ],
        "outputs": [
            {"name": "y", "dtype": "float32", "shape": [], "dynamic_dimensions": []}
        ],
        "states": [
            {"name": "counter", "dtype": "float32", "shape": [], "dynamic_dimensions": []}
        ],
    }


def _entrypoint_manifest(
    exported_program: Any,
    *,
    input_shape: list[int | str],
    dynamic_dimensions: list[dict[str, int | str]],
) -> dict[str, Any]:
    graph_signature = exported_program.graph_signature
    user_inputs = list(graph_signature.user_inputs)
    user_outputs = list(graph_signature.user_outputs)
    if len(user_inputs) != 1 or len(user_outputs) != 1:
        raise ValueError(
            "expected each exported graph to have exactly one user input and one user output; "
            f"got inputs={user_inputs!r}, outputs={user_outputs!r}"
        )
    return {
        "graph_signature": {
            "user_inputs": user_inputs,
            "user_outputs": user_outputs,
            "registered_inputs": ["x"],
            "registered_outputs": ["y"],
        },
        "inputs": [
            {
                "name": "x",
                "dtype": "float32",
                "shape": input_shape,
                "dynamic_dimensions": dynamic_dimensions,
            }
        ],
        "outputs": [
            {
                "name": "y",
                "dtype": "float32",
                "shape": input_shape,
                "dynamic_dimensions": dynamic_dimensions,
            }
        ],
    }


def _build_asset(torch: Any, asset_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    versions = _versions(torch)
    coreai_torch = importlib.import_module("coreai_torch")
    torch_export = torch.export

    class Identity(torch.nn.Module):
        def forward(self, x: Any) -> Any:
            return x.clone()

    class IdentityMatrix(torch.nn.Module):
        def forward(self, x: Any) -> Any:
            return x.clone()

    class Increment(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("counter", torch.zeros((), dtype=torch.float32))

        def forward(self, delta: Any) -> Any:
            self.counter.add_(delta)
            return self.counter.clone()


    torch.set_num_threads(1)
    identity_input = torch.tensor([1.25, -2.0], dtype=torch.float32)
    identity_export = torch_export.export(Identity(), (identity_input,)).run_decompositions(
        coreai_torch.get_decomp_table()
    )

    rows_dimension = torch_export.Dim("rows", min=1, max=8)
    matrix_input = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32
    )
    matrix_export = torch_export.export(
        IdentityMatrix(),
        (matrix_input,),
        dynamic_shapes={"x": {0: rows_dimension}},
    ).run_decompositions(coreai_torch.get_decomp_table())
    increment_delta = torch.tensor(1.0, dtype=torch.float32)
    increment_export = torch_export.export(Increment(), (increment_delta,)).run_decompositions(
        coreai_torch.get_decomp_table()
    )
    increment_state = increment_export.state_dict.get("counter")
    if (
        increment_state is None
        or increment_state.dtype != torch.float32
        or tuple(increment_state.shape) != ()
    ):
        raise ValueError(
            "increment export must retain a scalar float32 counter buffer; "
            f"got {increment_state!r}"
        )


    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        identity_export,
        input_names=["x"],
        output_names=["y"],
        entrypoint_name="identity",
    )
    converter.add_exported_program(
        matrix_export,
        input_names=["x"],
        output_names=["y"],
        entrypoint_name="identity_matrix",
    )
    converter.add_exported_program(
        increment_export,
        input_names=["delta"],
        output_names=["y"],
        state_names=["counter"],
        entrypoint_name="increment",
    )

    converter.to_coreai().save_asset(asset_path)

    files, asset_sha256 = _asset_hashes(asset_path)
    manifest = {
        "format": 1,
        "asset": {
            "path": ASSET_NAME,
            "kind": "directory",
            "sha256": asset_sha256,
            "hash_algorithm": "sha256(relative_posix_path + NUL + lowercase_file_sha256 + LF, sorted by relative path)",
            "files": files,
        },
        "versions": versions,
        "source_models": {
            "identity": {
                "class": "Identity",
                "forward": "return x.clone()",
                "example": {"dtype": "float32", "shape": [2], "values": [1.25, -2.0]},
                **_entrypoint_manifest(
                    identity_export,
                    input_shape=[2],
                    dynamic_dimensions=[],
                ),
            },
            "identity_matrix": {
                "class": "IdentityMatrix",
                "forward": "return x.clone()",
                "example": {
                    "dtype": "float32",
                    "shape": [2, 3],
                    "values": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                },
                **_entrypoint_manifest(
                    matrix_export,
                    input_shape=["rows", 3],
                    dynamic_dimensions=[{"name": "rows", "min": 1, "max": 8}],
                ),
            },
            "increment": {
                "class": "Increment",
                "forward": "self.counter.add_(delta); return self.counter.clone()",
                "example": {
                    "initial_state": {"dtype": "float32", "shape": [], "value": 0.0},
                    "inputs": {
                        "delta": {"dtype": "float32", "shape": [], "value": 1.0}
                    },
                    "outputs": {"y": {"dtype": "float32", "shape": [], "value": 1.0}},
                    "state_after": {
                        "counter": {"dtype": "float32", "shape": [], "value": 1.0}
                    },
                },
                **_increment_entrypoint_manifest(increment_export),
            },
        },
    }
    return manifest, versions

def _write_json_atomically(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert pinned PyTorch exports into the authentic Core AI parity model."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "apple" / "models",
        help="directory in which parity.aimodel and manifest.json are written",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    parser = _parser()
    if not arguments:
        parser.print_help(sys.stderr)
    options = parser.parse_args(arguments)

    temporary_asset: Path | None = None
    try:
        torch = importlib.import_module("torch")
        output_directory = options.output.resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        destination = output_directory / ASSET_NAME
        build_path = output_directory / f".parity-{os.getpid()}.aimodel"
        temporary_asset = build_path
        if build_path.exists():
            if build_path.is_dir():
                shutil.rmtree(build_path)
            else:
                build_path.unlink()

        # save_asset requires the .aimodel suffix; build under a sibling path so a
        # conversion failure cannot replace the checked-in asset with partial output.
        manifest, versions = _build_asset(torch, build_path)
        files = manifest["asset"]["files"]
        asset_sha256 = manifest["asset"]["sha256"]

        manifest_path = output_directory / "manifest.json"
        backup_path = output_directory / f".parity-backup-{os.getpid()}"
        if destination.exists():
            destination.replace(backup_path)
        try:
            build_path.replace(destination)
            _write_json_atomically(manifest_path, manifest)
        except Exception:
            if destination.exists():
                _remove_path(destination)
            if backup_path.exists():
                backup_path.replace(destination)
            raise
        if backup_path.exists():
            _remove_path(backup_path)
        sys.stdout.write(
            f"Generated {destination} with {len(files)} files; "
            f"source asset SHA-256 {asset_sha256}; "
            f"Python {versions['python']}, torch {versions['torch']}, "
            f"coreai-torch {versions['coreai_torch']}, coreai-core {versions['coreai_core']}\n"
        )
        return 0
    except Exception as error:
        print(f"generate_coreai_models.py: {error}", file=sys.stderr)
        return 2
    finally:
        if temporary_asset is not None and temporary_asset.exists():
            if temporary_asset.is_dir():
                _remove_path(temporary_asset)
            else:
                temporary_asset.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
