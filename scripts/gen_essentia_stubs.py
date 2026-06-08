#!/usr/bin/env python3
"""
Generate type stubs for essentia.standard.

Combines two sources:
  - Standard algorithms: queried live from the C extension via Algorithm.getStruct()
  - TensorflowPredict* algorithms: hardcoded from the official docs, as they are
    absent from the non-TF .so registry at generation time.

Usage:
    uv run python scripts/gen_essentia_stubs.py
    uv run python scripts/gen_essentia_stubs.py --out typings/essentia/standard.pyi
    uv run python scripts/gen_essentia_stubs.py --check
"""

import argparse
import sys
from pathlib import Path

import essentia._essentia as _e

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUT = PROJECT_ROOT / "stubs" / "essentia" / "standard.pyi"

# TensorflowPredict* algorithms absent from the plain (non-TF) .so registry.
# Sourced from https://essentia.upf.edu/algorithms_reference.html
# Parameter dicts: {"name": str, "type": str, "default": str | None}
_TF_STRUCTS: dict[str, dict] = {
    "TensorflowPredict": {
        "inputs": [{"name": "poolIn", "type": "pool"}],
        "outputs": [{"name": "poolOut", "type": "pool"}],
        "parameters": [
            {"name": "graphFilename", "type": "string", "default": ""},
            {"name": "inputs", "type": "vector_string", "default": None},
            {"name": "isTraining", "type": "bool", "default": "false"},
            {"name": "isTrainingName", "type": "string", "default": ""},
            {"name": "outputs", "type": "vector_string", "default": None},
            {"name": "savedModel", "type": "string", "default": ""},
            {"name": "squeeze", "type": "bool", "default": "true"},
            {"name": "tags", "type": "vector_string", "default": None},
        ],
    },
    "TensorflowPredict2D": {
        "inputs": [{"name": "signal", "type": "matrix_real"}],
        "outputs": [{"name": "predictions", "type": "vector_vector_real"}],
        "parameters": [
            {"name": "accumulate", "type": "bool", "default": "false"},
            {"name": "batchSize", "type": "integer", "default": "64"},
            {"name": "dimensions", "type": "integer", "default": "200"},
            {"name": "graphFilename", "type": "string", "default": ""},
            {"name": "input", "type": "string", "default": "model/Placeholder_"},
            {"name": "isTrainingName", "type": "string", "default": ""},
            {"name": "lastPatchMode", "type": "string", "default": "discard"},
            {"name": "output", "type": "string", "default": "model/Sigmoid_"},
            {"name": "patchHopSize", "type": "integer", "default": "1"},
            {"name": "patchSize", "type": "integer", "default": "1"},
            {"name": "savedModel", "type": "string", "default": ""},
        ],
    },
    "TensorflowPredictCREPE": {
        "inputs": [{"name": "signal", "type": "vector_real"}],
        "outputs": [{"name": "predictions", "type": "vector_vector_real"}],
        "parameters": [
            {"name": "batchSize", "type": "integer", "default": "16"},
            {"name": "graphFilename", "type": "string", "default": ""},
            {"name": "hopSize", "type": "real", "default": "10"},
            {"name": "input", "type": "string", "default": "frames"},
            {
                "name": "output",
                "type": "string",
                "default": "model/classifier/Sigmoid_",
            },
            {"name": "savedModel", "type": "string", "default": ""},
        ],
    },
    "TensorflowPredictEffnetDiscogs": {
        "inputs": [{"name": "signal", "type": "vector_real"}],
        "outputs": [{"name": "predictions", "type": "vector_vector_real"}],
        "parameters": [
            {"name": "batchSize", "type": "integer", "default": "64"},
            {"name": "graphFilename", "type": "string", "default": ""},
            {
                "name": "input",
                "type": "string",
                "default": "serving_default_melspectrogram",
            },
            {"name": "lastBatchMode", "type": "string", "default": "same"},
            {"name": "lastPatchMode", "type": "string", "default": "discard"},
            {"name": "output", "type": "string", "default": "PartitionedCall"},
            {"name": "patchHopSize", "type": "integer", "default": "62"},
            {"name": "patchSize", "type": "integer", "default": "128"},
            {"name": "savedModel", "type": "string", "default": ""},
        ],
    },
    "TensorflowPredictFSDSINet": {
        "inputs": [{"name": "signal", "type": "vector_real"}],
        "outputs": [{"name": "predictions", "type": "vector_vector_real"}],
        "parameters": [
            {"name": "batchSize", "type": "integer", "default": "64"},
            {"name": "graphFilename", "type": "string", "default": ""},
            {"name": "input", "type": "string", "default": "x"},
            {"name": "lastPatchMode", "type": "string", "default": "discard"},
            {"name": "normalize", "type": "bool", "default": "true"},
            {
                "name": "output",
                "type": "string",
                "default": "model/predictions/Sigmoid_",
            },
            {"name": "patchHopSize", "type": "integer", "default": "50"},
            {"name": "savedModel", "type": "string", "default": ""},
        ],
    },
    "TensorflowPredictMAEST": {
        "inputs": [{"name": "signal", "type": "vector_real"}],
        "outputs": [{"name": "predictions", "type": "tensor_real"}],
        "parameters": [
            {"name": "batchSize", "type": "integer", "default": "1"},
            {"name": "graphFilename", "type": "string", "default": ""},
            {"name": "input", "type": "string", "default": "melspectrogram"},
            {"name": "isTrainingName", "type": "string", "default": ""},
            {"name": "lastPatchMode", "type": "string", "default": "discard"},
            {"name": "output", "type": "string", "default": "Identity"},
            {"name": "patchHopSize", "type": "integer", "default": "1875"},
            {"name": "patchSize", "type": "integer", "default": "1876"},
            {"name": "savedModel", "type": "string", "default": ""},
        ],
    },
    "TensorflowPredictMusiCNN": {
        "inputs": [{"name": "signal", "type": "vector_real"}],
        "outputs": [{"name": "predictions", "type": "vector_vector_real"}],
        "parameters": [
            {"name": "accumulate", "type": "bool", "default": "false"},
            {"name": "batchSize", "type": "integer", "default": "64"},
            {"name": "graphFilename", "type": "string", "default": ""},
            {"name": "input", "type": "string", "default": "model/Placeholder"},
            {"name": "isTrainingName", "type": "string", "default": ""},
            {"name": "lastPatchMode", "type": "string", "default": "discard"},
            {"name": "output", "type": "string", "default": "model/Sigmoid"},
            {"name": "patchHopSize", "type": "integer", "default": "93"},
            {"name": "patchSize", "type": "integer", "default": "187"},
            {"name": "savedModel", "type": "string", "default": ""},
        ],
    },
    "TensorflowPredictTempoCNN": {
        "inputs": [{"name": "signal", "type": "vector_real"}],
        "outputs": [{"name": "predictions", "type": "vector_vector_real"}],
        "parameters": [
            {"name": "batchSize", "type": "integer", "default": "16"},
            {"name": "graphFilename", "type": "string", "default": ""},
            {"name": "input", "type": "string", "default": "input"},
            {"name": "lastPatchMode", "type": "string", "default": "discard"},
            {"name": "output", "type": "string", "default": "output"},
            {"name": "patchHopSize", "type": "integer", "default": "128"},
            {"name": "savedModel", "type": "string", "default": ""},
        ],
    },
    "TensorflowPredictVGGish": {
        "inputs": [{"name": "signal", "type": "vector_real"}],
        "outputs": [{"name": "predictions", "type": "vector_vector_real"}],
        "parameters": [
            {"name": "accumulate", "type": "bool", "default": "false"},
            {"name": "batchSize", "type": "integer", "default": "64"},
            {"name": "graphFilename", "type": "string", "default": ""},
            {"name": "input", "type": "string", "default": "model/Placeholder"},
            {"name": "isTrainingName", "type": "string", "default": ""},
            {"name": "lastPatchMode", "type": "string", "default": "discard"},
            {"name": "output", "type": "string", "default": "model/Sigmoid"},
            {"name": "patchHopSize", "type": "integer", "default": "93"},
            {"name": "patchSize", "type": "integer", "default": "96"},
            {"name": "savedModel", "type": "string", "default": ""},
        ],
    },
}

_ESSENTIA_TO_PY: dict[str, str] = {
    "real": "float",
    "integer": "int",
    "string": "str",
    "bool": "bool",
    "vector_real": "np.ndarray",
    "vector_integer": "list[int]",
    "vector_string": "list[str]",
    "matrix_real": "np.ndarray",
    "vector_vector_real": "np.ndarray",
    "vector_stereosample": "np.ndarray",
    "stereosample": "tuple[float, float]",
    "pool": "Pool",
    "tensor_real": "np.ndarray",
    "vector_complex": "np.ndarray",
    "complex": "complex",
}


def _py_type(essentia_type: str) -> str:
    return _ESSENTIA_TO_PY.get(essentia_type, "Any")


def _format_default(ptype: str, default: str | None) -> str:
    if default is None:
        return ""
    if ptype == "string":
        if not (default.startswith('"') or default.startswith("'")):
            return f' = "{default}"'
        return f" = {default}"
    if ptype == "bool":
        return f" = {default.capitalize()}"
    try:
        float(default)
        return f" = {default}"
    except ValueError:
        return f" = {default!r}"


def _return_annotation(outputs: list[dict]) -> str:
    if not outputs:
        return "None"
    types = [_py_type(o["type"]) for o in outputs]
    if len(types) == 1:
        return types[0]
    return f"tuple[{', '.join(types)}]"


def _class_stub(name: str, struct: dict) -> str:
    outputs = struct.get("outputs", [])
    params = struct.get("parameters", [])
    lines = [f"class {name}:"]
    if params:
        parts = ["self"]
        for p in params:
            py_t = _py_type(p["type"])
            dflt = _format_default(p["type"], p.get("default"))
            parts.append(f"{p['name']}: {py_t}{dflt}")
        parts.append("**kwargs: Any")
        lines.append(f"    def __init__({', '.join(parts)}) -> None: ...")
    else:
        lines.append("    def __init__(self, **kwargs: Any) -> None: ...")
    lines.append(
        f"    def __call__(self, *args: Any) -> {_return_annotation(outputs)}: ..."
    )
    return "\n".join(lines)


def _collect_structs() -> dict[str, dict]:
    structs: dict[str, dict] = {}
    for name in _e.keys():
        try:
            structs[name] = _e.Algorithm(name).getStruct()
        except Exception:
            pass
    structs.update(_TF_STRUCTS)
    return structs


def generate() -> str:
    structs = _collect_structs()
    n_standard = len(structs) - len(_TF_STRUCTS)
    header = f"""\
# Auto-generated stubs for essentia.standard
# {n_standard} standard algorithms from Algorithm.getStruct(), \
{len(_TF_STRUCTS)} TensorflowPredict* from official docs.
# Regenerate: uv run python scripts/gen_essentia_stubs.py
# Do not edit manually.
from typing import Any
import numpy as np
from essentia import Pool

"""
    classes = [_class_stub(name, structs[name]) for name in sorted(structs)]
    return header + "\n\n".join(classes) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        metavar="PATH",
        help=f"output path (default: {DEFAULT_OUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the existing file differs from generated output",
    )
    args = parser.parse_args()

    content = generate()
    structs = _collect_structs()

    if args.check:
        if not args.out.exists():
            print(f"MISSING: {args.out}", file=sys.stderr)
            sys.exit(1)
        if args.out.read_text() != content:
            print(f"OUT OF DATE: {args.out}", file=sys.stderr)
            sys.exit(1)
        print(f"OK: {args.out} is up to date ({len(structs)} algorithms)")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content)
    n_tf = len(_TF_STRUCTS)
    n_std = len(structs) - n_tf
    try:
        display = args.out.relative_to(PROJECT_ROOT)
    except ValueError:
        display = args.out
    print(
        f"Written {display} "
        f"({n_std} standard + {n_tf} TF-only = {len(structs)} algorithms, "
        f"{args.out.stat().st_size // 1024} KB)"
    )


if __name__ == "__main__":
    main()
