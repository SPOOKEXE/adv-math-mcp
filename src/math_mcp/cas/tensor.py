"""Symbolic tensor materialisation and execution planning.

This module plans; it does not allocate user-sized tensors or pretend to be a kernel compiler.
The useful answer is a memory/traffic Pareto frontier under an explicit output contract, not a
single "minimum memory" schedule that rereads every input one scalar at a time.

The planner has two memory levels. Inputs, materialised intermediates and resident outputs live
in slow memory (normally HBM); tiles and accumulators live in fast memory (normally SRAM/cache).
Every byte count is derived from named shapes, so the same axis cannot quietly mean two things.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from time import monotonic
from typing import Any

from .session import MathError

DTYPE_BYTES: dict[str, int] = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "fp8": 1,
    "float8": 1,
    "int16": 2,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int32": 4,
    "float32": 4,
    "fp32": 4,
    "int64": 8,
    "float64": 8,
    "fp64": 8,
}

EXACTNESS = frozenset({"real_exact", "floating_equivalent", "bitwise_same", "approximate"})
OUTPUT_MODES = frozenset({"materialize", "stream"})
KINDS = frozenset({"contraction", "pipeline", "checkpoint"})


@dataclass(frozen=True, slots=True)
class TensorValue:
    """One logical tensor and its semantic axes."""

    name: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    bytes_per_element: int
    source: bool = False

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.elements * self.bytes_per_element

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": list(self.axes),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "bytes_per_element": self.bytes_per_element,
            "elements": self.elements,
            "bytes": self.nbytes,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class TensorOp:
    """A normalized operation whose output has the operation id as its name."""

    id: str
    kind: str
    inputs: tuple[str, ...]
    output_axes: tuple[str, ...]
    input_axes: tuple[tuple[str, ...], ...] = ()
    reduction_axes: tuple[str, ...] = ()
    flops: int = 0
    axis: tuple[str, ...] = ()
    spec: str = ""
    window: int = 0
    stride: int = 1
    dilation: int = 1
    padding: str = "valid"

    def to_dict(self, output: TensorValue) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "op": self.kind,
            "inputs": list(self.inputs),
            "output": output.to_dict(),
            "reduction_axes": list(self.reduction_axes),
            "flops": self.flops,
            **({"spec": self.spec} if self.spec else {}),
        }
        if self.kind == "window":
            payload["window"] = {
                "axis": self.axis[0],
                "size": self.window,
                "stride": self.stride,
                "dilation": self.dilation,
                "padding": self.padding,
            }
        return payload


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MathError(f"`{name}` must be a positive integer")
    return value


def _dtype(raw: Any, fallback: str = "float32") -> tuple[str, int]:
    name = str(raw or fallback).lower()
    if name not in DTYPE_BYTES:
        raise MathError(f"unknown dtype `{name}`; supported: {', '.join(sorted(DTYPE_BYTES))}")
    return name, DTYPE_BYTES[name]


def _parse_dimensions(raw: Any) -> dict[str, int]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise MathError("`dims` must map axis names to positive integer sizes")
    return {str(name): _positive_int(value, f"dims.{name}") for name, value in raw.items()}


def _parse_tensor(name: str, raw: Any, dims: dict[str, int], default_dtype: str) -> TensorValue:
    if not isinstance(raw, dict):
        raise MathError(f"tensor `{name}` must be an object with a shape")
    shape_raw = raw.get("shape")
    if not isinstance(shape_raw, list) or not shape_raw:
        raise MathError(f"tensor `{name}` needs a non-empty shape")

    axes_raw = raw.get("axes")
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    if all(isinstance(item, str) for item in shape_raw):
        axes = tuple(str(item) for item in shape_raw)
        missing = [axis for axis in axes if axis not in dims]
        if missing:
            raise MathError(f"tensor `{name}` uses axis `{missing[0]}` but `dims` does not size it")
        shape = tuple(dims[axis] for axis in axes)
    else:
        shape = tuple(_positive_int(item, f"tensors.{name}.shape") for item in shape_raw)
        if axes_raw is None:
            axes = ()
        else:
            if not isinstance(axes_raw, list) or len(axes_raw) != len(shape):
                raise MathError(f"tensor `{name}` axes must have the same length as its shape")
            axes = tuple(str(axis) for axis in axes_raw)
            for axis, size in zip(axes, shape):
                existing = dims.get(axis)
                if existing is not None and existing != size:
                    raise MathError(f"axis `{axis}` is {existing} elsewhere but {size} in tensor `{name}`")
                dims[axis] = size

    if axes and len(set(axes)) != len(axes):
        raise MathError(f"tensor `{name}` repeats an axis; diagonal tensors are not supported yet")

    dtype_name, natural_bytes = _dtype(raw.get("dtype"), default_dtype)
    bytes_per_element = raw.get("bytes_per_element", natural_bytes)
    bytes_per_element = _positive_int(bytes_per_element, f"tensors.{name}.bytes_per_element")
    return TensorValue(name, axes, shape, dtype_name, bytes_per_element, source=True)


def _bind_axes(tensor: TensorValue, labels: tuple[str, ...], dims: dict[str, int]) -> TensorValue:
    if len(labels) != len(tensor.shape):
        raise MathError(
            f"tensor `{tensor.name}` has rank {len(tensor.shape)} but its einsum operand `{''.join(labels)}` has rank {len(labels)}"
        )
    if tensor.axes and tensor.axes != labels:
        raise MathError(f"tensor `{tensor.name}` declares axes {list(tensor.axes)} but the contraction uses {list(labels)}")
    for axis, size in zip(labels, tensor.shape):
        existing = dims.get(axis)
        if existing is not None and existing != size:
            raise MathError(f"axis `{axis}` is {existing} elsewhere but {size} in tensor `{tensor.name}`")
        dims[axis] = size
    if tensor.axes:
        return tensor
    return TensorValue(tensor.name, labels, tensor.shape, tensor.dtype, tensor.bytes_per_element, tensor.source)


def _parse_einsum(spec: str) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    if "..." in spec:
        raise MathError("einsum ellipses are not supported; name every axis so the memory model is auditable")
    if "->" in spec:
        left, right = spec.split("->", 1)
        output = tuple(right.strip())
    else:
        left = spec
        counts: dict[str, int] = {}
        for character in left.replace(",", ""):
            counts[character] = counts.get(character, 0) + 1
        output = tuple(sorted(axis for axis, count in counts.items() if count == 1))
    operands = tuple(tuple(part.strip()) for part in left.split(","))
    if not operands or any(not operand for operand in operands):
        raise MathError("einsum spec needs one non-empty axis string per operand")
    if len(set(output)) != len(output):
        raise MathError("einsum output repeats an axis")
    unknown = set(output) - {axis for operand in operands for axis in operand}
    if unknown:
        raise MathError(f"einsum output axis `{min(unknown)}` appears in no input")
    return operands, output


def _contraction_flops(operands: tuple[tuple[str, ...], ...], dims: dict[str, int]) -> int:
    axes = {axis for operand in operands for axis in operand}
    # Multiply-add is two FLOPs. This is exact for the two-operand dense contractions accepted
    # by the tiler and a conventional estimate for higher-arity einsums.
    factor = 2 if len(operands) >= 2 else 1
    return factor * math.prod(dims[axis] for axis in axes)


def _normalize_einsum(raw: dict[str, Any], tensors: dict[str, TensorValue], dims: dict[str, int]) -> tuple[TensorOp, TensorValue]:
    op_id = str(raw.get("id", "")).strip()
    if not op_id:
        raise MathError("every operation needs a non-empty `id`")
    spec = str(raw.get("spec", ""))
    operands, output_axes = _parse_einsum(spec)
    input_names_raw = raw.get("inputs")
    if not isinstance(input_names_raw, list) or len(input_names_raw) != len(operands):
        raise MathError(f"einsum `{op_id}` needs one input name per operand")
    input_names = tuple(str(name) for name in input_names_raw)
    if op_id in tensors:
        raise MathError(f"operation id `{op_id}` is already a tensor name")

    bound: list[TensorValue] = []
    for name, labels in zip(input_names, operands):
        if name not in tensors:
            raise MathError(f"operation `{op_id}` refers to unknown input `{name}`")
        tensor = _bind_axes(tensors[name], labels, dims)
        tensors[name] = tensor
        bound.append(tensor)

    output_shape = tuple(dims[axis] for axis in output_axes)
    dtype_name, dtype_bytes = _dtype(raw.get("dtype"), bound[0].dtype)
    output = TensorValue(op_id, output_axes, output_shape, dtype_name, dtype_bytes)
    all_axes = {axis for operand in operands for axis in operand}
    reduction_axes = tuple(sorted(all_axes - set(output_axes)))
    op = TensorOp(op_id, "einsum", input_names, output_axes, operands, reduction_axes, _contraction_flops(operands, dims), spec=spec)
    return op, output


def _normalize_reduction(raw: dict[str, Any], tensors: dict[str, TensorValue]) -> tuple[TensorOp, TensorValue]:
    op_id = str(raw.get("id", "")).strip()
    kind = str(raw.get("op", ""))
    inputs_raw = raw.get("inputs")
    if not op_id or not isinstance(inputs_raw, list) or len(inputs_raw) != 1:
        raise MathError(f"`{kind}` needs an id and exactly one input")
    input_name = str(inputs_raw[0])
    if input_name not in tensors:
        raise MathError(f"operation `{op_id}` refers to unknown input `{input_name}`")
    if op_id in tensors:
        raise MathError(f"operation id `{op_id}` is already a tensor name")
    source = tensors[input_name]
    axis_raw = raw.get("axis")
    axes = (str(axis_raw),) if isinstance(axis_raw, str) else tuple(str(axis) for axis in (axis_raw or ()))
    if not axes:
        raise MathError(f"`{kind}` needs an axis")
    missing = set(axes) - set(source.axes)
    if missing:
        raise MathError(f"operation `{op_id}` names absent axis `{min(missing)}`")

    preserves_shape = kind == "softmax"
    output_axes = source.axes if preserves_shape else tuple(axis for axis in source.axes if axis not in axes)
    output_shape = source.shape if preserves_shape else tuple(size for axis, size in zip(source.axes, source.shape) if axis not in axes)
    output = TensorValue(op_id, output_axes, output_shape, source.dtype, source.bytes_per_element)
    output_elements = output.elements
    input_elements = source.elements
    costs = {"sum": 1, "mean": 2, "max": 1, "logsumexp": 4, "variance": 4, "softmax": 5}
    flops = max(1, costs[kind] * input_elements - output_elements)
    return TensorOp(op_id, kind, (input_name,), output_axes, reduction_axes=axes, flops=flops, axis=axes), output


def _normalize_elementwise(raw: dict[str, Any], tensors: dict[str, TensorValue]) -> tuple[TensorOp, TensorValue]:
    op_id = str(raw.get("id", "")).strip()
    inputs_raw = raw.get("inputs")
    if not op_id or not isinstance(inputs_raw, list) or not inputs_raw:
        raise MathError("`elementwise` needs an id and at least one input")
    names = tuple(str(name) for name in inputs_raw)
    if any(name not in tensors for name in names):
        missing = next(name for name in names if name not in tensors)
        raise MathError(f"operation `{op_id}` refers to unknown input `{missing}`")
    first = tensors[names[0]]
    for name in names[1:]:
        if tensors[name].axes != first.axes or tensors[name].shape != first.shape:
            raise MathError("elementwise broadcasting is not inferred; inputs must have identical named shapes")
    output = TensorValue(op_id, first.axes, first.shape, first.dtype, first.bytes_per_element)
    cost = _positive_int(raw.get("cost", 1), f"ops.{op_id}.cost")
    return TensorOp(op_id, "elementwise", names, first.axes, flops=cost * first.elements), output


def _normalize_window(raw: dict[str, Any], tensors: dict[str, TensorValue], dims: dict[str, int]) -> tuple[TensorOp, TensorValue]:
    op_id = str(raw.get("id", "")).strip()
    inputs_raw = raw.get("inputs")
    if not op_id or not isinstance(inputs_raw, list) or len(inputs_raw) != 1:
        raise MathError("`window` needs an id and exactly one input")
    input_name = str(inputs_raw[0])
    if input_name not in tensors:
        raise MathError(f"operation `{op_id}` refers to unknown input `{input_name}`")
    source = tensors[input_name]
    axis = str(raw.get("axis", ""))
    if axis not in source.axes:
        raise MathError(f"window operation `{op_id}` names absent axis `{axis}`")
    window = _positive_int(raw.get("window"), f"ops.{op_id}.window")
    stride = _positive_int(raw.get("stride", 1), f"ops.{op_id}.stride")
    dilation = _positive_int(raw.get("dilation", 1), f"ops.{op_id}.dilation")
    padding = str(raw.get("padding", "valid"))
    if padding not in {"valid", "same"}:
        raise MathError("window padding must be `valid` or `same`")
    source_length = dims[axis]
    effective = dilation * (window - 1) + 1
    if padding == "valid":
        if effective > source_length:
            raise MathError(f"effective window {effective} exceeds axis `{axis}` size {source_length}")
        output_length = (source_length - effective) // stride + 1
    else:
        output_length = math.ceil(source_length / stride)
    output_axis = str(raw.get("output_axis", f"{axis}_out"))
    existing_output = dims.get(output_axis)
    if existing_output is not None and existing_output != output_length:
        raise MathError(f"window output axis `{output_axis}` is already sized {existing_output}, not {output_length}")
    dims[output_axis] = output_length
    output_axes = tuple(output_axis if name == axis else name for name in source.axes)
    output_shape = tuple(output_length if name == axis else size for name, size in zip(source.axes, source.shape))
    output = TensorValue(op_id, output_axes, output_shape, source.dtype, source.bytes_per_element)
    return (
        TensorOp(
            op_id,
            "window",
            (input_name,),
            output_axes,
            reduction_axes=(axis,),
            flops=2 * output.elements * window,
            axis=(axis,),
            window=window,
            stride=stride,
            dilation=dilation,
            padding=padding,
        ),
        output,
    )


def _normalize_program(args: dict[str, Any]) -> tuple[dict[str, TensorValue], list[TensorOp], dict[str, int]]:
    dims = _parse_dimensions(args.get("dims"))
    default_dtype, _ = _dtype(args.get("dtype"), "float32")
    raw_tensors = args.get("tensors")
    if not isinstance(raw_tensors, dict) or not raw_tensors:
        raise MathError("`tensors` must be a non-empty mapping")
    tensors = {str(name): _parse_tensor(str(name), raw, dims, default_dtype) for name, raw in raw_tensors.items()}

    kind = str(args.get("kind") or ("pipeline" if args.get("ops") else "contraction"))
    if kind not in KINDS:
        raise MathError(f"unknown tensor plan kind `{kind}`; use contraction, pipeline or checkpoint")
    if kind == "contraction":
        raw_ops: Any = [
            {"id": str(args.get("id", "out")), "op": "einsum", "spec": args.get("spec", ""), "inputs": args.get("inputs", list(tensors))}
        ]
    else:
        raw_ops = args.get("ops")
    if not isinstance(raw_ops, list) or not raw_ops:
        raise MathError("a pipeline/checkpoint plan needs a non-empty `ops` list")

    operations: list[TensorOp] = []
    reduction_ops = {"sum", "mean", "max", "logsumexp", "variance", "softmax"}
    for raw in raw_ops:
        if not isinstance(raw, dict):
            raise MathError("each operation must be an object")
        op_kind = str(raw.get("op", ""))
        if op_kind in {"einsum", "matmul"}:
            if op_kind == "matmul" and "spec" not in raw:
                raw = {**raw, "spec": "mk,kn->mn"}
            op, output = _normalize_einsum(raw, tensors, dims)
        elif op_kind in reduction_ops:
            op, output = _normalize_reduction(raw, tensors)
        elif op_kind == "elementwise":
            op, output = _normalize_elementwise(raw, tensors)
        elif op_kind in {"window", "convolution"}:
            op, output = _normalize_window(raw, tensors, dims)
        else:
            raise MathError(
                f"unknown tensor op `{op_kind}`; supported: einsum, matmul, elementwise, window, "
                "sum, mean, max, variance, logsumexp, softmax"
            )
        tensors[output.name] = output
        operations.append(op)
    return tensors, operations, dims


def _final_outputs(args: dict[str, Any], operations: list[TensorOp]) -> tuple[str, ...]:
    raw = args.get("outputs")
    if raw is None:
        return (operations[-1].id,)
    if not isinstance(raw, list) or not raw:
        raise MathError("`outputs` must be a non-empty list")
    outputs = tuple(str(name) for name in raw)
    known = {op.id for op in operations}
    if not set(outputs) <= known:
        raise MathError(f"unknown requested output `{min(set(outputs) - known)}`")
    return outputs


def _fusable_intermediates(operations: list[TensorOp], allowed: set[str], outputs: set[str]) -> set[str]:
    if "fuse" not in allowed:
        return set()
    consumers: dict[str, list[TensorOp]] = {}
    for op in operations:
        for name in op.inputs:
            consumers.setdefault(name, []).append(op)
    return {op.id for op in operations if op.kind == "elementwise" and op.id not in outputs and len(consumers.get(op.id, ())) == 1}


def _liveness(
    tensors: dict[str, TensorValue], operations: list[TensorOp], outputs: tuple[str, ...], skipped: set[str] | None = None
) -> dict[str, Any]:
    skipped = skipped or set()
    last_use: dict[str, int] = {}
    for index, op in enumerate(operations):
        for name in op.inputs:
            last_use[name] = index
    for name in outputs:
        last_use[name] = len(operations)

    live = {name for name, tensor in tensors.items() if tensor.source}
    current = sum(tensors[name].nbytes for name in live)
    peak = current
    trace: list[dict[str, Any]] = [{"stage": "inputs", "live_bytes": current, "live": sorted(live)}]
    largest_intermediate = 0
    for index, op in enumerate(operations):
        output_bytes = 0 if op.id in skipped else tensors[op.id].nbytes
        current += output_bytes
        if output_bytes:
            live.add(op.id)
            largest_intermediate = max(largest_intermediate, output_bytes)
        peak = max(peak, current)
        allocated_peak = current
        for name in op.inputs:
            if last_use.get(name) == index and name in live:
                current -= tensors[name].nbytes
                live.remove(name)
        trace.append({"stage": op.id, "allocated_peak_bytes": allocated_peak, "live_bytes_after_free": current, "live": sorted(live)})
    return {"peak_live_bytes": peak, "largest_intermediate_bytes": largest_intermediate, "trace": trace}


def _tile_options(dim: int, alignment: int, limit: int = 7) -> tuple[int, ...]:
    values = {1, dim}
    start = min(dim, max(1, alignment))
    value = start
    while value < dim:
        values.add(value)
        value *= 2
    ordered = sorted(values)
    if len(ordered) <= limit:
        return tuple(ordered)
    indexes = {round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)}
    return tuple(ordered[index] for index in sorted(indexes))


def _pareto(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep plans not dominated in both workspace and slow-memory traffic."""
    ordered = sorted(candidates, key=lambda plan: (plan["workspace_bytes"], plan["slow_traffic_bytes"]))
    frontier: list[dict[str, Any]] = []
    best_traffic = math.inf
    for plan in ordered:
        traffic = plan["slow_traffic_bytes"]
        if traffic >= best_traffic:
            continue
        frontier.append(plan)
        best_traffic = traffic
    return frontier


def _sample_frontier(frontier: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(frontier) <= count:
        return frontier
    indexes = {round(index * (len(frontier) - 1) / (count - 1)) for index in range(count)}
    return [frontier[index] for index in sorted(indexes)]


def _contraction_plans(
    op: TensorOp,
    tensors: dict[str, TensorValue],
    dims: dict[str, int],
    fast_bytes: int,
    alignment: int,
    deadline_at: float,
    max_candidates: int,
    max_plans: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(op.inputs) != 2:
        return [], {"status": "unknown", "detail": "tiling currently supports two-operand contractions"}
    unique_axes = tuple(dict.fromkeys(axis for operand in op.input_axes for axis in operand))
    choices = [_tile_options(dims[axis], alignment) for axis in unique_axes]
    evaluated = 0
    timed_out = False
    feasible: list[dict[str, Any]] = []
    output = tensors[op.id]
    accumulator_bytes = max(4, output.bytes_per_element)

    for selected in itertools.product(*choices):
        if evaluated >= max_candidates or monotonic() >= deadline_at:
            timed_out = True
            break
        evaluated += 1
        tile = dict(zip(unique_axes, selected))
        input_workspace = sum(
            math.prod(tile[axis] for axis in labels) * tensors[name].bytes_per_element for name, labels in zip(op.inputs, op.input_axes)
        )
        output_elements = math.prod(tile[axis] for axis in op.output_axes) if op.output_axes else 1
        workspace = input_workspace + output_elements * accumulator_bytes
        if workspace > fast_bytes:
            continue

        loop_tiles = {axis: math.ceil(dims[axis] / tile[axis]) for axis in unique_axes}
        traffic = 0
        for name, labels in zip(op.inputs, op.input_axes):
            absent_reuse = math.prod(loop_tiles[axis] for axis in unique_axes if axis not in labels)
            traffic += tensors[name].nbytes * absent_reuse
        traffic += output.nbytes
        parallel_tiles = math.prod(loop_tiles[axis] for axis in op.output_axes) if op.output_axes else 1
        feasible.append(
            {
                "schedule": "tiled_contraction",
                "tile": tile,
                "workspace_bytes": workspace,
                "slow_traffic_bytes": traffic,
                "flops": op.flops,
                "arithmetic_intensity_flops_per_byte": op.flops / max(1, traffic),
                "parallel_output_tiles": parallel_tiles,
                "exactness": "real_exact",
                "floating_point_order_changed": any(tile[axis] < dims[axis] for axis in op.reduction_axes),
            }
        )

    minimum_workspace = sum(tensors[name].bytes_per_element for name in op.inputs) + accumulator_bytes
    search = {
        "status": "unknown" if timed_out else "complete-for-generated-tiles",
        "evaluated_candidates": evaluated,
        "timed_out": timed_out,
        "minimum_workspace_lower_bound_bytes": minimum_workspace,
        "lower_bound_scope": "one scalar from each input plus one accumulator in this execution model",
    }
    return _sample_frontier(_pareto(feasible), max_plans), search


def _attention_pattern(operations: list[TensorOp]) -> tuple[TensorOp, TensorOp, TensorOp] | None:
    if len(operations) < 3:
        return None
    for index in range(len(operations) - 2):
        score, softmax, output = operations[index : index + 3]
        if score.kind != "einsum" or softmax.kind != "softmax" or output.kind != "einsum":
            continue
        if softmax.inputs != (score.id,) or softmax.id not in output.inputs:
            continue
        return score, softmax, output
    return None


def _attention_plans(
    pattern: tuple[TensorOp, TensorOp, TensorOp],
    tensors: dict[str, TensorValue],
    dims: dict[str, int],
    fast_bytes: int,
    alignment: int,
    max_plans: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    score, softmax, output = pattern
    if len(score.inputs) != 2 or len(output.inputs) != 2 or len(softmax.axis) != 1:
        return [], []
    key_axis = softmax.axis[0]
    q_name, k_name = score.inputs
    probability_name = softmax.id
    v_name = output.inputs[0] if output.inputs[1] == probability_name else output.inputs[1]
    q_tensor, k_tensor, v_tensor = tensors[q_name], tensors[k_name], tensors[v_name]
    if key_axis not in k_tensor.axes or key_axis not in v_tensor.axes:
        return [], []

    contract_axes = set(q_tensor.axes) & set(k_tensor.axes) - set(score.output_axes)
    shared_axes = (set(q_tensor.axes) & set(k_tensor.axes)) - contract_axes
    query_axes = set(score.output_axes) - set(k_tensor.axes) - {key_axis}
    if not contract_axes or not query_axes:
        return [], []
    q_count = math.prod(dims[axis] for axis in query_axes)
    k_count = dims[key_axis]
    depth = math.prod(dims[axis] for axis in contract_axes)
    value_axes = set(v_tensor.axes) - {key_axis} - shared_axes
    value_depth = math.prod(dims[axis] for axis in value_axes) if value_axes else depth
    shared_count = math.prod(dims[axis] for axis in shared_axes) if shared_axes else 1
    accumulator_bytes = 4
    candidates: list[dict[str, Any]] = []
    for query_block in _tile_options(q_count, alignment):
        for key_block in _tile_options(k_count, alignment):
            # One shared batch/head slice is scheduled in fast memory at a time. Multiplying
            # workspace by every independent slice would model parallel work as simultaneously
            # live storage and grossly overstate SRAM. Traffic below still covers all slices.
            workspace = (
                query_block * depth * q_tensor.bytes_per_element
                + key_block * depth * k_tensor.bytes_per_element
                + key_block * value_depth * v_tensor.bytes_per_element
                + query_block * key_block * accumulator_bytes
                + query_block * value_depth * accumulator_bytes
                + 2 * query_block * accumulator_bytes
            )
            if workspace > fast_bytes:
                continue
            key_passes = math.ceil(k_count / key_block)
            traffic_per_shared = (
                k_count * (depth * k_tensor.bytes_per_element + value_depth * v_tensor.bytes_per_element)
                + key_passes * q_count * depth * q_tensor.bytes_per_element
                + 2 * key_passes * q_count * (value_depth + 2) * accumulator_bytes
                + q_count * value_depth * tensors[output.id].bytes_per_element
            )
            traffic = shared_count * traffic_per_shared
            candidates.append(
                {
                    "schedule": "online_tiled_attention",
                    "tile": {"query": query_block, "key": key_block},
                    "workspace_bytes": workspace,
                    "slow_traffic_bytes": traffic,
                    "flops": sum(op.flops for op in pattern),
                    "arithmetic_intensity_flops_per_byte": sum(op.flops for op in pattern) / max(1, traffic),
                    "parallel_output_tiles": shared_count * math.ceil(q_count / query_block),
                    "exactness": "real_exact",
                    "floating_point_order_changed": True,
                    "state": ["running_max", "scaled_normalizer", "weighted_value_accumulator"],
                    "merge_rule": "online max-shifted softmax reduction",
                }
            )
    avoided = [score.id, softmax.id]
    return _sample_frontier(_pareto(candidates), max_plans), avoided


def _reduction_plans(
    op: TensorOp, tensors: dict[str, TensorValue], dims: dict[str, int], fast_bytes: int, alignment: int, output_mode: str, max_plans: int
) -> list[dict[str, Any]]:
    source = tensors[op.inputs[0]]
    reduced_elements = math.prod(dims[axis] for axis in op.reduction_axes)
    output = tensors[op.id]
    state_scalars = {"sum": 1, "mean": 2, "max": 1, "variance": 3, "logsumexp": 2, "softmax": 2}[op.kind]
    candidates: list[dict[str, Any]] = []
    for chunk in _tile_options(reduced_elements, alignment):
        state_bytes = state_scalars * max(4, source.bytes_per_element)
        workspace = chunk * source.bytes_per_element + state_bytes
        if workspace > fast_bytes:
            continue
        passes = 2 if op.kind == "softmax" else 1
        traffic = passes * source.nbytes + (output.nbytes if output_mode == "materialize" else 0)
        candidates.append(
            {
                "schedule": "two_pass_streaming_softmax" if op.kind == "softmax" else "streamed_reduction",
                "operation": op.id,
                "tile": {"reduction_elements": chunk},
                "workspace_bytes": workspace,
                "slow_traffic_bytes": traffic,
                "flops": op.flops,
                "arithmetic_intensity_flops_per_byte": op.flops / max(1, traffic),
                "parallel_output_tiles": output.elements,
                "exactness": "real_exact",
                "floating_point_order_changed": chunk < reduced_elements,
                **STREAM_RULES[op.kind],
            }
        )
    # Equal traffic makes the smallest chunk dominate numerically, but the full chunk is a useful
    # throughput endpoint. Preserve both ends when they differ.
    if not candidates:
        return []
    candidates.sort(key=lambda plan: plan["workspace_bytes"])
    endpoints = [candidates[0]]
    if candidates[-1] is not candidates[0]:
        endpoints.append(candidates[-1])
    return endpoints[:max_plans]


def _window_plans(
    op: TensorOp,
    tensors: dict[str, TensorValue],
    dims: dict[str, int],
    fast_bytes: int,
    alignment: int,
    output_mode: str,
    max_plans: int,
) -> list[dict[str, Any]]:
    source = tensors[op.inputs[0]]
    output = tensors[op.id]
    axis = op.axis[0]
    source_length = dims[axis]
    output_length = output.shape[source.axes.index(axis)]
    outer = source.elements // source_length
    effective = op.dilation * (op.window - 1) + 1
    candidates: list[dict[str, Any]] = []
    for output_chunk in _tile_options(output_length, alignment):
        input_chunk = min(source_length, (output_chunk - 1) * op.stride + effective)
        workspace = input_chunk * source.bytes_per_element + output_chunk * max(4, output.bytes_per_element)
        if workspace > fast_bytes:
            continue
        chunks = math.ceil(output_length / output_chunk)
        traffic = outer * chunks * input_chunk * source.bytes_per_element
        if output_mode == "materialize":
            traffic += output.nbytes
        candidates.append(
            {
                "schedule": "windowed_stencil",
                "operation": op.id,
                "tile": {"output_elements": output_chunk, "input_elements_with_halo": input_chunk},
                "workspace_bytes": workspace,
                "slow_traffic_bytes": traffic,
                "flops": op.flops,
                "arithmetic_intensity_flops_per_byte": op.flops / max(1, traffic),
                "parallel_output_tiles": outer * chunks,
                "exactness": "real_exact",
                "floating_point_order_changed": False,
                "state": {"halo_elements": max(0, effective - op.stride)},
            }
        )
    return _sample_frontier(_pareto(candidates), max_plans)


def _annotate_hardware(plans: list[dict[str, Any]], raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise MathError("`hardware` must be an object with peak_flops and bandwidth_bytes_per_s")
    try:
        peak_flops = float(raw["peak_flops"])
        bandwidth = float(raw["bandwidth_bytes_per_s"])
    except (KeyError, TypeError, ValueError) as error:
        raise MathError("`hardware` needs numeric peak_flops and bandwidth_bytes_per_s") from error
    if not math.isfinite(peak_flops) or not math.isfinite(bandwidth) or peak_flops <= 0 or bandwidth <= 0:
        raise MathError("hardware rates must be positive finite numbers")
    ridge = peak_flops / bandwidth
    for plan in plans:
        compute_seconds = plan["flops"] / peak_flops
        traffic_seconds = plan["slow_traffic_bytes"] / bandwidth
        intensity = plan["flops"] / max(1, plan["slow_traffic_bytes"])
        plan["roofline"] = {
            "ridge_flops_per_byte": ridge,
            "classification": "memory-bound" if intensity < ridge else "compute-bound",
            "latency_lower_bound_seconds": max(compute_seconds, traffic_seconds),
            "compute_seconds": compute_seconds,
            "traffic_seconds": traffic_seconds,
        }
    return {"peak_flops": peak_flops, "bandwidth_bytes_per_s": bandwidth, "ridge_flops_per_byte": ridge}


STREAM_RULES: dict[str, dict[str, Any]] = {
    "sum": {"state": ["sum"], "merge": "s = s_left + s_right", "exactness": "real_exact"},
    "mean": {
        "state": ["count", "sum"],
        "merge": "(n, s) = (n_left + n_right, s_left + s_right)",
        "finalize": "s / n",
        "exactness": "real_exact",
    },
    "max": {"state": ["max"], "merge": "m = max(m_left, m_right)", "exactness": "real_exact"},
    "variance": {"state": ["count", "mean", "M2"], "merge": "parallel Welford merge", "exactness": "real_exact"},
    "logsumexp": {"state": ["running_max", "scaled_sum"], "merge": "max-shift and rescale both partial sums", "exactness": "real_exact"},
    "softmax": {
        "state": ["running_max", "scaled_normalizer"],
        "merge": "max-shift and rescale both partial normalizers",
        "exactness": "real_exact",
        "note": "materializing every normalized element still requires the output or another pass",
    },
}


def _checkpoint_policy(
    tensors: dict[str, TensorValue], operations: list[TensorOp], outputs: tuple[str, ...], budget: int
) -> dict[str, Any]:
    candidates = [(op.id, tensors[op.id].nbytes, op.flops) for op in operations if op.id not in outputs]
    states: list[tuple[int, int, tuple[str, ...]]] = [(0, 0, ())]
    exact = True
    for name, size, cost in candidates:
        expanded = states + [(used + size, saved + cost, names + (name,)) for used, saved, names in states if used + size <= budget]
        expanded.sort(key=lambda row: (row[0], -row[1]))
        pruned: list[tuple[int, int, tuple[str, ...]]] = []
        best = -1
        for state in expanded:
            if state[1] <= best:
                continue
            pruned.append(state)
            best = state[1]
        if len(pruned) > 4096:
            exact = False
            indexes = {round(i * (len(pruned) - 1) / 4095) for i in range(4096)}
            pruned = [pruned[index] for index in sorted(indexes)]
        states = pruned
    chosen = max(states, key=lambda row: (row[1], -row[0]))
    saved_names = set(chosen[2])
    total_recomputable = sum(cost for _, _, cost in candidates)
    return {
        "method": "independent-activation knapsack",
        "status": "proved-optimal-in-model" if exact else "heuristic-frontier-trimmed",
        "memory_budget_bytes": budget,
        "saved_activation_bytes": chosen[0],
        "saved": sorted(saved_names),
        "recompute": sorted(name for name, _, _ in candidates if name not in saved_names),
        "extra_forward_flops": total_recomputable - chosen[1],
        "caveat": "DAG dependency recomputation and allocator overhead are outside this independent-activation model",
    }


def tensor_plan(args: dict[str, Any]) -> dict[str, Any]:
    """Plan tensor materialisation without allocating tensors of the requested size."""
    tensors, operations, dims = _normalize_program(args)
    kind = str(args.get("kind") or ("pipeline" if args.get("ops") else "contraction"))
    outputs = _final_outputs(args, operations)

    output_raw = args.get("output", {})
    if not isinstance(output_raw, dict):
        raise MathError("`output` must be an object")
    output_mode = str(output_raw.get("mode", "materialize"))
    if output_mode not in OUTPUT_MODES:
        raise MathError("output mode must be `materialize` or `stream`")

    semantics_raw = args.get("semantics", {})
    if not isinstance(semantics_raw, dict):
        raise MathError("`semantics` must be an object")
    exactness = str(semantics_raw.get("mode", "real_exact"))
    if exactness not in EXACTNESS:
        raise MathError(f"unknown semantics mode `{exactness}`")

    memory_raw = args.get("memory", {})
    if not isinstance(memory_raw, dict):
        raise MathError("`memory` must be an object")
    default_fast = args.get("memory_budget", 64 * 1024 * 1024)
    fast_bytes = _positive_int(memory_raw.get("fast_bytes", default_fast), "memory.fast_bytes")
    slow_bytes_raw = memory_raw.get("slow_bytes")
    slow_bytes = _positive_int(slow_bytes_raw, "memory.slow_bytes") if slow_bytes_raw is not None else None
    alignment = _positive_int(args.get("tile_alignment", 16), "tile_alignment")
    max_candidates = _positive_int(args.get("max_candidates", 100_000), "max_candidates")
    max_candidates = min(max_candidates, 1_000_000)
    max_plans = min(_positive_int(args.get("max_plans", 8), "max_plans"), 64)
    try:
        timeout = float(args.get("timeout", 2.0))
    except (TypeError, ValueError) as error:
        raise MathError("`timeout` must be a positive finite number") from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise MathError("`timeout` must be a positive finite number")
    deadline_at = monotonic() + timeout

    allowed_raw = args.get("allowed", ["tile", "fuse", "recompute", "stream"])
    if not isinstance(allowed_raw, list):
        raise MathError("`allowed` must be a list")
    allowed = {str(item) for item in allowed_raw}
    final_set = set(outputs)
    fused = _fusable_intermediates(operations, allowed, final_set)
    naive = _liveness(tensors, operations, outputs)
    fused_liveness = _liveness(tensors, operations, outputs, fused) if fused else naive

    input_bytes = sum(tensor.nbytes for tensor in tensors.values() if tensor.source)
    output_bytes = sum(tensors[name].nbytes for name in outputs) if output_mode == "materialize" else 0
    intermediate_bytes = sum(tensors[op.id].nbytes for op in operations if op.id not in final_set)
    warnings: list[str] = []
    if exactness == "bitwise_same" and any(op.reduction_axes for op in operations if op.kind != "window"):
        warnings.append("tiling a reduction changes floating-point evaluation order; bitwise identity is not promised")

    plans: list[dict[str, Any]] = []
    searches: list[dict[str, Any]] = []
    avoided: list[str] = sorted(fused)
    attention = _attention_pattern(operations) if "fuse" in allowed and "tile" in allowed else None
    if attention is not None:
        attention_plans, attention_avoided = _attention_plans(attention, tensors, dims, fast_bytes, alignment, max_plans)
        if attention_plans:
            plans.extend(attention_plans)
            avoided = sorted(set(avoided) | set(attention_avoided))
            searches.append({"operation": attention[-1].id, "status": "template-enumerated"})

    if not plans and "tile" in allowed:
        contraction_ops = [op for op in operations if op.kind == "einsum"]
        operation_frontiers: list[tuple[TensorOp, list[dict[str, Any]]]] = []
        for op in contraction_ops:
            op_plans, search = _contraction_plans(op, tensors, dims, fast_bytes, alignment, deadline_at, max_candidates, max_plans)
            operation_frontiers.append((op, op_plans))
            searches.append({"operation": op.id, **search})
        if len(operation_frontiers) == 1:
            op, plans = operation_frontiers[0]
            for plan in plans:
                plan["operation"] = op.id
                plan["flops"] = sum(item.flops for item in operations)
        elif operation_frontiers and all(frontier for _, frontier in operation_frontiers):
            combined: list[dict[str, Any]] = [
                {
                    "schedule": "tiled_pipeline",
                    "tiles": {},
                    "workspace_bytes": 0,
                    "slow_traffic_bytes": 0,
                    "flops": sum(item.flops for item in operations),
                    "parallel_output_tiles": 1,
                    "exactness": "real_exact",
                    "floating_point_order_changed": False,
                }
            ]
            for op, frontier in operation_frontiers:
                expanded: list[dict[str, Any]] = []
                for partial, local in itertools.product(combined, frontier):
                    expanded.append(
                        {
                            **partial,
                            "tiles": {**partial["tiles"], op.id: local["tile"]},
                            "workspace_bytes": max(partial["workspace_bytes"], local["workspace_bytes"]),
                            "slow_traffic_bytes": partial["slow_traffic_bytes"] + local["slow_traffic_bytes"],
                            "parallel_output_tiles": max(partial["parallel_output_tiles"], local["parallel_output_tiles"]),
                            "floating_point_order_changed": (
                                partial["floating_point_order_changed"] or local["floating_point_order_changed"]
                            ),
                        }
                    )
                combined = _sample_frontier(_pareto(expanded), max_plans)
            for plan in combined:
                plan["arithmetic_intensity_flops_per_byte"] = plan["flops"] / max(1, plan["slow_traffic_bytes"])
            plans = combined

    if not plans and "stream" in allowed:
        stream_frontiers: list[tuple[TensorOp, list[dict[str, Any]]]] = []
        for op in operations:
            if op.kind in STREAM_RULES:
                local_plans = _reduction_plans(op, tensors, dims, fast_bytes, alignment, output_mode, max_plans)
                searches.append({"operation": op.id, "status": "stream-template-enumerated"})
            elif op.kind == "window":
                local_plans = _window_plans(op, tensors, dims, fast_bytes, alignment, output_mode, max_plans)
                searches.append({"operation": op.id, "status": "window-template-enumerated"})
            else:
                continue
            stream_frontiers.append((op, local_plans))
        if len(stream_frontiers) == 1:
            plans = stream_frontiers[0][1]
            for plan in plans:
                plan["flops"] = sum(item.flops for item in operations)
        elif stream_frontiers and all(frontier for _, frontier in stream_frontiers):
            combined = [
                {
                    "schedule": "streamed_pipeline",
                    "stages": {},
                    "workspace_bytes": 0,
                    "slow_traffic_bytes": 0,
                    "flops": sum(item.flops for item in operations),
                    "parallel_output_tiles": 1,
                    "exactness": "real_exact",
                    "floating_point_order_changed": False,
                }
            ]
            for op, frontier in stream_frontiers:
                expanded = []
                for partial, stage_plan in itertools.product(combined, frontier):
                    expanded.append(
                        {
                            **partial,
                            "stages": {
                                **partial["stages"],
                                op.id: {"schedule": stage_plan["schedule"], "tile": stage_plan["tile"]},
                            },
                            "workspace_bytes": max(partial["workspace_bytes"], stage_plan["workspace_bytes"]),
                            "slow_traffic_bytes": partial["slow_traffic_bytes"] + stage_plan["slow_traffic_bytes"],
                            "parallel_output_tiles": max(partial["parallel_output_tiles"], stage_plan["parallel_output_tiles"]),
                            "floating_point_order_changed": (
                                partial["floating_point_order_changed"] or stage_plan["floating_point_order_changed"]
                            ),
                        }
                    )
                combined = _sample_frontier(_pareto(expanded), max_plans)
            for plan in combined:
                plan["arithmetic_intensity_flops_per_byte"] = plan["flops"] / max(1, plan["slow_traffic_bytes"])
            plans = combined

    if not plans:
        plans.append(
            {
                "schedule": "materialize-in-program-order",
                "workspace_bytes": fused_liveness["largest_intermediate_bytes"],
                "slow_traffic_bytes": input_bytes + intermediate_bytes + output_bytes,
                "flops": sum(op.flops for op in operations),
                "exactness": exactness,
            }
        )

    optimized_skipped = set(avoided)
    if output_mode == "stream" and all(plan["schedule"] != "materialize-in-program-order" for plan in plans):
        optimized_skipped.update(outputs)
    optimized_liveness = _liveness(tensors, operations, outputs, optimized_skipped)
    slow_feasible = slow_bytes is None or optimized_liveness["peak_live_bytes"] <= slow_bytes
    fast_feasible = any(plan["workspace_bytes"] <= fast_bytes for plan in plans)
    streamability = [{"operation": op.id, **STREAM_RULES[op.kind]} for op in operations if op.kind in STREAM_RULES]
    hardware = _annotate_hardware(plans, args.get("hardware"))
    result: dict[str, Any] = {
        "kind": kind,
        "status": "feasible" if slow_feasible and fast_feasible else "infeasible-under-budget",
        "semantics": exactness,
        "output_mode": output_mode,
        "dimensions": dict(sorted(dims.items())),
        "memory": {"fast_bytes": fast_bytes, "slow_bytes": slow_bytes},
        "hardware": hardware,
        "compulsory": {
            "input_bytes": input_bytes,
            "resident_output_bytes": output_bytes,
            "logical_output_bytes": sum(tensors[name].nbytes for name in outputs),
        },
        "tensors": {name: tensor.to_dict() for name, tensor in tensors.items()},
        "operations": [op.to_dict(tensors[op.id]) for op in operations],
        "naive": {**naive, "total_flops": sum(op.flops for op in operations), "materialized_intermediate_bytes": intermediate_bytes},
        "optimized_liveness": optimized_liveness,
        "materializations_avoided": avoided,
        "plans": plans,
        "search": searches,
        "streamability": streamability,
        "warnings": warnings,
        "cost_model": {
            "flop_convention": "multiply-add = 2 FLOPs",
            "traffic": "deterministic tiled-loop estimate, not a latency measurement",
            "allocator_overhead": "not modelled",
        },
    }
    if kind == "checkpoint" or bool(args.get("backward")):
        checkpoint_budget = slow_bytes if slow_bytes is not None else fast_bytes
        result["checkpoint"] = _checkpoint_policy(tensors, operations, outputs, checkpoint_budget)
    return result
