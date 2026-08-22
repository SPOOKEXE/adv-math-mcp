# Tensor planning

`tensor_plan` answers a constrained question: under an explicit fast-memory budget, output
residency contract and numerical semantics, which tensor schedules are legal and which are not
dominated in both workspace and slow-memory traffic?

It is a symbolic planner, not a compiler or profiler. It does not allocate tensors of the supplied
size. FLOPs and byte traffic are deterministic model results; roofline latency is a lower bound,
not a wall-clock prediction.

## Contraction

```json
{
  "kind": "contraction",
  "spec": "mk,kn->mn",
  "inputs": ["a", "b"],
  "tensors": {
    "a": {"shape": [4096, 8192], "dtype": "bf16"},
    "b": {"shape": [8192, 4096], "dtype": "bf16"}
  },
  "memory": {"fast_bytes": 196608},
  "output": {"mode": "materialize"},
  "semantics": {"mode": "real_exact"},
  "tile_alignment": 16
}
```

The result includes compulsory input/output storage, naive program-order liveness, a fast-memory
workspace lower bound, generated tile-search coverage, and a Pareto frontier. A multiply-add is
counted as two FLOPs. The traffic model describes a deterministic tiled loop and deliberately does
not include allocator or kernel-launch overhead.

## Streaming attention

```json
{
  "kind": "pipeline",
  "dims": {"b": 2, "q": 4096, "k": 4096, "h": 32, "d": 128},
  "tensors": {
    "q": {"shape": ["b", "q", "h", "d"], "dtype": "bf16"},
    "k": {"shape": ["b", "k", "h", "d"], "dtype": "bf16"},
    "v": {"shape": ["b", "k", "h", "d"], "dtype": "bf16"}
  },
  "ops": [
    {"id": "scores", "op": "einsum", "spec": "bqhd,bkhd->bhqk", "inputs": ["q", "k"]},
    {"id": "prob", "op": "softmax", "inputs": ["scores"], "axis": "k"},
    {"id": "out", "op": "einsum", "spec": "bhqk,bkhd->bqhd", "inputs": ["prob", "v"]}
  ],
  "memory": {"fast_bytes": 196608}
}
```

The recognized online-attention plan keeps a running maximum, scaled normalizer and weighted-value
accumulator. It avoids materializing both `scores` and `prob`. This is exact over real arithmetic,
but tiling changes floating-point reduction order, so it does not promise bitwise identity.

Standalone `sum`, `mean`, `max`, `variance`, `logsumexp` and `softmax` reductions also expose their
bounded streaming state. Materialized softmax uses two passes because normalization is not known
until the first pass finishes. `window`/`convolution` operations report the required input tile,
overlapping halo, stride/dilation-aware output shape and the traffic caused by rereading halos.
Their transformed axis is named `<axis>_out` by default, or explicitly with `output_axis`, so a
shortened valid-convolution axis cannot be confused with its input axis.

## Checkpointing and hardware

Set `kind` to `checkpoint`, or add `backward: true`, to receive a saved-versus-recomputed activation
policy. The current optimizer is exact for its independent-activation knapsack model. It explicitly
labels arbitrary DAG dependency recomputation and allocator behavior as outside that proof.

An optional roofline model annotates every plan:

```json
{
  "hardware": {
    "peak_flops": 9.89e14,
    "bandwidth_bytes_per_s": 3.35e12
  }
}
```

The annotation reports the ridge point, memory/compute classification and a lower bound from the
larger of compute time and traffic time.

## Expression and matrix companions

`analyze` provides five modes:

- `stability`: cancellation/range/division findings and symbolic relative condition expressions.
- `rigorous_bounds`: a proved symbolic interval enclosure for the supported arithmetic/function
  subset. Unsupported functions and domain holes return `unknown`.
- `complexity`: exact scalar expression-tree counts and common-subexpression savings.
- `error_budget`: first-order absolute error propagation, explicitly excluding the higher-order
  remainder.
- `optimize`: proved global ranges for univariate expressions on a closed interval; multivariate
  calls return a certified enclosure and an `unknown` optimization verdict.

`linalg` additionally supports `svd`, `condition` and `structure`. Structure analysis distinguishes
proved zero, proved nonzero and unknown symbolic entries before suggesting diagonal, triangular,
banded or Toeplitz algorithms.

## Supported and refused behavior

Tensor operations are `einsum`, `matmul`, `elementwise`, `window`/`convolution`, `sum`, `mean`,
`max`, `variance`, `logsumexp` and `softmax`. Axes must be named explicitly. Einsum ellipses, inferred elementwise
broadcasting, arbitrary automatic stream-state discovery and generated GPU kernels are refused or
left as `unknown` rather than guessed.
