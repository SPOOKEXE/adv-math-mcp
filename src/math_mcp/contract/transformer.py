"""The worked example: one context per transformer variant.

Used as the integration fixture, because a registry only earns its keep on a system big enough
that a human loses track — and this one is exactly that. Parameter count, FLOPs per token,
activation memory, KV-cache size, muP scaling and init variance are each individually easy and
collectively a place where everyone has, at some point, been off by a factor of two.

**The tokens-versus-sequences ambiguity is resolved once, explicitly**, and that is the point of
the whole layer. ``B`` means batch-in-sequences to one formula and batch-in-tokens to another;
both formulas are correct; the system built from them is wrong by a factor of ``seq_len``.
Declaring ``batch_tokens`` and ``batch_seqs`` as separate variables with a relation between them
is what makes the two impossible to confuse.
"""

from __future__ import annotations

from .model import Assumption, Formula, Scope, Variable

VARIABLES: tuple[Variable, ...] = (
    Variable("d_model", "residual stream width", "integer", status="hyperparameter", constraints=("d_model > 0",)),
    Variable("n_heads", "attention heads per layer", "integer", status="hyperparameter", constraints=("n_heads > 0",)),
    Variable("d_head", "width of one attention head", "integer", status="derived"),
    Variable("n_layers", "transformer blocks", "integer", status="hyperparameter", constraints=("n_layers > 0",)),
    Variable("d_ff", "feed-forward inner width", "integer", status="hyperparameter"),
    Variable("seq_len", "tokens per sequence", "integer", status="hyperparameter", constraints=("seq_len > 0",)),
    # The ambiguity, resolved: two variables, one relation, no shared name.
    Variable(
        "batch_seqs",
        "sequences per batch",
        "integer",
        status="hyperparameter",
        units="sequences",
        constraints=("batch_seqs > 0",),
    ),
    Variable("batch_tokens", "tokens per batch", "integer", status="derived", units="tokens"),
    Variable("n_params", "trainable parameters", "integer", status="derived", units="parameters"),
    Variable("flops_per_token", "forward FLOPs for one token", "integer", status="derived", units="flops/token"),
    Variable("activation_bytes", "activation memory for one batch", "integer", status="derived", units="bytes"),
    Variable("kv_cache_bytes", "KV cache for one batch", "integer", status="derived", units="bytes"),
    Variable("bytes_per_elem", "bytes per stored element", "integer", status="hyperparameter", units="bytes"),
    # 16 = four Adam-family states in fp32. Named rather than inlined, because a naked `16`
    # added to bytes hides its bytes-per-parameter units; the units audit rightly objects.
    Variable(
        "opt_bytes_per_param",
        "optimiser and weight bytes per parameter",
        "integer",
        status="hyperparameter",
        units="bytes/parameters",
    ),
    Variable("peak_memory", "peak device memory", "integer", status="derived", units="bytes"),
    Variable("device_flops", "accelerator peak FLOPs per second", "real", status="hyperparameter", units="flops/s"),
    Variable("mfu", "model FLOPs utilisation", "real", status="hyperparameter"),
    Variable("step_time", "wall-clock seconds per optimiser step", "real", status="derived", units="seconds"),
    Variable("init_var", "initialisation variance for a weight", "real", status="derived"),
    Variable("lr_scale", "muP learning-rate multiplier", "real", status="derived"),
    Variable("base_width", "muP base width", "integer", status="hyperparameter", constraints=("base_width > 0",)),
)

FORMULAS: tuple[Formula, ...] = (
    Formula("head-width", "definition", "d_head = d_model / n_heads", provenance="standard"),
    # The relation that makes the ambiguity harmless.
    Formula("batch-conversion", "definition", "batch_tokens = batch_seqs * seq_len", provenance="explicit"),
    Formula(
        "param-count",
        "approximation",
        "n_params = 12 * n_layers * d_model**2",
        validity=("d_ff = 4 * d_model", "embedding and bias terms ignored"),
        error_term="O(vocab * d_model)",
        provenance="Kaplan et al. scaling laws",
        assumes=("ffn-ratio",),
    ),
    Formula(
        "flops-per-token",
        "approximation",
        "flops_per_token = 2 * n_params + 2 * n_layers * seq_len * d_model",
        validity=("forward pass only",),
        error_term="O(n_layers * d_model)",
        provenance="Kaplan et al.",
        assumes=("ffn-ratio",),
    ),
    Formula(
        "activation-memory",
        "approximation",
        "activation_bytes = batch_tokens * n_layers * d_model * 34 * bytes_per_elem",
        validity=("no activation checkpointing",),
        error_term="O(batch_tokens * n_layers * seq_len * n_heads)",
        provenance="Korthikanti et al.",
        assumes=("no-checkpointing",),
    ),
    Formula(
        "kv-cache",
        "definition",
        "kv_cache_bytes = 2 * batch_seqs * seq_len * n_layers * d_model * bytes_per_elem",
        provenance="standard",
    ),
    Formula("opt-bytes", "definition", "opt_bytes_per_param = 16", provenance="Adam: 4 states, 4 bytes each"),
    Formula(
        "peak-memory",
        "derived",
        "peak_memory = activation_bytes + kv_cache_bytes + opt_bytes_per_param * n_params",
        validity=("Adam states in fp32",),
        provenance="derived here",
    ),
    Formula(
        "step-time",
        "empirical-fit",
        "step_time = 3 * flops_per_token * batch_tokens / (device_flops * mfu)",
        validity=("mfu measured on this cluster, this model size",),
        error_term="MFU varies by 10-20% with sequence length and parallelism strategy",
        provenance="fitted from run logs",
        assumes=("ffn-ratio",),
    ),
    Formula("mup-lr", "definition", "lr_scale = base_width / d_model", provenance="Yang & Hu, muP"),
    Formula("init-variance", "definition", "init_var = 1 / d_model", provenance="muP"),
)

ASSUMPTIONS: tuple[Assumption, ...] = (
    Assumption("ffn-ratio", "d_ff = 4 * d_model", provenance="the standard transformer ratio"),
    Assumption("no-checkpointing", "activations are retained for the backward pass"),
    Assumption("iid-batches", "batch elements are independent and identically distributed"),
)


def build_scope(name: str = "default", scope: Scope | None = None) -> Scope:
    """Populate a scope with the baseline architecture."""
    target = scope or Scope(name)
    for variable in VARIABLES:
        target.define_variable(variable)
    for assumption in ASSUMPTIONS:
        target.define_assumption(assumption)
    for formula in FORMULAS:
        target.define_formula(formula)
    return target


def seed_mup_conflict(scope: Scope) -> Scope:
    """Plant a muP rule that disagrees with the init rule.

    A deliberately realistic error: both formulas are individually plausible, both cite a real
    paper, and the disagreement only exists where they meet. Catching it costs a numeric probe;
    not catching it costs a training run.
    """
    scope.define_formula(
        Formula(
            "mup-init",
            "definition",
            "init_var = base_width / d_model**2",
            provenance="misread from the muP appendix",
            assumes=("ffn-ratio",),
        )
    )
    return scope
