"""The MCP surface.

Twenty-four tools over stdio: seventeen CAS/planning, six contract, one for the environment. The
handlers are plain functions taking and
returning JSON-shaped data, so the suite exercises them directly rather than through a
subprocess — the transport is not the thing under test, and shelling out makes every test slow
and flaky.

``mcp`` is an optional dependency: both layers are usable, and testable, without it.

The contract layer's tools are **six verbs with a parameterised noun**. Tool count is a context
cost paid on every single turn, and ``define-variable`` + ``define-formula`` +
``define-assumption`` is three schemas to describe one operation.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .cas.algebra import simplify as simplify_expr
from .cas.algebra import solve as solve_kinds
from .cas.analysis import calc
from .cas.calculus import check_grad, hessian, matrix_grad, shape_check, to_code
from .cas.equivalence import batch_equivalence, check_derivation, check_equivalence
from .cas.insight import analyze
from .cas.linalg import linalg
from .cas.numeric import evaluate
from .cas.numtheory import numtheory
from .cas.prob import prob
from .cas.session import MathError, ParseError, Session, canonical_form, pretty
from .cas.tensor import tensor_plan
from .contract.model import (
    Assumption,
    ContractError,
    Formula,
    Scope,
    Variable,
    assumption_closure,
    audit,
    diff_contexts,
    impact,
    relax,
    resolve,
)
from .layout import ROOT_ENV, resolve_root

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "parse",
        "description": "Parse LaTeX/ASCII/SymPy into an opaque expr_id handle. Parse errors carry line and column.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "syntax": {"type": "string", "enum": ["auto", "latex", "ascii"]}},
            "required": ["text"],
        },
    },
    {
        "name": "declare",
        "description": "Declare or refine a symbol's assumptions (real, positive, integer, …).",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "assumptions": {"type": "object"}},
            "required": ["name", "assumptions"],
        },
    },
    {
        "name": "check_equivalence",
        "description": "proved | disproved (+counterexample) | unknown. Never a bare boolean. Accepts a batch.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "left": {"type": "string"},
                "right": {"type": "string"},
                "pairs": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                "samples": {"type": "integer"},
                "timeout": {"type": "number"},
            },
        },
    },
    {
        "name": "check_derivation",
        "description": "Check a chain of equalities; returns the index of the first invalid step.",
        "inputSchema": {
            "type": "object",
            "properties": {"steps": {"type": "array", "items": {"type": "string"}}, "timeout": {"type": "number"}},
            "required": ["steps"],
        },
    },
    {
        "name": "matrix_grad",
        "description": (
            "Differentiate with an explicit numerator/denominator layout flag. "
            "`hessian: true` returns the Hessian instead (scalar expressions; no layout needed)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "expr": {"type": "string"},
                "wrt": {"type": "array", "items": {"type": "string"}},
                "layout": {"type": "string", "enum": ["numerator", "denominator"]},
                "hessian": {"type": "boolean"},
            },
            "required": ["expr", "wrt"],
        },
    },
    {
        "name": "check_grad",
        "description": "Validate a claimed gradient against central finite differences on seeded inputs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expr": {"type": "string"},
                "claimed": {"type": "string"},
                "wrt": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["expr", "claimed", "wrt"],
        },
    },
    {
        "name": "to_code",
        "description": "Emit NumPy/PyTorch/JAX source with common subexpressions extracted.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expr": {"type": "string"},
                "target": {"type": "string", "enum": ["numpy", "torch", "jax"]},
                "name": {"type": "string"},
            },
            "required": ["expr"],
        },
    },
    {
        "name": "shape_check",
        "description": "Check an einsum expression over named dims; a failure names the offending axis.",
        "inputSchema": {
            "type": "object",
            "properties": {"spec": {"type": "string"}, "shapes": {"type": "object"}, "dims": {"type": "object"}},
            "required": ["spec", "shapes"],
        },
    },
    {
        "name": "tensor_plan",
        "description": (
            "Plan tensor contractions, streaming pipelines and activation checkpointing without allocating the tensors. "
            "Returns named-shape checks, peak live bytes, traffic/FLOPs, legal tiles and a memory/traffic Pareto frontier."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["contraction", "pipeline", "checkpoint"]},
                "spec": {"type": "string", "description": "For kind=contraction, e.g. mk,kn->mn."},
                "inputs": {"type": "array", "items": {"type": "string"}},
                "tensors": {"type": "object", "description": "Tensor name to {shape, axes?, dtype?}. Shapes may use names from dims."},
                "dims": {"type": "object", "description": "Named positive integer dimensions."},
                "ops": {"type": "array", "items": {"type": "object"}},
                "outputs": {"type": "array", "items": {"type": "string"}},
                "dtype": {"type": "string"},
                "memory": {"type": "object", "description": "{fast_bytes, slow_bytes?}."},
                "hardware": {"type": "object", "description": "Optional {peak_flops, bandwidth_bytes_per_s} roofline model."},
                "memory_budget": {"type": "integer", "description": "Shorthand for memory.fast_bytes."},
                "output": {"type": "object", "description": "{mode: materialize|stream}."},
                "semantics": {"type": "object", "description": "{mode: real_exact|floating_equivalent|bitwise_same|approximate}."},
                "allowed": {"type": "array", "items": {"type": "string"}},
                "tile_alignment": {"type": "integer"},
                "max_candidates": {"type": "integer"},
                "max_plans": {"type": "integer"},
                "backward": {"type": "boolean"},
                "timeout": {"type": "number"},
            },
            "required": ["tensors"],
        },
    },
    {
        "name": "analyze",
        "description": (
            "Analyze scalar numerical stability, prove interval enclosures, count expression complexity, "
            "propagate first-order error budgets, or optimize exactly over a univariate box."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["stability", "rigorous_bounds", "complexity", "error_budget", "optimize"]},
                "expr": {"type": "string"},
                "wrt": {"type": "array", "items": {"type": "string"}},
                "box": {"type": "object", "description": "Closed interval per variable."},
                "at": {"type": "object", "description": "Point substitutions."},
                "errors": {"type": "object", "description": "Absolute error per input variable."},
                "goal": {"type": "string", "enum": ["min", "max", "both"]},
                "timeout": {"type": "number"},
                "render": {"type": "boolean"},
            },
            "required": ["op", "expr"],
        },
    },
    {
        "name": "solve",
        "description": (
            "Solve equations, inequalities, systems, diophantine equations, recurrences or ODEs. "
            "Roots come back substituted and verified. `steps` and `render` (LaTeX) are opt-in."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["equation", "inequality", "system", "diophantine", "recurrence", "ode"]},
                "exprs": {"type": "array", "items": {"type": "string"}},
                "wrt": {"type": "array", "items": {"type": "string"}, "description": "Unknowns; for recurrence/ode: [function, variable]."},
                "given": {"type": "object", "description": 'Initial conditions, e.g. {"a(0)": "1"}.'},
                "steps": {"type": "boolean"},
                "render": {"type": "boolean"},
                "timeout": {"type": "number"},
            },
            "required": ["kind", "exprs"],
        },
    },
    {
        "name": "simplify",
        "description": "Run named rewrite strategies and rank candidates by operation count. All candidates are sound rewrites.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expr": {"type": "string"},
                "strategies": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["auto", "factor", "expand", "cancel", "together", "trig", "radical", "partfrac"]},
                },
                "wrt": {"type": "string", "description": "The variable, where a strategy needs one (partfrac)."},
                "render": {"type": "boolean"},
                "timeout": {"type": "number"},
            },
            "required": ["expr"],
        },
    },
    {
        "name": "calc",
        "description": (
            "Integrate, take limits, expand series, and evaluate sums/products. Antiderivatives are "
            "verified by differentiating back; series carry their error term; a limit whose sides disagree says so."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["integrate", "limit", "series", "sum", "product"]},
                "expr": {"type": "string"},
                "wrt": {"type": "string"},
                "bounds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "[low, high]; makes integrate definite, required for sum/product.",
                },
                "point": {"type": "string", "description": "Where limit/series expand."},
                "direction": {"type": "string", "enum": ["+", "-", "both"]},
                "order": {"type": "integer"},
                "steps": {"type": "boolean"},
                "render": {"type": "boolean"},
                "timeout": {"type": "number"},
            },
            "required": ["op", "expr", "wrt"],
        },
    },
    {
        "name": "linalg",
        "description": (
            "Symbolic matrix operations: det, rank, eigen, inverse, nullspace, solve, decompose, SVD, condition and structure. "
            "Inverses, solves and SVDs are verified by multiplying back."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["det", "rank", "eigen", "inverse", "nullspace", "solve", "decompose", "svd", "condition", "structure"],
                },
                "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                "rhs": {"type": "array", "items": {"type": "string"}, "description": "Right-hand side for op=solve."},
                "method": {"type": "string", "enum": ["LU", "QR", "cholesky"]},
                "vectors": {"type": "boolean", "description": "op=eigen: include eigenvectors."},
                "tolerance": {"type": "number", "description": "Numerical-rank threshold for op=svd."},
                "render": {"type": "boolean"},
                "timeout": {"type": "number"},
            },
            "required": ["op", "matrix"],
        },
    },
    {
        "name": "numtheory",
        "description": (
            "Integer operations: factorint, gcd, lcm, is_prime (composite answers carry a factor witness), "
            "totient, mod_inverse, crt over [residue, modulus] pairs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["factorint", "gcd", "lcm", "is_prime", "totient", "mod_inverse", "crt"]},
                "values": {"type": "array", "items": {"type": "string"}},
                "modulus": {"type": "string"},
                "pairs": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                    "description": "op=crt: [[residue, modulus], ...].",
                },
                "timeout": {"type": "number"},
            },
            "required": ["op"],
        },
    },
    {
        "name": "prob",
        "description": (
            "Distributions with symbolic parameters: expectation, variance, pdf, cdf, probability of a "
            "condition, and KL divergence. Families: normal, uniform, bernoulli, binomial, poisson, exponential, beta, gamma."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["expectation", "variance", "pdf", "cdf", "probability", "kl"]},
                "family": {"type": "string"},
                "params": {"type": "object"},
                "expr": {"type": "string", "description": "Expression in the variable name, e.g. `X**2`."},
                "at": {"type": "string", "description": "Evaluation point for pdf/cdf."},
                "condition": {"type": "string", "description": "op=probability: a comparison like `X > 1`."},
                "other": {"type": "object", "description": "op=kl: {family, params} of the second distribution."},
                "name": {"type": "string"},
                "render": {"type": "boolean"},
                "timeout": {"type": "number"},
            },
            "required": ["op", "family", "params"],
        },
    },
    {
        "name": "eval",
        "description": (
            "Numeric work: arbitrary-precision evalf, root finding from a start point, empirical bounds over a "
            "box (labelled empirical, never a proof), and convexity (proved symbolically or refuted with a witness point)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["evalf", "root", "bounds", "convexity"]},
                "expr": {"type": "string"},
                "at": {"type": "object", "description": "Substitutions for evalf."},
                "digits": {"type": "integer"},
                "start": {"type": "object", "description": "op=root: initial guess per variable."},
                "box": {"type": "object", "description": 'Interval per variable, e.g. {"x": [-1, 1]}.'},
                "wrt": {"type": "array", "items": {"type": "string"}},
                "samples": {"type": "integer"},
                "timeout": {"type": "number"},
            },
            "required": ["op", "expr"],
        },
    },
    {
        "name": "define",
        "description": "Define a variable, formula or assumption. Unknown symbols auto-register as provisional.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "noun": {"type": "string", "enum": ["variable", "formula", "assumption"]},
                "scope": {"type": "string"},
                "body": {"type": "object"},
            },
            "required": ["noun", "body"],
        },
    },
    {
        "name": "list",
        "description": "List variables, formulas or assumptions, with filtering and a summary mode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "noun": {"type": "string", "enum": ["variable", "formula", "assumption"]},
                "scope": {"type": "string"},
                "filter": {"type": "object"},
                "mode": {"type": "string", "enum": ["summary", "full"]},
            },
            "required": ["noun"],
        },
    },
    {
        "name": "audit",
        "description": "Tiered consistency check. Returns witnesses, never verdicts, and never reports `consistent`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "samples": {"type": "integer"},
                "roots": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "resolve",
        "description": "Solve for a target and return the path: solve order, blocks, and any approximation chain.",
        "inputSchema": {
            "type": "object",
            "properties": {"target": {"type": "string"}, "given": {"type": "object"}, "scope": {"type": "string"}},
            "required": ["target"],
        },
    },
    {
        "name": "fork",
        "description": "Fork a scope, or diff two scopes.",
        "inputSchema": {
            "type": "object",
            "properties": {"scope": {"type": "string"}, "name": {"type": "string"}, "diff_against": {"type": "string"}},
        },
    },
    {
        "name": "env",
        "description": (
            "Save, load, list, delete or clear the symbolic environment — declarations, scopes, "
            "formulas and assumptions — so one situation can be put down and another picked up."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["save", "load", "list", "delete", "clear"]},
                "name": {"type": "string", "description": "For save, load and delete."},
                "note": {"type": "string", "description": "For save. What this environment is for."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "impact",
        "description": "Downstream closure of a symbol, with staleness propagation. Also does assumption closure.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "formula": {"type": "string"},
                "relax": {"type": "string"},
                "mark_stale": {"type": "boolean"},
                "scope": {"type": "string"},
            },
        },
    },
]


#: Characters a saved name may use. Anything else is a path, and a path is not a name.
ENV_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: The on-disk shape, so a file written by an older build is recognised rather than misread.
ENV_FORMAT = 1


class MathServer:
    """The tool layer. One CAS session and a scope registry, both per server instance."""

    def __init__(self, root: Path | None = None) -> None:
        self.session = Session()
        self.scopes: dict[str, Scope] = {"default": Scope("default", self.session)}
        #: Where `env save` writes. Resolved lazily, so a server started outside a project can
        #: still do everything that does not touch the disk.
        self._root = root

    def scope(self, name: str | None = None) -> Scope:
        key = name or "default"
        if key not in self.scopes:
            raise ContractError(f"unknown scope `{key}`; fork one first")
        return self.scopes[key]

    # -- CAS ------------------------------------------------------------------

    def parse(self, args: dict[str, Any]) -> dict[str, Any]:
        handle, expr = self.session.parse(args["text"], syntax=args.get("syntax", "auto"))
        return {
            "expr_id": handle,
            "pretty": pretty(expr),
            "free_symbols": sorted(str(symbol) for symbol in expr.free_symbols),
            "canonical": pretty(canonical_form(expr)),
        }

    def declare(self, args: dict[str, Any]) -> dict[str, Any]:
        symbol = self.session.declare(args["name"], **args["assumptions"])
        return {"name": str(symbol), "assumptions": self.session.assumptions_of(args["name"])}

    def check_equivalence(self, args: dict[str, Any]) -> dict[str, Any]:
        if "pairs" in args:
            # One round trip for twenty independent checks; serialising them through the model
            # is twenty turns and no extra information.
            return {
                "results": batch_equivalence(
                    self.session,
                    [(pair[0], pair[1]) for pair in args["pairs"]],
                    samples=args.get("samples", 24),
                    timeout=args.get("timeout", 5.0),
                )
            }
        return check_equivalence(
            self.session, args["left"], args["right"], samples=args.get("samples", 24), timeout=args.get("timeout", 5.0)
        ).to_dict()

    def check_derivation(self, args: dict[str, Any]) -> dict[str, Any]:
        return check_derivation(self.session, args["steps"], timeout=args.get("timeout", 5.0)).to_dict()

    def matrix_grad(self, args: dict[str, Any]) -> dict[str, Any]:
        if args.get("hessian"):
            return hessian(self.session, args["expr"], args["wrt"])
        return matrix_grad(self.session, args["expr"], args["wrt"], layout=args.get("layout")).to_dict()

    def check_grad(self, args: dict[str, Any]) -> dict[str, Any]:
        return check_grad(self.session, args["expr"], args["claimed"], args["wrt"]).to_dict()

    def to_code(self, args: dict[str, Any]) -> dict[str, Any]:
        return to_code(self.session, args["expr"], target=args.get("target", "numpy"), name=args.get("name", "f")).to_dict()

    def shape_check(self, args: dict[str, Any]) -> dict[str, Any]:
        return shape_check(args["spec"], args["shapes"], args.get("dims")).to_dict()

    def tensor_plan(self, args: dict[str, Any]) -> dict[str, Any]:
        return tensor_plan(args)

    def analyze(self, args: dict[str, Any]) -> dict[str, Any]:
        return analyze(
            self.session,
            args["op"],
            args["expr"],
            wrt=args.get("wrt"),
            box=args.get("box"),
            at=args.get("at"),
            errors=args.get("errors"),
            goal=args.get("goal", "both"),
            timeout=args.get("timeout", 10.0),
            render=args.get("render", False),
        )

    def solve(self, args: dict[str, Any]) -> dict[str, Any]:
        return solve_kinds(
            self.session,
            args["kind"],
            args["exprs"],
            args.get("wrt"),
            given=args.get("given"),
            timeout=args.get("timeout", 5.0),
            steps=args.get("steps", False),
            render=args.get("render", False),
        ).to_dict()

    def simplify(self, args: dict[str, Any]) -> dict[str, Any]:
        return simplify_expr(
            self.session,
            args["expr"],
            args.get("strategies"),
            args.get("wrt"),
            timeout=args.get("timeout", 5.0),
            render=args.get("render", False),
        ).to_dict()

    def calc(self, args: dict[str, Any]) -> dict[str, Any]:
        return calc(
            self.session,
            args["op"],
            args["expr"],
            args["wrt"],
            bounds=args.get("bounds"),
            point=args.get("point"),
            direction=args.get("direction", "both"),
            order=args.get("order", 6),
            timeout=args.get("timeout", 10.0),
            steps=args.get("steps", False),
            render=args.get("render", False),
        ).to_dict()

    def linalg(self, args: dict[str, Any]) -> dict[str, Any]:
        return linalg(
            self.session,
            args["op"],
            args["matrix"],
            rhs=args.get("rhs"),
            method=args.get("method", "LU"),
            vectors=args.get("vectors", False),
            tolerance=args.get("tolerance", 1e-10),
            timeout=args.get("timeout", 10.0),
            render=args.get("render", False),
        )

    def numtheory(self, args: dict[str, Any]) -> dict[str, Any]:
        return numtheory(
            self.session,
            args["op"],
            args.get("values"),
            modulus=args.get("modulus"),
            pairs=args.get("pairs"),
            timeout=args.get("timeout", 10.0),
        )

    def prob(self, args: dict[str, Any]) -> dict[str, Any]:
        return prob(
            self.session,
            args["op"],
            args["family"],
            args["params"],
            expr=args.get("expr"),
            at=args.get("at"),
            condition=args.get("condition"),
            other=args.get("other"),
            name=args.get("name", "X"),
            timeout=args.get("timeout", 10.0),
            render=args.get("render", False),
        )

    def eval(self, args: dict[str, Any]) -> dict[str, Any]:
        return evaluate(
            self.session,
            args["op"],
            args["expr"],
            at=args.get("at"),
            digits=args.get("digits", 30),
            start=args.get("start"),
            box=args.get("box"),
            wrt=args.get("wrt"),
            samples=args.get("samples", 200),
            timeout=args.get("timeout", 10.0),
        )

    # -- contract -------------------------------------------------------------

    def define(self, args: dict[str, Any]) -> dict[str, Any]:
        scope = self.scope(args.get("scope"))
        noun = args["noun"]
        body = args["body"]

        if noun == "variable":
            variable = scope.define_variable(
                Variable(
                    name=body["name"],
                    semantics=body.get("semantics", ""),
                    domain=body.get("domain", "real"),
                    shape=tuple(body.get("shape", ())),
                    units=body.get("units", ""),
                    status=body.get("status", "free"),
                    aliases=tuple(body.get("aliases", ())),
                    provenance=body.get("provenance", ""),
                    constraints=tuple(body.get("constraints", ())),
                )
            )
            return {"defined": "variable", "variable": variable.to_dict()}

        if noun == "formula":
            formula, provisional = scope.define_formula(
                Formula(
                    id=body["id"],
                    kind=body.get("kind", "definition"),
                    expression=body["expression"],
                    validity=tuple(body.get("validity", ())),
                    provenance=body.get("provenance", ""),
                    status=body.get("status", "unverified"),
                    assumes=tuple(body.get("assumes", ())),
                    error_term=body.get("error_term", ""),
                )
            )
            return {"defined": "formula", "formula": formula.to_dict(), "auto_registered": provisional}

        if noun == "assumption":
            assumption = scope.define_assumption(
                Assumption(
                    id=body["id"], statement=body["statement"], provenance=body.get("provenance", ""), active=body.get("active", True)
                )
            )
            return {"defined": "assumption", "assumption": assumption.to_dict()}

        raise ContractError(f"unknown noun `{noun}`")

    def list(self, args: dict[str, Any]) -> dict[str, Any]:
        scope = self.scope(args.get("scope"))
        noun = args["noun"]
        filters = args.get("filter", {})
        mode = args.get("mode", "summary")

        source: dict[str, Any] = {"variable": scope.variables, "formula": scope.formulas, "assumption": scope.assumptions}[noun]

        entries = [item for item in source.values() if all(getattr(item, key, None) == value for key, value in filters.items())]

        if mode == "summary":
            # Dumping 200 relations into context defeats the purpose of having a registry.
            return {"noun": noun, "count": len(entries), "ids": sorted(getattr(item, "id", getattr(item, "name", "")) for item in entries)}
        return {"noun": noun, "count": len(entries), "entries": [item.to_dict() for item in entries]}

    def audit(self, args: dict[str, Any]) -> dict[str, Any]:
        return audit(self.scope(args.get("scope")), samples=args.get("samples", 12), roots=args.get("roots", ())).to_dict()

    def resolve(self, args: dict[str, Any]) -> dict[str, Any]:
        return resolve(self.scope(args.get("scope")), args["target"], args.get("given")).to_dict()

    def fork(self, args: dict[str, Any]) -> dict[str, Any]:
        scope = self.scope(args.get("scope"))
        if "diff_against" in args:
            return diff_contexts(scope, self.scope(args["diff_against"]))

        name = args["name"]
        if name in self.scopes:
            raise ContractError(f"scope `{name}` already exists")
        self.scopes[name] = scope.fork(name)
        return {"forked": name, "from": scope.name, "variables": len(self.scopes[name].variables)}

    # -- environment ----------------------------------------------------------

    @property
    def root(self) -> Path:
        """The directory saved environments live in, created on first use."""
        if self._root is None:
            self._root = resolve_root()
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    def _path(self, name: str) -> Path:
        """A name, as a file. Refused rather than sanitised — a sanitised name is a different one."""
        if not ENV_NAME.match(name):
            raise ContractError(f"`{name}` is not a usable environment name; use letters, digits, `.`, `-` and `_`")
        return self.root / f"{name}.json"

    def env(self, args: dict[str, Any]) -> dict[str, Any]:
        """Put one situation down and pick another up.

        **Everything, or it is not an environment.** A save carries the symbol declarations as
        well as every scope, because a scope whose `x` is positive in one session and
        unconstrained in the next is not the same scope however identical its records read. That
        is also why `load` replaces rather than merges: two half-environments sharing a symbol
        table is the failure this exists to prevent, and there is no answer to "which `x` wins"
        that is right more than half the time.

        **`clear` is `load` of nothing**, and it keeps `default` — a scope registry with no
        scopes is not a clean slate, it is a broken server that reports `unknown scope` for
        everything until someone forks one.
        """
        action = args.get("action", "")

        if action == "list":
            saved = []
            for path in sorted(self.root.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    # Listed anyway, marked. A file that will not parse is exactly what someone
                    # asking "what have I saved" needs to be told about.
                    saved.append({"name": path.stem, "unreadable": True})
                    continue
                saved.append(
                    {
                        "name": path.stem,
                        "note": data.get("note", ""),
                        "scopes": sorted(scope["name"] for scope in data.get("scopes", [])),
                        "symbols": len(data.get("declarations", {})),
                    }
                )
            return {"saved": saved, "directory": str(self.root)}

        if action == "clear":
            self.session = Session()
            self.scopes = {"default": Scope("default", self.session)}
            return {"cleared": True, "scopes": ["default"]}

        if action == "save":
            name = str(args.get("name", ""))
            payload = {
                "format": ENV_FORMAT,
                "note": str(args.get("note", "")),
                "declarations": self.session.declarations(),
                "scopes": [scope.to_dict() for scope in self.scopes.values()],
            }
            path = self._path(name)
            # Temp-and-rename: a half-written environment that still parses is worse than no
            # environment, because it loads and is quietly missing half the work.
            staging = path.with_suffix(".json.tmp")
            staging.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(staging, path)
            return {"saved": name, "path": str(path), "scopes": sorted(self.scopes), "symbols": len(payload["declarations"])}

        if action == "delete":
            path = self._path(str(args.get("name", "")))
            if not path.is_file():
                raise ContractError(f"no saved environment called `{path.stem}`")
            path.unlink()
            return {"deleted": path.stem}

        if action == "load":
            path = self._path(str(args.get("name", "")))
            if not path.is_file():
                raise ContractError(f"no saved environment called `{path.stem}`; `env list` shows what there is")
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ContractError(f"`{path.stem}` is not readable: {error}") from error

            stored = data.get("format", 0)
            if stored != ENV_FORMAT:
                raise ContractError(f"`{path.stem}` was written in format {stored}, and this build reads {ENV_FORMAT}")

            # Built to one side and swapped in, so a failure half way through leaves the session
            # it had rather than a mixture of two.
            session = Session()
            session.restore(data.get("declarations", {}))
            scopes = {}
            for record in data.get("scopes", []):
                scope = Scope.from_dict(record, session)
                scopes[scope.name] = scope
            if "default" not in scopes:
                scopes["default"] = Scope("default", session)

            self.session = session
            self.scopes = scopes
            return {
                "loaded": path.stem,
                "note": data.get("note", ""),
                "scopes": sorted(scopes),
                "symbols": len(data.get("declarations", {})),
            }

        raise ContractError(f"unknown action `{action}`; use save, load, list, delete or clear")

    def impact(self, args: dict[str, Any]) -> dict[str, Any]:
        scope = self.scope(args.get("scope"))
        if "relax" in args:
            return relax(scope, args["relax"])
        if "formula" in args:
            return assumption_closure(scope, args["formula"])
        return impact(scope, args["symbol"], mark_stale=args.get("mark_stale", False)).to_dict()

    # -- dispatch -------------------------------------------------------------

    @property
    def handlers(self) -> dict[str, ToolHandler]:
        return {
            "parse": self.parse,
            "declare": self.declare,
            "check_equivalence": self.check_equivalence,
            "check_derivation": self.check_derivation,
            "matrix_grad": self.matrix_grad,
            "check_grad": self.check_grad,
            "to_code": self.to_code,
            "shape_check": self.shape_check,
            "tensor_plan": self.tensor_plan,
            "analyze": self.analyze,
            "solve": self.solve,
            "simplify": self.simplify,
            "calc": self.calc,
            "linalg": self.linalg,
            "numtheory": self.numtheory,
            "prob": self.prob,
            "eval": self.eval,
            "define": self.define,
            "list": self.list,
            "audit": self.audit,
            "resolve": self.resolve,
            "fork": self.fork,
            "env": self.env,
            "impact": self.impact,
        }

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch, turning a domain error into a structured result.

        An exception crossing the MCP boundary becomes a transport error the model cannot
        act on; a parse error with a column position is something it can fix.
        """
        handler = self.handlers.get(name)
        if handler is None:
            return {"error": "unknown-tool", "message": f"no tool named `{name}`", "available": sorted(self.handlers)}
        try:
            return handler(args)
        except ParseError as error:
            return error.to_dict()
        except (MathError, ContractError) as error:
            return {"error": type(error).__name__, "message": str(error)}
        except KeyError as error:
            return {"error": "missing-argument", "message": f"required argument {error} was not given"}


def main() -> None:  # pragma: no cover - transport, exercised by hand
    """Serve over stdio. Imported lazily so the tool layer works without ``mcp`` installed."""
    import argparse
    import asyncio

    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    # Injected by the harness where possible, discovered otherwise — and resolved lazily inside
    # `MathServer`, so a server started outside a project still serves every tool that does not
    # touch the disk.
    parser = argparse.ArgumentParser(prog="math")
    parser.add_argument("--root", default=None, help="saved-environment directory; overrides $" + ROOT_ENV)
    root = parser.parse_args().root
    service = MathServer(Path(root).expanduser().resolve() if root else None)
    server: Server = Server("math")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [Tool(**schema) for schema in TOOL_SCHEMAS]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        result = service.call(name, arguments or {})
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    async def run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    main()
