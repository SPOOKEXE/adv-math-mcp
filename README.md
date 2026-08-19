# adv-math-mcp

Two layers, one MCP server over stdio.

The **CAS layer** is SymPy exposed as what a model cannot fake: parsing to stable handles,
equivalence and derivation checking, gradients and Hessians, solving, simplification, calculus,
linear algebra, number theory, probability, and numeric probing. The **contract layer** is a
registry of variables, formulas and assumptions that catches the cross-formula errors a CAS
structurally cannot see: one name with two meanings, a factor of two where definitions meet, a
result resting on an assumption relaxed three steps ago.

## House rules

- Never a bare boolean. Verdicts are `proved` | `disproved` (with a counterexample) | `unknown`.
- Witnesses, not verdicts. `audit` returns reproducible evidence and never says "consistent".
- Results that can be cheaply checked are checked: solutions substitute back, antiderivatives
  differentiate back, inverses multiply back.
- Empirical answers say so. `bounds` and sampled convexity are labelled, never presented as proof.
- Every call that can hang has a hard timeout, and a timeout is an honest `unknown`.
- Expressions parse through an allowlisted parser. `sympify` is never called on input.

## Tools

CAS: `parse`, `declare`, `check_equivalence`, `check_derivation`, `matrix_grad` (with
`hessian: true`), `check_grad`, `to_code`, `shape_check`, `solve` (equation, inequality, system,
diophantine, recurrence, ode), `simplify`, `calc` (integrate, limit, series, sum, product),
`linalg`, `numtheory`, `prob`, `eval` (evalf, root, bounds, convexity).

Contract: `define`, `list`, `audit` (shapes, units, orphans, numeric probes), `resolve`,
`fork`, `impact`. Plus `env` to save and load whole environments.

`render: true` on the symbolic tools adds LaTeX output; `steps: true` on `solve` and `calc`
adds checkpoint traces. Both are off by default so agent turns stay cheap.

## Run

```sh
uv run python -m math_mcp.server
uv run pytest
```

`server.toml` is the launch contract; its tool list is checked against the served schemas by
the test suite.
