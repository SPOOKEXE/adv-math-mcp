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

CAS and planning: `parse`, `declare`, `check_equivalence`, `check_derivation`, `matrix_grad` (with
`hessian: true`), `check_grad`, `to_code`, `shape_check`, `tensor_plan` (contractions, streaming,
checkpointing, liveness and roofline costs), `analyze` (stability, certified bounds, complexity,
error budgets and optimization), `solve` (equation, inequality, system, diophantine, recurrence,
ode), `simplify`, `calc` (integrate, limit, series, sum, product), `linalg` (including SVD,
conditioning and structure), `numtheory`, `prob`, `eval` (evalf, root, empirical bounds,
convexity).

Contract: `define`, `list`, `audit` (shapes, units, orphans, numeric probes), `resolve`,
`fork`, `impact`. Plus `env` to save and load whole environments.

`render: true` on the symbolic tools adds LaTeX output; `steps: true` on `solve` and `calc`
adds checkpoint traces. Both are off by default so agent turns stay cheap.

`tensor_plan` never allocates tensors at the requested sizes. It distinguishes resident input and
output bytes, materialised intermediates, fast-memory tile workspace, slow-memory traffic and
FLOPs, then returns non-dominated schedules. See [Tensor planning](docs/tensor-planning.md) for the
request format, supported operations and exactness limits.

## Run

```sh
uv run python -m math_mcp.server
uv run pytest
```

The test command uses two `pytest-xdist` workers by default. This keeps the symbolic timeout cases
isolated and avoids the CPU oversubscription that `-n auto` can cause. Use `-n 0` for a serial
debugging run.

`server.toml` is the launch contract; its tool list is checked against the served schemas by
the test suite.

## Add to a client

Every example runs the same command:

```sh
uv run --directory /path/to/adv-math-mcp python -m math_mcp.server
```

`uv` resolves and syncs `uv.lock` on launch, so there is no venv to activate and no install
step. Use `--directory` with an absolute path rather than the `--project .` in `server.toml`:
clients spawn the server from their own working directory, which is rarely this repo.

**Claude Code**

```sh
claude mcp add math -- uv run --directory /path/to/adv-math-mcp python -m math_mcp.server
```

Add `--scope project` to write `.mcp.json` in the current repo instead of user config.

**Codex** (`~/.codex/config.toml`)

```toml
[mcp_servers.math]
command = "uv"
args = ["run", "--directory", "/path/to/adv-math-mcp", "python", "-m", "math_mcp.server"]
```

**Claude Desktop** (`claude_desktop_config.json`), **Cursor** (`.cursor/mcp.json`), **Gemini
CLI** (`~/.gemini/settings.json`), and anything else taking the `mcpServers` shape:

```json
{
  "mcpServers": {
    "math": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/adv-math-mcp", "python", "-m", "math_mcp.server"]
    }
  }
}
```

**VS Code** (`.vscode/mcp.json`) uses `servers` and wants an explicit transport:

```json
{
  "servers": {
    "math": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/adv-math-mcp", "python", "-m", "math_mcp.server"]
    }
  }
}
```

On Windows, escape the path in JSON as `C:\\path\\to\\adv-math-mcp`, and give `uv` its full
path if it is not on the launcher's `PATH`.

To confirm the server answers before wiring it into a client:

```sh
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}' \
  | uv run --directory /path/to/adv-math-mcp python -m math_mcp.server
```

A `serverInfo` line naming `math` means the launch contract holds.
