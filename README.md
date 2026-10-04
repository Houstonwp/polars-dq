# polars-dq

An experimental, expression-based data-quality checker for Polars DataFrames.
This is a prototype, not a production-ready validation framework. The API may
change. Review the results and test your own data before relying on it.

## Quick start

Requires Python 3.11 or newer.

```sh
python -m pip install -e .
```

```python
import polars as pl

from polars_dq.actions import Action
from polars_dq.agent import Agent
from polars_dq.validation import Validation

agent = (
    Agent(pl.DataFrame({"amount": [10, -2, None]}), actions=Action(warn_atol=0.0))
    .with_validation(Validation(pl.col("amount") > 0, label="Positive amounts"))
    .interrogate()
)

result = agent.validation_set[0]
assert result.table_checked and not result.eval_error
assert (result.n, result.n_passed, result.n_failed) == (3, 1, 2)
assert result.warn is True
print(result.row_sample)  # Up to 10 failing source rows, including null results.
```

## Evaluation semantics

- Validations and standalone `ValidationStep` objects are enabled by default.
  Use `Validation(..., eval_active=False)` to skip a validation, including all
  its segment steps. A disabled step remains unchecked. Setting its
  `eval_active` to `True` enables it for a later interrogation.
- Each successfully checked step is evaluated once. Repeating `interrogate()`
  keeps its results. Create a new Agent/validation plan for new data.
- A validation expression must produce Boolean values. `True` passes;
  `False` and null fail. For a different null policy, express it explicitly,
  for example `(pl.col("amount") > 0).fill_null(True)`.
- A precondition filters the rows before validation. A list of segment column
  names, such as `segments=["region"]`, creates one step per group followed by
  an overall step for all rows that pass the precondition.
- Empty input has zero passed/failed counts, `all_passed=True`, and both
  fractions equal to `0.0`. This is not proof that required data exists; check
  `n` separately when an empty table is unacceptable.
- `*_atol` limits apply to the failed row count; `*_rtol` limits apply to the
  failed fraction. A flag becomes true when either configured limit is
  **strictly exceeded**. Explicit `0`/`0.0` means no failures are tolerated.
- A validation-level tolerance overrides the same Agent-level tolerance.
  Only `None` inherits the Agent value. An effective `None` disables that
  particular limit; with no limits configured all action flags are `False`.
- An evaluation error sets `eval_error=True`, retains the exception in
  `capture_stack`, disables the step, and leaves `table_checked=False`.
  Inspect these fields before using counts or pass/fail results. Warnings are
  recorded via `eval_warning` and the last warning in `capture_stack`.

## Prototype limits

The package currently handles in-memory Polars DataFrames and Boolean Polars
expressions. Results are exposed as `agent.validation_set`; there is no stable
report/export API. `warn`, `notify`, and `stop` are result flags only: they do
not send notifications, raise on bad data, or stop later validations. Sample
rows may contain sensitive source data; handle them accordingly.

There is no production-readiness or security certification, no locked
application environment, and no supported persistence or distributed execution
layer.

## Development

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e . 'pytest>=8.3.4' 'ruff>=0.9.2'
python -m pytest -q -W error
ruff check .
ruff format --check .
```

The tests include failing-data end-to-end checks, zero-tolerance overrides,
explicitly disabled steps, segments/preconditions, null/empty data, bounded
samples, and error reporting. For the minimum supported Polars version, install
`polars==1.20.0` and rerun the same tests.
