import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from time import perf_counter

import polars as pl

from polars_dq.actions import Action


@dataclass
class Validation:
    validation: pl.Expr
    precondition: pl.Expr | list[pl.Expr] | None = None
    segments: pl.Expr | list[pl.Expr] | None = None
    actions: Action = field(default_factory=Action)
    label: str | None = None
    eval_active: bool = True


@dataclass
class ValidationStep:
    id: int
    table: pl.LazyFrame
    validation: pl.Expr
    actions: Action
    precondition: pl.Expr | list[pl.Expr] | None = None
    segment_columns: pl.Expr | list[pl.Expr] | None = None
    segment_values: tuple | None = None
    label: str | None = None
    table_name: str | None = None
    table_checked: bool = False
    eval_active: bool = True
    eval_error: bool = False
    eval_warning: bool = False
    capture_stack = None
    all_passed: bool = False
    n: int = 0
    n_passed: int = 0
    n_failed: int = 0
    f_passed: float = 0.0
    f_failed: float = 0.0
    warn: bool = False
    notify: bool = False
    stop: bool = False
    row_sample: pl.DataFrame | None = None
    iterrogation_notes: str | None = None
    time_processed: timedelta | None = None

    def interrogate(self, notes: str | None = None):
        """Evaluate once when enabled; retain errors without marking a step checked."""
        if not self.eval_active or self.table_checked:
            return self

        start_time = perf_counter()
        self.iterrogation_notes = notes
        self.eval_error = False
        self.eval_warning = False
        self.capture_stack = None

        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")

                # Keep every source column intact in the failing-row sample.
                result_column = "__polars_dq_validation"
                source_columns = self.table.collect_schema().names()
                while result_column in source_columns:
                    result_column += "_"
                # A scalar expression on a zero-column frame otherwise creates a row.
                source_table = self.table
                internal_columns = [result_column]
                if not source_columns:
                    row_index_column = "__polars_dq_row"
                    source_table = source_table.with_row_index(row_index_column)
                    internal_columns.append(row_index_column)
                validation_table = source_table.with_columns(self.validation.alias(result_column))
                if validation_table.collect_schema()[result_column] != pl.Boolean:
                    raise TypeError("Validation expressions must produce Boolean values")
                # An unknown predicate is not a passing row.
                validation_table = validation_table.with_columns(
                    pl.col(result_column).fill_null(False)
                )

                validation_results = (
                    validation_table.select(
                        pl.len().alias("n"),
                        pl.col(result_column).sum().alias("n_passed"),
                        pl.col(result_column).not_().sum().alias("n_failed"),
                    )
                    .with_columns(
                        pl.col("n_passed").eq(pl.col("n")).alias("all_passed"),
                        pl.when(pl.col("n") > 0)
                        .then(pl.col("n_passed") / pl.col("n"))
                        .otherwise(0.0)
                        .alias("f_passed"),
                        pl.when(pl.col("n") > 0)
                        .then(pl.col("n_failed") / pl.col("n"))
                        .otherwise(0.0)
                        .alias("f_failed"),
                    )
                    .with_columns(
                        _action_trigger(self.actions.warn_atol, self.actions.warn_rtol).alias(
                            "warn"
                        ),
                        _action_trigger(self.actions.notify_atol, self.actions.notify_rtol).alias(
                            "notify"
                        ),
                        _action_trigger(self.actions.stop_atol, self.actions.stop_rtol).alias(
                            "stop"
                        ),
                    )
                )
                validation_dict = validation_results.collect().to_dicts()[0]
                row_sample = (
                    validation_table.filter(pl.col(result_column).not_())
                    .head(10)
                    .drop(internal_columns)
                    .collect()
                )

            self.all_passed = validation_dict["all_passed"]
            self.n = validation_dict["n"]
            self.n_passed = validation_dict["n_passed"]
            self.n_failed = validation_dict["n_failed"]
            self.f_passed = validation_dict["f_passed"]
            self.f_failed = validation_dict["f_failed"]
            self.warn = validation_dict["warn"]
            self.notify = validation_dict["notify"]
            self.stop = validation_dict["stop"]
            self.row_sample = row_sample
            self.eval_warning = bool(caught_warnings)
            if caught_warnings:
                self.capture_stack = caught_warnings[-1].message
            self.table_checked = True

        except Exception as error:  # noqa: BLE001 - evaluation errors are part of the result API
            self.eval_error = True
            self.capture_stack = error
            self.eval_active = False
        finally:
            self.time_processed = timedelta(seconds=perf_counter() - start_time)

        return self


def _action_trigger(atol: float | None, rtol: float | None) -> pl.Expr:
    """Only configured limits participate; a limit is triggered when exceeded."""
    absolute = pl.col("n_failed") > atol if atol is not None else pl.lit(False)
    relative = pl.col("f_failed") > rtol if rtol is not None else pl.lit(False)
    return absolute | relative
