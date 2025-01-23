from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import polars as pl

from polars_dq.actions import Action


@dataclass
class Validation:
    validation: pl.Expr
    precondition: pl.Expr | list[pl.Expr] | None = None
    segments: pl.Expr | list[pl.Expr] | None = None
    actions: Action = field(default_factory=Action)
    label: str | None = None


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
    eval_active: bool = False
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
        start_time = datetime.now()
        if self.eval_active:
            if self.table_checked:
                return self

            self.iterrogation_notes = notes

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("always")

                    validation_table = self.table.with_columns(
                        (self.validation).alias("validation")
                    )

                    validation_results = (
                        validation_table.select(
                            pl.len().alias("n"),
                            pl.col("validation").sum().alias("n_passed"),
                            pl.col("validation").not_().sum().alias("n_failed"),
                        )
                        .with_columns(
                            pl.col("n_passed").eq(pl.col("n")).alias("all_passed"),
                            pl.col("n_passed").truediv(pl.col("n")).alias("f_passed"),
                            pl.col("n_failed").truediv(pl.col("n")).alias("f_failed"),
                        )
                        .with_columns(
                            (pl.col("f_failed") > self.actions.warn_rtol)
                            .or_(pl.col("n_failed") > self.actions.warn_atol)
                            .alias("warn"),
                            (pl.col("f_failed") > self.actions.notify_rtol)
                            .or_(pl.col("n_failed") > self.actions.notify_atol)
                            .alias("notify"),
                            (pl.col("f_failed") > self.actions.stop_rtol)
                            .or_(pl.col("n_failed") > self.actions.stop_atol)
                            .alias("stop"),
                        )
                    )

                    self.table_checked = True
                    self.row_sample = (
                        validation_table.filter(pl.col("validation").not_())
                        .head(10)
                        .drop("validation")
                        .collect()
                    )

                    validation_dict = validation_results.collect().to_dicts()[0]
                    self.all_passed = validation_dict["all_passed"]
                    self.n = validation_dict["n"]
                    self.n_passed = validation_dict["n_passed"]
                    self.n_failed = validation_dict["n_failed"]
                    self.f_passed = validation_dict["f_passed"]
                    self.f_failed = validation_dict["f_failed"]
                    self.warn = validation_dict["warn"]
                    self.notify = validation_dict["notify"]
                    self.stop = validation_dict["stop"]

            except Exception as e:
                self.eval_error = True
                self.capture_stack = e
                self.eval_active = False
                return self
            except Warning as w:
                self.eval_warning = True
                self.capture_stack = w
                self.eval_active = False
                return self

            self.table_checked = True

        end_time = datetime.now()
        self.time_processed = end_time - start_time
        return self
