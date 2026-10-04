"""End-to-end checks of evaluated data, rather than validation metadata alone."""

from dataclasses import fields

import polars as pl
import pytest

from polars_dq.actions import Action
from polars_dq.agent import Agent, coalesce_actions
from polars_dq.validation import Validation, ValidationStep

ACTION_FIELDS = [field.name for field in fields(Action)]


def assert_results(step, passed, failed):
    assert step.table_checked is True
    assert step.eval_error is False, step.capture_stack
    assert step.n == passed + failed
    assert step.n_passed == passed
    assert step.n_failed == failed
    assert step.all_passed is (failed == 0)
    assert step.f_passed == pytest.approx(passed / step.n if step.n else 0.0)
    assert step.f_failed == pytest.approx(failed / step.n if step.n else 0.0)
    assert step.time_processed is not None


def test_agent_evaluates_failing_data_by_default():
    table = pl.DataFrame({"a": [-1, 0, 2, 3]})
    agent = Agent(table).with_validation(Validation(pl.col("a") > 0))

    assert agent.interrogate() is agent

    step = agent.validation_set[0]
    assert step.eval_active is True
    assert_results(step, passed=2, failed=2)
    assert step.row_sample.to_dict(as_series=False) == {"a": [-1, 0]}
    assert (step.warn, step.notify, step.stop) == (False, False, False)
    assert all(isinstance(flag, bool) for flag in (step.warn, step.notify, step.stop))


def test_validation_step_is_active_by_default():
    step = ValidationStep(1, pl.DataFrame({"a": [-1]}).lazy(), pl.col("a") > 0, Action())
    assert step.interrogate() is step
    assert_results(step, passed=0, failed=1)


@pytest.mark.parametrize("segments", [None, ["group"]])
def test_explicitly_disabled_validation_skips_all_generated_steps(segments):
    table = pl.DataFrame({"a": [-1, 2], "group": ["x", "y"]})
    agent = Agent(table).with_validation(
        Validation(pl.col("missing") > 0, segments=segments, eval_active=False)
    )
    agent.interrogate()

    assert len(agent.validation_set) == (3 if segments else 1)
    for step in agent.validation_set:
        assert step.eval_active is False
        assert step.table_checked is False
        assert step.eval_error is False
        assert step.n == 0
        assert step.row_sample is None


def test_disabled_step_can_be_enabled_and_successful_step_is_not_repeated():
    step = ValidationStep(
        1, pl.DataFrame({"a": [-1, 2]}).lazy(), pl.col("a") > 0, Action(), eval_active=False
    )
    step.interrogate()
    assert step.table_checked is False
    step.eval_active = True
    step.interrogate(notes="first run")
    assert_results(step, passed=1, failed=1)
    original_time = step.time_processed
    original_sample = step.row_sample
    step.interrogate(notes="second run")
    assert step.iterrogation_notes == "first run"
    assert step.time_processed == original_time
    assert step.row_sample is original_sample


@pytest.mark.parametrize("field", ACTION_FIELDS)
@pytest.mark.parametrize("zero", [0, 0.0])
@pytest.mark.parametrize("agent_value", [None, 0.75])
def test_coalesce_preserves_explicit_zero_without_mutating_actions(field, zero, agent_value):
    validation_action = Action(**{field: zero})
    agent_action = Action(**{field: agent_value})
    result = coalesce_actions(validation_action, agent_action)
    assert getattr(result, field) == zero
    assert getattr(result, field) is not None
    assert getattr(validation_action, field) == zero
    assert getattr(agent_action, field) == agent_value
    assert result is not validation_action and result is not agent_action


@pytest.mark.parametrize("field", ACTION_FIELDS)
def test_zero_override_triggers_action_on_failing_data(field):
    agent = Agent(pl.DataFrame({"a": [-1, 1]}), actions=Action(**{field: 10.0}))
    agent.with_validation(Validation(pl.col("a") > 0, actions=Action(**{field: 0.0})))
    agent.interrogate()
    step = agent.validation_set[0]
    assert_results(step, passed=1, failed=1)
    assert getattr(step.actions, field) == 0.0
    assert getattr(step, field.split("_")[0]) is True


@pytest.mark.parametrize("field", ACTION_FIELDS)
def test_zero_tolerance_does_not_trigger_when_all_rows_pass(field):
    agent = (
        Agent(pl.DataFrame({"a": [1, 2]}))
        .with_validation(Validation(pl.col("a") > 0, actions=Action(**{field: 0.0})))
        .interrogate()
    )
    step = agent.validation_set[0]
    assert_results(step, passed=2, failed=0)
    assert getattr(step, field.split("_")[0]) is False
    assert step.row_sample.height == 0


@pytest.mark.parametrize("level", ["warn", "notify", "stop"])
def test_thresholds_are_strictly_exceeded_and_combined_with_or(level):
    table = pl.DataFrame({"a": [-1, 1, 2, 3]})
    for atol, rtol, expected in [(1, 0.25, False), (0, 0.25, True), (1, 0.0, True)]:
        action = Action(**{f"{level}_atol": atol, f"{level}_rtol": rtol})
        step = (
            Agent(table, actions=action)
            .with_validation(Validation(pl.col("a") > 0))
            .interrogate()
            .validation_set[0]
        )
        assert_results(step, passed=3, failed=1)
        assert getattr(step, level) is expected


def test_precondition_and_segments_evaluate_rows_and_overall_total():
    table = pl.DataFrame(
        {
            "a": [-100, -1, 2, -2, 3],
            "group": ["excluded", "x", "x", "y", "y"],
            "include": [False, True, True, True, True],
        }
    )
    validation = Validation(
        pl.col("a") > 0,
        precondition=pl.col("include"),
        segments=["group"],
        actions=Action(stop_atol=0.0),
    )
    steps = Agent(table).with_validation(validation).interrogate().validation_set
    assert len(steps) == 3
    assert [step.segment_values for step in steps] == [("x",), ("y",), None]
    for step, passed, failed in zip(steps, [1, 1, 2], [1, 1, 2]):
        assert_results(step, passed, failed)
        assert step.stop is True
    assert steps[-1].row_sample["a"].to_list() == [-1, -2]


def test_null_predicates_are_failures_and_appear_in_sample():
    step = (
        Agent(pl.DataFrame({"a": [None, -1, 2]}))
        .with_validation(Validation(pl.col("a") > 0, actions=Action(warn_atol=0.0)))
        .interrogate()
        .validation_set[0]
    )
    assert_results(step, passed=1, failed=2)
    assert step.warn is True
    assert step.row_sample["a"].to_list() == [None, -1]


def test_empty_filtered_data_has_finite_fractions_and_no_actions():
    action = Action(**dict.fromkeys(ACTION_FIELDS, 0.0))
    step = (
        Agent(pl.DataFrame({"a": [-1]}))
        .with_validation(Validation(pl.col("a") > 0, precondition=pl.col("a") > 10, actions=action))
        .interrogate()
        .validation_set[0]
    )
    assert_results(step, passed=0, failed=0)
    assert (step.warn, step.notify, step.stop) == (False, False, False)
    assert step.row_sample.height == 0


def test_failure_sample_is_bounded_and_preserves_source_columns():
    table = pl.DataFrame(
        {
            "a": [-1] * 15,
            "validation": list(range(15)),
            "__polars_dq_validation": ["original"] * 15,
        }
    )
    step = Agent(table).with_validation(Validation(pl.col("a") > 0)).interrogate().validation_set[0]
    assert_results(step, passed=0, failed=15)
    assert step.row_sample.equals(table.head(10))


@pytest.mark.parametrize("predicate", [pl.col("missing") > 0, pl.col("a")])
def test_evaluation_error_is_not_marked_successfully_checked(predicate):
    step = (
        Agent(pl.DataFrame({"a": [1, 2]}))
        .with_validation(Validation(predicate))
        .interrogate()
        .validation_set[0]
    )
    assert step.eval_error is True
    assert isinstance(step.capture_stack, Exception)
    assert step.table_checked is False
    assert step.eval_active is False
    assert step.all_passed is False
    assert step.time_processed is not None


def test_agents_do_not_share_default_actions():
    first = Agent(pl.DataFrame({"a": [1]}))
    second = Agent(pl.DataFrame({"a": [1]}))
    first.actions.warn_atol = 5
    assert second.actions.warn_atol is None


def test_evaluation_error_can_be_corrected_and_explicitly_retried():
    step = ValidationStep(1, pl.DataFrame({"a": [-1, 2]}).lazy(), pl.col("missing") > 0, Action())
    step.interrogate()
    assert step.eval_error is True
    step.validation = pl.col("a") > 0
    step.eval_active = True
    step.interrogate()
    assert_results(step, passed=1, failed=1)
    assert step.capture_stack is None


def test_evaluation_warning_is_recorded_without_losing_results(monkeypatch):
    import warnings

    original_collect = pl.LazyFrame.collect

    def collect_with_warning(frame, *args, **kwargs):
        warnings.warn("test evaluation warning", UserWarning, stacklevel=2)
        return original_collect(frame, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", collect_with_warning)
    step = (
        Agent(pl.DataFrame({"a": [-1, 2]}))
        .with_validation(Validation(pl.col("a") > 0))
        .interrogate()
        .validation_set[0]
    )
    assert_results(step, passed=1, failed=1)
    assert step.eval_warning is True
    assert isinstance(step.capture_stack, UserWarning)
    assert str(step.capture_stack) == "test evaluation warning"
    assert step.eval_active is True


@pytest.mark.parametrize("value", [False, True])
def test_empty_zero_column_input_does_not_gain_rows_from_scalar_predicate(value):
    step = (
        Agent(pl.DataFrame(), actions=Action(warn_atol=0.0))
        .with_validation(Validation(pl.lit(value)))
        .interrogate()
        .validation_set[0]
    )
    assert_results(step, passed=0, failed=0)
    assert step.warn is False
    assert step.row_sample.shape == (0, 0)
