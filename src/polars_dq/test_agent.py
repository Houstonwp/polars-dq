import polars as pl
from polars_dq.agent import Agent, coalesce_actions
from polars_dq.validation import Validation
from polars_dq.actions import Action


def test_with_validation_no_precondition_no_segments():
    table = pl.DataFrame({"a": [1, 2, 3]})
    agent = Agent(
        table=table,
    )
    validation = Validation(validation=pl.col("a").gt(0))

    agent.with_validation(validation)

    assert len(agent.validation_set) == 1
    assert agent.validation_set[0].validation.meta.eq(validation.validation)
    assert agent.validation_set[0].precondition is None
    assert agent.validation_set[0].segment_columns is None
    assert agent.validation_set[0].segment_values is None


def test_with_validation_with_precondition():
    table = pl.DataFrame({"a": [1, 2, 3]})
    agent = Agent(table=table)
    validation = Validation(
        validation=pl.col("a") > 0, precondition=pl.col("a") > 1, actions=Action()
    )

    agent.with_validation(validation)

    assert len(agent.validation_set) == 1
    assert agent.validation_set[0].validation.meta.eq(validation.validation)
    assert agent.validation_set[0].precondition.meta.eq(validation.precondition)


def test_with_validation_with_segments():
    table = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    agent = Agent(table=table)
    validation = Validation(validation=pl.col("a") > 0, segments=["b"], actions=Action())

    agent.with_validation(validation)

    assert len(agent.validation_set) == 3
    assert agent.validation_set[0].segment_columns == ["b"]
    assert agent.validation_set[0].segment_values == ("x",)
    assert agent.validation_set[1].segment_values == ("y",)


def test_with_validation_coalesce_actions():
    table = pl.DataFrame({"a": [1, 2, 3]})
    agent_action = Action(warn_atol=0.1)
    agent = Agent(table=table, actions=agent_action)
    validation_action = Action(warn_rtol=0.2)
    validation = Validation(validation=pl.col("a") > 0, actions=validation_action)

    agent.with_validation(validation)

    assert len(agent.validation_set) == 1
    assert agent.validation_set[0].actions.warn_atol == 0.1
    assert agent.validation_set[0].actions.warn_rtol == 0.2


def test_interrogate():
    table = pl.DataFrame({"a": [1, 2, 3]})
    agent = Agent(table=table)
    validation = Validation(validation=pl.col("a") > 0, actions=Action())

    agent.with_validation(validation)
    agent.interrogate()

    assert len(agent.validation_set) == 1
    assert agent.validation_set[0].validation.meta.eq(validation.validation)
    assert agent.validation_set[0].precondition is None
    assert agent.validation_set[0].segment_columns is None
    assert agent.validation_set[0].segment_values is None


def test_interrogate_with_precondition():
    table = pl.DataFrame({"a": [1, 2, 3]})
    agent = Agent(table=table)
    validation = Validation(
        validation=pl.col("a") > 0, precondition=pl.col("a") > 1, actions=Action()
    )

    agent.with_validation(validation)
    agent.interrogate()

    assert len(agent.validation_set) == 1
    assert agent.validation_set[0].validation.meta.eq(validation.validation)
    assert agent.validation_set[0].precondition.meta.eq(validation.precondition)


def test_interrogate_with_segments():
    table = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    agent = Agent(table=table)
    validation = Validation(validation=pl.col("a") > 0, segments=["b"], actions=Action())

    agent.with_validation(validation)
    agent.interrogate()

    assert len(agent.validation_set) == 3
    assert agent.validation_set[0].segment_columns == ["b"]
    assert agent.validation_set[0].segment_values == ("x",)
    assert agent.validation_set[1].segment_values == ("y",)


def test_interrogate_coalesce_actions():
    table = pl.DataFrame({"a": [1, 2, 3]})
    agent_action = Action(warn_atol=0.1)
    agent = Agent(table=table, actions=agent_action)
    validation_action = Action(warn_rtol=0.2)
    validation = Validation(validation=pl.col("a") > 0, actions=validation_action)

    agent.with_validation(validation)
    agent.interrogate()

    assert len(agent.validation_set) == 1
    assert agent.validation_set[0].actions.warn_atol == 0.1
    assert agent.validation_set[0].actions.warn_rtol == 0.2


def test_coalesce_actions_all_none():
    validation_action = Action()
    agent_action = Action()
    result = coalesce_actions(validation_action, agent_action)

    assert result.warn_atol is None
    assert result.warn_rtol is None
    assert result.stop_atol is None
    assert result.stop_rtol is None
    assert result.notify_atol is None
    assert result.notify_rtol is None


def test_coalesce_actions_validation_overrides():
    validation_action = Action(
        warn_atol=0.1, warn_rtol=0.2, stop_atol=0.3, stop_rtol=0.4, notify_atol=0.5, notify_rtol=0.6
    )
    agent_action = Action()
    result = coalesce_actions(validation_action, agent_action)

    assert result.warn_atol == 0.1
    assert result.warn_rtol == 0.2
    assert result.stop_atol == 0.3
    assert result.stop_rtol == 0.4
    assert result.notify_atol == 0.5
    assert result.notify_rtol == 0.6


def test_coalesce_actions_agent_overrides():
    validation_action = Action()
    agent_action = Action(
        warn_atol=0.1, warn_rtol=0.2, stop_atol=0.3, stop_rtol=0.4, notify_atol=0.5, notify_rtol=0.6
    )
    result = coalesce_actions(validation_action, agent_action)

    assert result.warn_atol == 0.1
    assert result.warn_rtol == 0.2
    assert result.stop_atol == 0.3
    assert result.stop_rtol == 0.4
    assert result.notify_atol == 0.5
    assert result.notify_rtol == 0.6


def test_coalesce_actions_mixed():
    validation_action = Action(warn_atol=0.1, stop_rtol=0.4)
    agent_action = Action(warn_rtol=0.2, stop_atol=0.3, notify_atol=0.5, notify_rtol=0.6)
    result = coalesce_actions(validation_action, agent_action)

    assert result.warn_atol == 0.1
    assert result.warn_rtol == 0.2
    assert result.stop_atol == 0.3
    assert result.stop_rtol == 0.4
    assert result.notify_atol == 0.5
    assert result.notify_rtol == 0.6
