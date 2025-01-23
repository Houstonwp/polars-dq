import polars as pl

from polars_dq.actions import Action
from polars_dq.validation import Validation, ValidationStep


def coalesce_actions(validation_action: Action, agent_action: Action):
    return Action(
        warn_atol=validation_action.warn_atol or agent_action.warn_atol,
        warn_rtol=validation_action.warn_rtol or agent_action.warn_rtol,
        stop_atol=validation_action.stop_atol or agent_action.stop_atol,
        stop_rtol=validation_action.stop_rtol or agent_action.stop_rtol,
        notify_atol=validation_action.notify_atol or agent_action.notify_atol,
        notify_rtol=validation_action.notify_rtol or agent_action.notify_rtol,
    )


class Agent:
    def __init__(
        self,
        table: pl.DataFrame,
        actions: Action = Action(),
        table_name: str | None = None,
        label: str | None = None,
    ):
        self.table_name = table_name
        self.table = table
        self.label = label
        self.actions = actions
        self.validation_set: list[ValidationStep] = []

    def with_validation(self, validation: Validation):
        table = self.table
        if validation.precondition is not None:
            table = table.filter(validation.precondition)

        if validation.segments:
            for name, data in table.group_by(validation.segments, maintain_order=True):
                validation_step = ValidationStep(
                    id=len(self.validation_set) + 1,
                    validation=validation.validation,
                    precondition=validation.precondition,
                    segment_columns=validation.segments,
                    segment_values=name,
                    actions=coalesce_actions(validation.actions, self.actions),
                    label=validation.label,
                    table=data.lazy(),
                    table_name=self.table_name,
                )
                self.validation_set.append(validation_step)

        validation_step = ValidationStep(
            id=len(self.validation_set) + 1,
            validation=validation.validation,
            precondition=validation.precondition,
            actions=coalesce_actions(validation.actions, self.actions),
            label=validation.label,
            table=table.lazy(),
            table_name=self.table_name,
        )
        self.validation_set.append(validation_step)
        return self

    def interrogate(self):
        for validation_step in self.validation_set:
            validation_step.interrogate()
        return self
