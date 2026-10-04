from dataclasses import fields

import polars as pl

from polars_dq.actions import Action
from polars_dq.validation import Validation, ValidationStep


def coalesce_actions(validation_action: Action, agent_action: Action) -> Action:
    """Inherit only unspecified tolerances; zero is a valid override."""
    return Action(
        **{
            field.name: (
                getattr(validation_action, field.name)
                if getattr(validation_action, field.name) is not None
                else getattr(agent_action, field.name)
            )
            for field in fields(Action)
        }
    )


class Agent:
    def __init__(
        self,
        table: pl.DataFrame,
        actions: Action | None = None,
        table_name: str | None = None,
        label: str | None = None,
    ):
        self.table_name = table_name
        self.table = table
        self.label = label
        self.actions = actions if actions is not None else Action()
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
                    eval_active=validation.eval_active,
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
            eval_active=validation.eval_active,
        )
        self.validation_set.append(validation_step)
        return self

    def interrogate(self):
        for validation_step in self.validation_set:
            validation_step.interrogate()
        return self
